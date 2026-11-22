"""Optional downloader for PDF links listed in textbook_manifest.csv.

Only retrieve books whose source and terms you have independently verified.
"""

import csv
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "textbook_manifest.csv"
REPORT = ROOT / "data" / "download_report.json"
HEADERS = {"User-Agent": "ED05-university-project/1.0 (educational use)"}


class DownloadLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href and "/download/" in href:
            self.links.append(href)


def request(url: str, *, timeout: int = 45):
    req = Request(url, headers=HEADERS)
    try:
        return urlopen(req, timeout=timeout)
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(f"Request failed for {url}: {error}") from error


def find_pdf_link(page_url: str) -> str:
    with request(page_url) as response:
        parser = DownloadLinks()
        parser.feed(response.read(5_000_000).decode("utf-8", errors="replace"))
    if not parser.links:
        raise RuntimeError("No download link found (the listing may have changed).")
    return urljoin(page_url, parser.links[0])


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0].keys()) if rows else []
    if not rows:
        raise RuntimeError("The textbook manifest contains no book records.")

    report = {"downloaded": [], "failed": [], "duplicates": []}
    seen: dict[str, str] = {}
    for row in rows:
        target_dir = ROOT / "data" / "raw_pdfs" / f"class_{row['class_level']}"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{row['book_id']}.pdf"
        temporary = target.with_suffix(".pdf.part")
        try:
            pdf_url = find_pdf_link(row["source_url"])
            with request(pdf_url, timeout=120) as response, temporary.open("wb") as output:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    output.write(block)
            with temporary.open("rb") as handle:
                if handle.read(5) != b"%PDF-":
                    raise RuntimeError("Downloaded response is not a PDF.")
            temporary.replace(target)
            digest = hash_file(target)
            if digest in seen:
                report["duplicates"].append([row["book_id"], seen[digest]])
            seen[digest] = row["book_id"]
            row.update(
                local_pdf_path=target.relative_to(ROOT).as_posix(),
                file_hash=digest,
                download_status="downloaded",
            )
            report["downloaded"].append(row["book_id"])
        except (OSError, RuntimeError) as error:
            temporary.unlink(missing_ok=True)
            row["download_status"] = "failed"
            report["failed"].append({
                "book_id": row["book_id"],
                "url": row["source_url"],
                "error": str(error),
            })
        time.sleep(2)

    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
