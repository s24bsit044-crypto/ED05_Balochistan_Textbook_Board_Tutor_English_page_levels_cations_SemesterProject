"""Collect the openly licensed OpenStax Physics source from its GitHub repository."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
OWNER = "philschatz"
REPOSITORY = "physics-book"
COMMIT = "a7efaa7e221fab1694022d8bb98b83a73ee31d3d"
API_ROOT = f"https://api.github.com/repos/{OWNER}/{REPOSITORY}"
RAW_ROOT = f"https://raw.githubusercontent.com/{OWNER}/{REPOSITORY}/{COMMIT}"
OUT_DIR = ROOT / "data" / "github_oer" / "openstax_physics"
USER_AGENT = "ED05-textbook-tutor/1.0 (educational dataset collection)"
MAX_FILES = 1000
MAX_FILE_BYTES = 2_000_000


def fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"})
    try:
        with urlopen(request, timeout=45) as response:
            payload = response.read(MAX_FILE_BYTES + 1)
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(f"Could not retrieve {url}: {error}") from error
    if len(payload) > MAX_FILE_BYTES:
        raise RuntimeError(f"Refusing oversized response from {url}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=MAX_FILES, help="maximum number of Markdown files to collect")
    args = parser.parse_args()
    if args.limit < 1 or args.limit > MAX_FILES:
        parser.error(f"--limit must be between 1 and {MAX_FILES}")

    tree = json.loads(fetch(f"{API_ROOT}/git/trees/{COMMIT}?recursive=1"))
    if tree.get("truncated"):
        raise RuntimeError("GitHub returned a truncated repository tree; refusing an incomplete dataset.")
    paths = [
        item["path"] for item in tree.get("tree", [])
        if item.get("type") == "blob"
        and item["path"].startswith("contents/")
        and item["path"].lower().endswith(".md")
    ]
    paths = sorted(paths)
    if not paths:
        raise RuntimeError("No Markdown textbook chapters were found in the pinned GitHub revision.")
    if len(paths) > args.limit:
        raise RuntimeError(
            f"Repository has {len(paths)} chapters, exceeding --limit {args.limit}; "
            "raise the limit to avoid silently collecting a partial corpus."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    license_text = fetch(f"{RAW_ROOT}/LICENSE.txt").decode("utf-8")
    (OUT_DIR / "LICENSE.txt").write_text(license_text, encoding="utf-8")
    manifest = OUT_DIR / "dataset_manifest.jsonl"
    existing = {}
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            existing[record["path"]] = record
    records = []
    for path in paths:
        destination = OUT_DIR / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        prior = existing.get(path, {})
        if destination.is_file() and prior.get("commit") == COMMIT and sha256_file(destination) == prior.get("sha256"):
            digest = prior["sha256"]
        else:
            content = fetch(f"{RAW_ROOT}/{path}")
            destination.write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
        records.append({
            "path": path,
            "local_path": destination.relative_to(ROOT).as_posix(),
            "source_url": f"https://github.com/{OWNER}/{REPOSITORY}/blob/{COMMIT}/{path}",
            "commit": COMMIT,
            "license": "CC BY 3.0",
            "sha256": digest,
        })
        time.sleep(0.05)

    manifest.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    attribution = (
        "Supplemental demonstration corpus (not a Balochistan Board textbook)\n"
        "OpenStax College, College Physics, mirrored at philschatz/physics-book.\n"
        f"Pinned GitHub revision: {COMMIT}\n"
        "License: Creative Commons Attribution 3.0 Unported (CC BY 3.0).\n"
        "See LICENSE.txt and dataset_manifest.jsonl for license text, source URLs, and hashes.\n"
    )
    (OUT_DIR / "ATTRIBUTION.txt").write_text(attribution, encoding="utf-8")
    print(f"Collected {len(records)} Markdown chapters into {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
