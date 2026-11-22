"""OCR local textbook PDFs and build a source-aware passage dataset.

The source PDFs are read-only inputs. OCR requires Tesseract (English traineddata
by default); set TESSERACT_CMD to use a non-default executable.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import pymupdf

from retrieval import split_text

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "processed"
TESSERACT_DEFAULTS = (
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
)
CHAPTER_RE = re.compile(r"^\s*(chapter|unit|lesson)\s*[-:]?\s*(\d+(?:\.\d+)*)\s*[:.\-–]?\s*(.*)$", re.I)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_tesseract(configured: str | None) -> str:
    candidate = configured or os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if not candidate:
        candidate = next((str(path) for path in TESSERACT_DEFAULTS if path.is_file()), None)
    if not candidate or not Path(candidate).is_file():
        raise RuntimeError(
            "Tesseract OCR was not found. Install Tesseract OCR or pass --tesseract PATH."
        )
    return str(candidate)


def read_manifest() -> list[dict[str, str]]:
    path = ROOT / "textbook_manifest.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("textbook_manifest.csv contains no books.")
    return rows


def clean_ocr(text: str) -> str:
    text = text.replace("\ufffd", " ").replace("\x0c", " ")
    text = re.sub(r"(?m)^\s*\+?\s*92\s*\d[\d\s-]{8,}\s*$", " ", text)
    text = re.sub(r"(?m)^\s*MDCAT\s+BY\s*FUTURE\s+DOCTORS.*$", " ", text, flags=re.I)
    text = re.sub(r"(?m)^\s*Compiled\s+by\s*:\s*Sir\s+Muneeb\s+Saleem\s*$", " ", text, flags=re.I)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ocr_book(row: dict[str, str], tesseract: str, languages: str, dpi: int) -> dict:
    relative = (row.get("local_pdf_path") or "").strip()
    source = (ROOT / relative).resolve()
    if not relative or not source.is_relative_to(ROOT) or not source.is_file():
        return {"book_id": row.get("book_id", ""), "error": f"PDF missing or unsafe path: {relative}"}

    pages: list[dict] = []
    started = time.time()
    try:
        doc = pymupdf.open(source)
        if doc.needs_pass and not doc.authenticate(""):
            return {"book_id": row["book_id"], "error": "PDF is password protected; empty password rejected."}
        page_count = len(doc)
        scale = dpi / 72
        for page_number, page in enumerate(doc, start=1):
            try:
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
                result = subprocess.run(
                    [tesseract, "stdin", "stdout", "-l", languages, "--psm", "3"],
                    input=pixmap.tobytes("png"),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    timeout=120,
                )
                text = clean_ocr(result.stdout.decode("utf-8", errors="replace"))
                pages.append({"page_number": page_number, "text": text})
            except (subprocess.SubprocessError, OSError) as error:
                pages.append({"page_number": page_number, "text": "", "error": str(error)})
            if page_number % 50 == 0:
                print(f"[{row['book_id']}] OCR {page_number}/{page_count}", flush=True)
        doc.close()
        return {
            "book_id": row["book_id"],
            "row": row,
            "source": relative,
            "sha256": sha256_file(source),
            "page_count": page_count,
            "pages": pages,
            "elapsed_seconds": round(time.time() - started, 2),
        }
    except Exception as error:
        return {"book_id": row.get("book_id", ""), "error": f"{type(error).__name__}: {error}"}


def infer_chapters(pages: list[dict]) -> list[str]:
    current = "Unlabeled section"
    result = []
    for page in pages:
        for line in page["text"].splitlines():
            match = CHAPTER_RE.match(line.strip())
            if match and len(line.strip()) < 140:
                title = match.group(3).strip(" .:-")
                current = f"{match.group(1).title()} {match.group(2)}" + (f": {title}" if title else "")
                break
        result.append(current)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2, help="PDFs OCRed concurrently (default: 2)")
    parser.add_argument("--dpi", type=int, default=144, help="OCR render resolution (default: 144 DPI)")
    parser.add_argument("--languages", default="eng", help="Tesseract languages, e.g. eng or eng+urd")
    parser.add_argument("--tesseract", help="Path to tesseract executable")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 96 <= args.dpi <= 300:
        parser.error("--workers must be 1..8 and --dpi must be 96..300")
    tesseract = find_tesseract(args.tesseract)
    try:
        language_output = subprocess.run(
            [tesseract, "--list-langs"], check=True, capture_output=True, text=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Tesseract is not working: {error.stderr}") from error
    available_languages = set(language_output.splitlines()[1:])
    missing_languages = set(args.languages.split("+")) - available_languages
    if missing_languages:
        raise RuntimeError(
            "Tesseract language data missing: " + ", ".join(sorted(missing_languages))
            + ". Install the trained data before running OCR."
        )

    rows = [row for row in read_manifest() if (row.get("local_pdf_path") or "").strip()]
    if not rows:
        raise RuntimeError("No local PDFs are configured in textbook_manifest.csv.")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(ocr_book, row, tesseract, args.languages, args.dpi): row for row in rows}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result.get("error"):
                print(f"ERROR [{result['book_id']}]: {result['error']}", flush=True)
            else:
                print(
                    f"DONE [{result['book_id']}]: {result['page_count']} pages in "
                    f"{result['elapsed_seconds']}s", flush=True,
                )

    good = sorted((item for item in results if not item.get("error")), key=lambda item: item["book_id"])
    if not good:
        raise RuntimeError("No PDFs could be processed; see errors above.")

    records = []
    seen_content: set[tuple[str, str]] = set()
    page_stats = []
    for book in good:
        row = book["row"]
        chapters = infer_chapters(book["pages"])
        line_frequency = Counter(
            re.sub(r"\W+", " ", line).strip().casefold()
            for page in book["pages"]
            for line in page["text"].splitlines()
            if len(re.sub(r"\W+", " ", line).strip()) >= 16
        )
        repeated_lines = {
            line for line, count in line_frequency.items()
            if count >= max(8, int(len(book["pages"]) * 0.10))
        }
        if repeated_lines:
            for page in book["pages"]:
                page["text"] = "\n".join(
                    line for line in page["text"].splitlines()
                    if re.sub(r"\W+", " ", line).strip().casefold() not in repeated_lines
                ).strip()
        nonempty_pages = sum(bool(page["text"]) for page in book["pages"])
        chars = 0
        for page, chapter in zip(book["pages"], chapters):
            text = page["text"]
            if not text:
                continue
            chars += len(text)
            dedup_key = (row["book_id"], re.sub(r"\W+", " ", text).strip().casefold())
            if dedup_key in seen_content:
                continue
            seen_content.add(dedup_key)
            for chunk_index, passage in enumerate(split_text(text), start=1):
                records.append({
                    "id": f"{row['book_id']}:p{page['page_number']}:c{chunk_index}",
                    "book_id": row["book_id"],
                    "book_title": row["title"],
                    "board": row["board"],
                    "class_level": str(row["class_level"]),
                    "subject": row["subject"],
                    "chapter": chapter,
                    "page_number": page["page_number"],
                    "text": passage,
                    "chunk_id": chunk_index,
                    "source_file": book["source"],
                    "source_type": "PDF OCR",
                })
        page_stats.append({
            "book_id": row["book_id"], "source_file": book["source"],
            "page_count": book["page_count"], "pages_with_text": nonempty_pages,
            "pages_without_text": book["page_count"] - nonempty_pages,
            "ocr_characters": chars, "sha256": book["sha256"],
            "repeated_header_footer_lines_removed": len(repeated_lines),
            "elapsed_seconds": book["elapsed_seconds"],
        })

    if not records:
        raise RuntimeError("OCR produced no nonempty textbook passages.")

    # Keep every book wholly in one split to prevent pages/chunks leaking across sets.
    book_ids = sorted({record["book_id"] for record in records})
    import random
    rng = random.Random(42)
    rng.shuffle(book_ids)
    test_book_count = max(1, round(len(book_ids) * 0.2)) if len(book_ids) > 1 else 0
    test_books = set(book_ids[:test_book_count])
    test_records = [record for record in records if record["book_id"] in test_books]
    train_records = [record for record in records if record["book_id"] not in test_books]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    datasets = {
        "passages.jsonl": records,
        "train.jsonl": train_records,
        "test.jsonl": test_records,
    }
    for name, values in datasets.items():
        with (OUTPUT / name).open("w", encoding="utf-8", newline="\n") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")

    failures = [{"book_id": item.get("book_id"), "error": item["error"]} for item in results if item.get("error")]
    result_by_id = {item.get("book_id"): item for item in results}
    manifest_rows = read_manifest()
    for row in manifest_rows:
        result = result_by_id.get(row.get("book_id"))
        if result:
            row["extraction_status"] = "failed" if result.get("error") else f"ocr_processed ({args.languages})"
    manifest_path = ROOT / "textbook_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    stats = {
        "seed": 42,
        "dataset_kind": "OCR-derived passage corpus; no synthetic QA labels",
        "source_book_count": len(book_ids),
        "passage_count": len(records),
        "train_passage_count": len(train_records),
        "test_passage_count": len(test_records),
        "train_percent": round(100 * len(train_records) / len(records), 2),
        "test_percent": round(100 * len(test_records) / len(records), 2),
        "split_grouping": "whole book_id groups; source documents do not cross splits",
        "train_book_ids": sorted({record["book_id"] for record in train_records}),
        "test_book_ids": sorted(test_books),
        "books": sorted(page_stats, key=lambda item: item["book_id"]),
        "failures": failures,
        "note": "This is a retrieval corpus, not model-training labels. Do not train a model on these passages as if they were supervised QA pairs.",
    }
    (OUTPUT / "dataset_statistics.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    evaluation = {
        "status": "corpus_validation_only",
        "labeled_question_count": 0,
        "recall_at_k": None,
        "mrr": None,
        "reason": "No human-reviewed textbook question/answer evaluation set exists; retrieval quality metrics are not fabricated.",
        "passages": len(records),
        "books_processed": len(good),
        "books_failed": len(failures),
    }
    (OUTPUT / "evaluation_results.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"Built {len(records)} passages from {len(good)} books in {OUTPUT.relative_to(ROOT)}", flush=True)
    print(f"Train: {len(train_records)} ({stats['train_percent']}%), test: {len(test_records)} ({stats['test_percent']}%)", flush=True)


if __name__ == "__main__":
    main()
