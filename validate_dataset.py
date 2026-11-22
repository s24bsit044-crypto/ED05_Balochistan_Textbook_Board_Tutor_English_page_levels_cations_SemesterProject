"""Validate the processed passage corpus, source files, and grouped split."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
REQUIRED = {
    "id", "book_id", "book_title", "subject", "chapter", "page_number",
    "text", "chunk_id", "source_file",
}


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name}:{line_number}: invalid JSON: {error}") from error
    return records


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    all_path, train_path, test_path = (DATA / name for name in ("passages.jsonl", "train.jsonl", "test.jsonl"))
    if not all_path.is_file() or not train_path.is_file() or not test_path.is_file():
        print("Dataset files are missing. Run build_dataset.py first.")
        return 2
    records = read_jsonl(all_path)
    train = read_jsonl(train_path)
    test = read_jsonl(test_path)
    if not records:
        errors.append("passages.jsonl is empty")

    ids: set[str] = set()
    content_seen: set[tuple[str, str]] = set()
    duplicate_content = 0
    for index, record in enumerate(records, start=1):
        missing = sorted(key for key in REQUIRED if record.get(key) in (None, ""))
        if missing:
            errors.append(f"record {index}: missing fields {', '.join(missing)}")
        identifier = record.get("id")
        if identifier in ids:
            errors.append(f"duplicate record id: {identifier}")
        ids.add(identifier)
        if not str(record.get("text", "")).strip():
            errors.append(f"record {identifier}: empty text")
        try:
            if int(record.get("page_number", 0)) < 1:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"record {identifier}: invalid page_number")
        key = (record.get("book_id", ""), " ".join(str(record.get("text", "")).casefold().split()))
        if key in content_seen:
            duplicate_content += 1
        content_seen.add(key)

    train_ids = {record.get("id") for record in train}
    test_ids = {record.get("id") for record in test}
    if train_ids & test_ids:
        errors.append("record IDs appear in both train and test splits")
    if train_ids | test_ids != ids:
        errors.append("train/test records do not exactly cover passages.jsonl")
    train_books = {record.get("book_id") for record in train}
    test_books = {record.get("book_id") for record in test}
    if train_books & test_books:
        errors.append("source book groups leak across train/test split")
    if duplicate_content:
        warnings.append(f"{duplicate_content} exact duplicate passage texts remain within a book")

    manifest_results = []
    manifest_books_without_pdf = []
    with (ROOT / "textbook_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            relative = (row.get("local_pdf_path") or "").strip()
            if not relative:
                manifest_books_without_pdf.append(row.get("book_id", ""))
                continue
            path = (ROOT / relative).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file():
                errors.append(f"{row.get('book_id')}: configured PDF missing or outside project: {relative}")
                continue
            with path.open("rb") as source:
                signature = source.read(5)
            if path.stat().st_size < 5 or signature != b"%PDF-":
                errors.append(f"{row.get('book_id')}: invalid PDF signature")
                continue
            actual_hash = digest(path)
            expected_hash = (row.get("file_hash") or "").strip().lower()
            if expected_hash and actual_hash != expected_hash:
                errors.append(f"{row.get('book_id')}: SHA-256 mismatch")
            manifest_results.append(row["book_id"])

    report = {
        "valid": not errors,
        "passages": len(records),
        "train_records": len(train),
        "test_records": len(test),
        "train_percent": round(100 * len(train) / len(records), 2) if records else 0,
        "test_percent": round(100 * len(test) / len(records), 2) if records else 0,
        "book_group_leakage": bool(train_books & test_books),
        "configured_local_pdfs": len(manifest_results),
        "manifest_books_without_local_pdf": manifest_books_without_pdf,
        "duplicate_passage_texts": duplicate_content,
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
