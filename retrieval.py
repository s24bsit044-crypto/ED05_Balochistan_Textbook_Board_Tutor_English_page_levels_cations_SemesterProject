"""Page-aware, local retrieval helpers for the ED-05 textbook tutor."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    citation: str
    book_id: str
    board: str
    class_level: str
    subject: str
    source_type: str
    book_title: str = ""
    chapter: str = ""


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


def split_text(text: str, max_chars: int = 1200, overlap: int = 180) -> list[str]:
    """Split text at sentence/word boundaries without crossing its citation unit."""
    if max_chars <= 0 or overlap < 0 or overlap >= max_chars:
        raise ValueError("Require max_chars > overlap >= 0.")
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    chunks = []
    start = 0
    while start < len(normalized):
        end = min(start + max_chars, len(normalized))
        if end < len(normalized):
            boundary = max(normalized.rfind(". ", start, end), normalized.rfind(" ", start, end))
            if boundary > start + max_chars // 2:
                end = boundary + 1
        value = normalized[start:end].strip()
        if value:
            chunks.append(value)
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunks_from_pdf(
    path: Path,
    *,
    book_id: str,
    book_title: str,
    board: str,
    class_level: str,
    subject: str,
) -> list[Chunk]:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("PDF support requires pypdf; install dependencies from requirements.txt.") from error

    chunks = []
    reader = PdfReader(str(path))
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for text_chunk in split_text(text):
            chunks.append(Chunk(
                text=text_chunk,
                source=path.name,
                citation=f"Page {page_number}",
                book_id=book_id,
                board=board,
                class_level=str(class_level),
                subject=subject,
                source_type="PDF",
                book_title=book_title,
            ))
    return chunks


def chunks_from_markdown(path: Path, source: str | None = None) -> list[Chunk]:
    """Index GitHub OER by section heading; Markdown has no reliable PDF page numbers."""
    text = path.read_text(encoding="utf-8")
    section = path.stem
    chunks = []
    section_lines = []

    def math_text(element: ET.Element) -> str:
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "annotation":
            return ""
        if tag in {"mi", "mn", "mo", "mtext"}:
            return (element.text or "").strip()
        values = [value for child in element for value in [math_text(child)] if value]
        if tag == "mfrac" and len(values) >= 2:
            return f"({values[0]}) / ({values[1]})"
        if tag == "msup" and len(values) >= 2:
            return f"{values[0]}^{values[1]}"
        if tag == "msub" and len(values) >= 2:
            return f"{values[0]}_{values[1]}"
        if tag == "mover" and values:
            return values[0] + "̅"
        return " ".join(values)

    def readable_math(match: re.Match) -> str:
        try:
            root = ET.fromstring(match.group(0))
            semantics = next((child for child in root if child.tag.rsplit("}", 1)[-1] == "semantics"), root)
            visual = next((child for child in semantics if child.tag.rsplit("}", 1)[-1] != "annotation"), semantics)
            return f" {math_text(visual)} "
        except ET.ParseError:
            fallback = re.sub(r"<[^>]*>", " ", match.group(0))
            return " " + re.sub(r"\s+", " ", fallback).strip() + " "

    def add_section() -> None:
        body = "\n".join(section_lines)
        body = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", body)
        body = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", body)
        body = re.sub(r"<math\b[^>]*>.*?</math>", readable_math, body, flags=re.I | re.S)
        body = re.sub(r"\{:\s*[^}]*\}", " ", body)
        body = re.sub(r"<[^>]+>", " ", body)
        body = html.unescape(body)
        body = re.sub(r"[`*_>#]", " ", body)
        for text_chunk in split_text(body):
            chunks.append(Chunk(
                text=text_chunk,
                source=source or path.name,
                citation=f"Section: {section}",
                book_id="openstax_physics",
                board="OpenStax (supplemental)",
                class_level="",
                subject="Physics",
                source_type="GitHub Markdown",
                book_title="OpenStax College Physics (supplemental)",
            ))

    for line in text.splitlines():
        heading = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if heading:
            add_section()
            section = heading.group(1).strip()
            section_lines = [heading.group(1)]
        else:
            section_lines.append(line)
    add_section()
    return chunks


def load_corpus(root: Path, *, include_supplemental: bool = True) -> list[Chunk]:
    """Load local manifest PDFs and the optional downloaded GitHub OER corpus."""
    chunks: list[Chunk] = []
    processed = root / "data" / "processed" / "passages.jsonl"
    if processed.is_file():
        with processed.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                chunks.append(Chunk(
                    text=record["text"],
                    source=record["source_file"],
                    citation=f"Page {record['page_number']}",
                    book_id=record["book_id"],
                    board=record["board"],
                    class_level=str(record["class_level"]),
                    subject=record["subject"],
                    source_type=record.get("source_type", "PDF OCR"),
                    book_title=record["book_title"],
                    chapter=record.get("chapter", ""),
                ))
        github_root = root / "data" / "github_oer" / "openstax_physics" / "contents"
        if include_supplemental and github_root.is_dir():
            for path in sorted(github_root.rglob("*.md")):
                chunks.extend(chunks_from_markdown(path, source=path.relative_to(root).as_posix()))
        return chunks

    manifest = root / "textbook_manifest.csv"
    if manifest.exists():
        import csv

        with manifest.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                relative_path = (row.get("local_pdf_path") or "").strip()
                if not relative_path:
                    continue
                path = root / relative_path
                if not path.is_file():
                    continue
                chunks.extend(chunks_from_pdf(
                    path,
                    book_id=row["book_id"],
                    book_title=row["title"],
                    board=row["board"],
                    class_level=row["class_level"],
                    subject=row["subject"],
                ))
    github_root = root / "data" / "github_oer" / "openstax_physics" / "contents"
    if include_supplemental and github_root.is_dir():
        for path in sorted(github_root.rglob("*.md")):
            chunks.extend(chunks_from_markdown(
                path,
                source=path.relative_to(root).as_posix(),
            ))
    return chunks


class Retriever:
    def __init__(
        self,
        chunks: Iterable[Chunk],
        *,
        semantic: bool = False,
        cache_path: Path | None = None,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.chunks = list(chunks)
        self.model_name = model_name
        self.semantic_model = None
        self.semantic_matrix = None
        # Keep both index attributes defined for every backend.  The semantic
        # path does not build a TF-IDF matrix, but search still needs a stable
        # object shape when deciding which backend is active.
        self.vectorizer = None
        self.matrix = None
        self.semantic_error = None
        if semantic and self.chunks:
            try:
                self._load_semantic_index(cache_path, model_name)
            except Exception as error:
                self.semantic_error = f"{type(error).__name__}: {error}"
                self._build_tfidf_index()
        else:
            self._build_tfidf_index()

    def _build_tfidf_index(self) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b[\w'-]+\b",
        )
        self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in self.chunks]) if self.chunks else None

    def _load_semantic_index(self, cache_path: Path | None, model_name: str) -> None:
        from fastembed import TextEmbedding

        fingerprint = hashlib.sha256()
        fingerprint.update(model_name.encode("utf-8"))
        for chunk in self.chunks:
            fingerprint.update(b"\0")
            fingerprint.update(chunk.text.encode("utf-8"))
        fingerprint_value = fingerprint.hexdigest()
        model_cache = cache_path.parent / "fastembed" if cache_path else None
        if cache_path and cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as cached:
                if str(cached["fingerprint"].item()) == fingerprint_value:
                    self.semantic_matrix = cached["vectors"].astype(np.float32, copy=False)
        self.semantic_model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(model_cache) if model_cache else None,
            threads=2,
        )
        if self.semantic_matrix is None:
            vectors = np.asarray(list(self.semantic_model.embed(
                [chunk.text for chunk in self.chunks], batch_size=64, parallel=1,
            )), dtype=np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            self.semantic_matrix = vectors / np.maximum(norms, 1e-12)
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_path, fingerprint=fingerprint_value, vectors=self.semantic_matrix)

    def search(
        self,
        query: str,
        *,
        limit: int = 4,
        class_level: str | None = None,
        subject: str | None = None,
        book_id: str | None = None,
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Enter a question before searching.")
        if limit < 1:
            raise ValueError("limit must be positive.")
        if not self.chunks:
            return []
        if self.semantic_model is not None and self.semantic_matrix is not None:
            query_vector = np.asarray(next(self.semantic_model.query_embed(query)), dtype=np.float32)
            query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
            scores = self.semantic_matrix @ query_vector
        else:
            if self.vectorizer is None or self.matrix is None:
                return []
            scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        eligible = [
            index for index, chunk in enumerate(self.chunks)
            if (class_level is None or chunk.class_level == str(class_level))
            and (subject is None or chunk.subject == subject)
            and (book_id is None or chunk.book_id == book_id)
        ]
        ranked = sorted(((index, scores[index]) for index in eligible), key=lambda item: item[1], reverse=True)
        return [
            SearchResult(self.chunks[index], float(score))
            for index, score in ranked[:limit]
            if score > 0
        ]
