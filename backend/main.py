"""FastAPI adapter exposing the local retriever to index.html."""

from pathlib import Path
import re
import csv
import os
import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from retrieval import Retriever, load_corpus

ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT / "index.html"
TFIDF_MIN_EVIDENCE_SCORE = 0.035
SEMANTIC_MIN_EVIDENCE_SCORE = 0.42
logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    class_level: str | None = None
    subject: str | None = None
    book_id: str | None = None


def build_retriever() -> tuple[list, Retriever | None]:
    include_supplemental = os.environ.get("ED05_INCLUDE_OPENSTAX", "false").lower() in {"1", "true", "yes"}
    chunks = load_corpus(ROOT, include_supplemental=include_supplemental)
    if not chunks:
        return chunks, None
    model_name = os.environ.get("ED05_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    index = Retriever(
        chunks, semantic=True, cache_path=ROOT / "models" / "retrieval_index.npz",
        model_name=model_name,
    )
    if index.semantic_error:
        logger.warning("Semantic embedder unavailable; using TF-IDF fallback: %s", index.semantic_error)
    return chunks, index


chunks, retriever = build_retriever()
PDF_PATHS = {}
manifest = ROOT / "textbook_manifest.csv"
if manifest.is_file():
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            relative_path = (row.get("local_pdf_path") or "").strip()
            if relative_path:
                path = (ROOT / relative_path).resolve()
                if path.is_relative_to(ROOT):
                    PDF_PATHS[row["book_id"]] = path
app = FastAPI(title="ED-05 Textbook Tutor API", version="1.0.0")
allowed_origins = {
    origin.strip()
    for origin in os.environ.get(
        "ED05_CORS_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:5500,http://localhost:5500",
    ).split(",")
    if origin.strip()
}
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(allowed_origins),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(INDEX_HTML)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "indexed_passages": len(chunks),
        "textbooks": len({chunk.book_id for chunk in chunks}),
        "retrieval_backend": f"fastembed:{retriever.model_name}" if retriever and retriever.semantic_model else "tfidf",
    }


@app.get("/textbooks")
def textbooks():
    books = {}
    for chunk in chunks:
        entry = books.setdefault(chunk.book_id, {
            "book_id": chunk.book_id,
            "title": chunk.book_title or chunk.book_id,
            "board": chunk.board,
            "class_level": chunk.class_level,
            "subject": chunk.subject,
            "source_type": chunk.source_type,
        })
    return sorted(books.values(), key=lambda book: (book["class_level"], book["subject"], book["title"]))


@app.post("/ask")
def ask(request: AskRequest):
    if retriever is None:
        raise HTTPException(status_code=503, detail="No textbook passages are indexed yet.")
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Enter a question before searching.")
    started = time.perf_counter()
    results = retriever.search(
        question,
        limit=4,
        class_level=request.class_level,
        subject=request.subject,
        book_id=request.book_id,
    )
    minimum_score = (
        SEMANTIC_MIN_EVIDENCE_SCORE if retriever.semantic_model is not None
        else TFIDF_MIN_EVIDENCE_SCORE
    )
    if not results or results[0].score < minimum_score:
        return {
            "question": question,
            "answer": "I could not find a sufficiently relevant passage in the selected textbooks.",
            "status": "insufficient_evidence",
            "citations": [],
            "warning": None,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    citations = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        page_match = re.fullmatch(r"Page\s+(\d+)", chunk.citation)
        citations.append({
            "book_id": chunk.book_id,
            "book_title": chunk.book_title or chunk.source,
            "class_level": chunk.class_level or None,
            "subject": chunk.subject,
            "chapter": chunk.chapter or (
                chunk.citation.removeprefix("Section: ")
                if chunk.citation.startswith("Section: ") else None
            ),
            "pdf_page": int(page_match.group(1)) if page_match else None,
            "printed_page": None,
            "source_id": f"{chunk.book_id}:{index}",
            "passage": chunk.text,
        })

    warning = None
    if results[0].chunk.source_type not in {"PDF", "PDF OCR"}:
        warning = "Supplemental OpenStax source; this is not a Balochistan Board textbook."
    return {
        "question": question,
        "answer": results[0].chunk.text,
        "status": "answered",
        "citations": citations,
        "warning": warning,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


@app.get("/pdf/{book_id}")
def get_pdf(book_id: str):
    if book_id == "openstax_physics":
        raise HTTPException(status_code=404, detail="This supplemental source is Markdown, not a PDF.")
    path = PDF_PATHS.get(book_id)
    if path and path.is_file():
        return FileResponse(path, media_type="application/pdf", filename=path.name)
    raise HTTPException(status_code=404, detail="The original PDF is not available locally.")
