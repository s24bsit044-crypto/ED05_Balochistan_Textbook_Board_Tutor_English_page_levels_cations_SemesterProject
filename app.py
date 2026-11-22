from pathlib import Path
import os

import streamlit as st

from retrieval import Retriever, load_corpus

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="ED-05 Textbook Tutor", page_icon="📘", layout="wide")
st.title("ED-05 · Textbook-Board Tutor")
st.caption("Ask curriculum questions and inspect the exact textbook passages behind each answer.")

with st.sidebar:
    st.header("Your corpus")
    st.write("Board PDFs are read from `local_pdf_path` in `textbook_manifest.csv`.")
    st.write("GitHub OpenStax Physics is supplemental demo material, not a Balochistan Board source.")
    top_k = st.slider("Number of citations", min_value=1, max_value=8, value=4)

@st.cache_resource
def get_index(corpus_root: str) -> tuple[list, Retriever | None]:
    include_supplemental = os.environ.get("ED05_INCLUDE_OPENSTAX", "false").lower() in {"1", "true", "yes"}
    chunks = load_corpus(Path(corpus_root), include_supplemental=include_supplemental)
    if not chunks:
        return chunks, None
    root = Path(corpus_root)
    return chunks, Retriever(
        chunks,
        semantic=True,
        cache_path=root / "models" / "retrieval_index.npz",
        model_name=os.environ.get("ED05_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
    )

chunks, retriever = get_index(str(ROOT))
if retriever is None:
    st.warning(
        "No searchable text was found. Add PDFs to the manifest and run the downloader, "
        "or collect the supplemental GitHub corpus with `python collect_github_dataset.py`."
    )
else:
    board_chunks = sum(chunk.source_type == "PDF" for chunk in chunks)
    github_chunks = sum(chunk.source_type == "GitHub Markdown" for chunk in chunks)
    left, right = st.columns(2)
    left.metric("Indexed PDF passages", board_chunks)
    right.metric("Supplemental GitHub passages", github_chunks)

    question = st.text_input(
        "Ask a question",
        placeholder="For example: How does Newton's second law relate force and acceleration?",
    )
    if question:
        results = retriever.search(question, limit=top_k)
        if not results:
            st.info("No matching passage was found. Try different wording or check that the source is text-searchable.")
        else:
            best = results[0].chunk
            st.subheader("Closest direct answer from the textbook")
            st.info(f'"{best.text}"')
            st.caption(f"Source: {best.source}, {best.citation}")
            st.caption(
                "This is a retrieved textbook excerpt, not a generated paraphrase. "
                "Open the supporting passages below to check the context."
            )
            for number, result in enumerate(results, start=1):
                chunk = result.chunk
                title = f"[{number}] {chunk.subject} · {chunk.citation} · {chunk.source}"
                with st.expander(title, expanded=number == 1):
                    st.caption(
                        f"Board: {chunk.board or 'not specified'}"
                        + (f" · Class {chunk.class_level}" if chunk.class_level else "")
                        + f" · Retrieval score {result.score:.3f}"
                    )
                    st.write(chunk.text)
                    st.markdown(f"**Citation:** {chunk.source}, {chunk.citation}")
