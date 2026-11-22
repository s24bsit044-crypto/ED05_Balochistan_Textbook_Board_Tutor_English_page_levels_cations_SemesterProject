from pathlib import Path
import os
import re

import streamlit as st

from retrieval import Retriever, load_corpus

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="ED-05 · Balochistan Textbook-Board Tutor",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Professional Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main {
        background-color: #f8fafc;
    }

    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #0284c7 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: #ffffff;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.14);
    }

    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.16);
        backdrop-filter: blur(6px);
        padding: 0.3rem 0.85rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0 0 0.45rem 0;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 1.02rem;
        color: #e2e8f0;
        margin: 0;
        max-width: 760px;
        line-height: 1.55;
    }

    .stat-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    .stat-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.25rem;
    }

    .stat-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0f172a;
    }

    .answer-card {
        background: #ffffff;
        border-left: 5px solid #0284c7;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        margin: 1rem 0 1.4rem 0;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }

    .answer-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.85rem;
        flex-wrap: wrap;
        gap: 0.5rem;
    }

    .mode-pill {
        background: #e0f2fe;
        color: #0369a1;
        font-weight: 700;
        font-size: 0.8rem;
        padding: 0.28rem 0.8rem;
        border-radius: 999px;
    }

    .answer-body {
        font-size: 1.04rem;
        line-height: 1.75;
        color: #1e293b;
    }

    .citation-footer {
        margin-top: 1.1rem;
        padding-top: 0.85rem;
        border-top: 1px dashed #cbd5e1;
        font-size: 0.88rem;
        color: #475569;
    }

    .stTextInput > div > div > input {
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        font-size: 1rem !important;
        border: 1.5px solid #cbd5e1 !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #0284c7 !important;
        box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.15) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def format_answer(results: list, mode: str) -> tuple[str, str]:
    """Build Short, Detailed, or Long textbook-grounded responses."""
    top = results[0].chunk
    primary_source = f"{top.book_title or top.source} · {top.citation} (Class {top.class_level}, {top.subject})"

    if mode == "Short Answer":
        sentences = split_sentences(top.text)
        concise = " ".join(sentences[:2]) if sentences else top.text[:280]
        if len(concise) > 340:
            concise = concise[:337].rsplit(" ", 1)[0] + "..."
        return concise, primary_source

    if mode == "Detailed Answer":
        combined = [top.text]
        if len(results) > 1 and results[1].chunk.book_id == top.book_id:
            second = results[1].chunk
            if second.text not in top.text:
                combined.append(second.text)
        body = "<br><br>".join(combined)
        sources = " | ".join(
            dict.fromkeys(f"{r.chunk.subject} ({r.chunk.citation})" for r in results[:2])
        )
        return body, sources

    # Long Answer (Comprehensive Exam-Ready Format)
    sections = []
    intro_sentences = split_sentences(top.text)
    intro = " ".join(intro_sentences[:2]) if intro_sentences else top.text
    sections.append(f"<strong>1. Core Concept & Definition:</strong><br>{intro}")

    sections.append(f"<strong>2. Primary Textbook Explanation ({top.citation}):</strong><br>{top.text}")

    supporting_points = []
    for res in results[1:]:
        c = res.chunk
        snippet = " ".join(split_sentences(c.text)[:3]) or c.text
        supporting_points.append(
            f"<li><strong>[{c.subject} · {c.citation}]:</strong> {snippet}</li>"
        )

    if supporting_points:
        sections.append(
            "<strong>3. Additional Context & Supporting Details:</strong>"
            f"<ul style='margin-top:0.4rem; padding-left:1.3rem;'>{''.join(supporting_points)}</ul>"
        )

    all_sources = " • ".join(
        dict.fromkeys(f"{r.chunk.source} ({r.chunk.citation})" for r in results)
    )
    return "<br><br>".join(sections), all_sources


@st.cache_resource
def get_index(corpus_root: str) -> tuple[list, Retriever | None]:
    include_supplemental = os.environ.get("ED05_INCLUDE_OPENSTAX", "false").lower() in {"1", "true", "yes"}
    use_semantic = os.environ.get("ED05_USE_SEMANTIC", "true").lower() in {"1", "true", "yes"}
    chunks = load_corpus(Path(corpus_root), include_supplemental=include_supplemental)
    if not chunks:
        return chunks, None
    root = Path(corpus_root)
    return chunks, Retriever(
        chunks,
        semantic=use_semantic,
        cache_path=root / "models" / "retrieval_index.npz",
        model_name=os.environ.get("ED05_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
    )


# Hero Header
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-badge">ED-05 · Curriculum Verified RAG System</div>
        <div class="hero-title">Balochistan Textbook-Board Tutor</div>
        <p class="hero-subtitle">
            Ask any curriculum question and get verified, citation-backed answers directly from official textbooks —
            tailored for quick revision, detailed understanding, or long exam-style preparation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

chunks, retriever = get_index(str(ROOT))

# Sidebar Controls
with st.sidebar:
    st.markdown("### ⚙️ Answer & Filter Settings")

    answer_mode = st.radio(
        "Select Answer Length",
        options=["Short Answer", "Detailed Answer", "Long Answer"],
        index=1,
        help="Choose between a quick summary, a complete textbook explanation, or a multi-source exam answer.",
    )

    default_k = {"Short Answer": 2, "Detailed Answer": 4, "Long Answer": 6}[answer_mode]
    top_k = st.slider("Supporting passages", min_value=1, max_value=8, value=default_k)

    st.divider()
    st.markdown("### 📚 Curriculum Filters")

    subjects = ["All Subjects"] + sorted({c.subject for c in chunks if c.subject})
    selected_subject = st.selectbox("Subject", subjects)

    classes = ["All Classes"] + sorted({c.class_level for c in chunks if c.class_level})
    selected_class = st.selectbox("Class Level", classes)

    st.divider()
    st.caption(
        "Board PDFs are indexed via `textbook_manifest.csv`. "
        "Every answer displays exact page-level citations from the source textbook."
    )

if retriever is None:
    st.warning(
        "No searchable text was found. Add PDFs to the manifest and run the downloader, "
        "or collect the supplemental GitHub corpus with `python collect_github_dataset.py`."
    )
    st.stop()

# Top Metrics Row
board_chunks = sum(chunk.source_type != "GitHub Markdown" for chunk in chunks)
unique_subjects = len({chunk.subject for chunk in chunks if chunk.subject})
engine_label = "Semantic + TF-IDF" if retriever.semantic_model is not None else "TF-IDF Lexical"

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Indexed Textbook Passages</div>
            <div class="stat-value">{board_chunks:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Covered Subjects</div>
            <div class="stat-value">{unique_subjects}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Retrieval Engine</div>
            <div class="stat-value" style="font-size:1.25rem; padding-top:0.35rem;">{engine_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Quick Sample Prompts
if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""

st.markdown("**Try a sample question:**")
q_col1, q_col2, q_col3 = st.columns(3)
if q_col1.button("⚡ Newton's Second Law of Motion", use_container_width=True):
    st.session_state.selected_question = "How does Newton's second law relate force and acceleration?"
if q_col2.button("🧪 Chemical Bonding & Valency", use_container_width=True):
    st.session_state.selected_question = "What is chemical bonding and how do atoms achieve stability?"
if q_col3.button("📖 Photosynthesis Process", use_container_width=True):
    st.session_state.selected_question = "Explain the process and importance of photosynthesis in plants."

# Main Search Input & Inline Mode Selector
search_col, mode_col = st.columns([3, 1])
with search_col:
    question = st.text_input(
        "Ask a curriculum question",
        value=st.session_state.selected_question,
        placeholder="Type your question from Balochistan Board textbooks...",
    )
with mode_col:
    quick_mode = st.selectbox(
        "Response Format",
        options=["Short Answer", "Detailed Answer", "Long Answer"],
        index=["Short Answer", "Detailed Answer", "Long Answer"].index(answer_mode),
    )

if question:
    subject_filter = None if selected_subject == "All Subjects" else selected_subject
    class_filter = None if selected_class == "All Classes" else selected_class

    results = retriever.search(
        question,
        limit=top_k,
        subject=subject_filter,
        class_level=class_filter,
    )

    if not results:
        st.info("No matching passage was found for the selected filters. Try broader wording or reset Subject/Class filters.")
    else:
        formatted_body, citation_summary = format_answer(results, quick_mode)
        mode_icons = {
            "Short Answer": "⚡ Short Answer",
            "Detailed Answer": "📘 Detailed Answer",
            "Long Answer": "📚 Long Answer (Exam Format)",
        }

        st.markdown(
            f"""
            <div class="answer-card">
                <div class="answer-header">
                    <span style="font-weight:700; font-size:1.1rem; color:#0f172a;">Verified Textbook Response</span>
                    <span class="mode-pill">{mode_icons[quick_mode]}</span>
                </div>
                <div class="answer-body">{formatted_body}</div>
                <div class="citation-footer">
                    <strong>📌 Primary Reference:</strong> {citation_summary}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"### 🔍 Supporting Textbook Passages ({len(results)} Citations)")
        for number, result in enumerate(results, start=1):
            chunk = result.chunk
            title = f"[{number}] {chunk.subject} · {chunk.citation} · {chunk.source}"
            with st.expander(title, expanded=(number == 1)):
                st.caption(
                    f"**Board:** {chunk.board or 'Balochistan Textbook Board'}"
                    + (f" · **Class:** {chunk.class_level}" if chunk.class_level else "")
                    + f" · **Relevance Score:** {result.score:.3f}"
                )
                st.write(chunk.text)
                st.markdown(f"**Exact Citation:** `{chunk.source}` — **{chunk.citation}**")
