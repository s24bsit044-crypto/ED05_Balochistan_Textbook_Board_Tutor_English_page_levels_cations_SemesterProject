# ED-05 Textbook Tutor

A local, citation-first textbook retrieval app. It searches the passages extracted from local PDFs and the explicitly supplemental OpenStax Physics corpus. Answers are retrieved excerpts, not generated or independently verified explanations.

## Current source data

`textbook_manifest.csv` tracks local textbooks and provenance. Eleven local PDFs are present in `data/`; four additional class 11/12 English and Urdu entries have no local file. The PDFs came from secondary mirror listings and their title, edition, language, board approval, and usage rights still need independent verification. Do not treat the manifest description as proof that a PDF is an official edition.

Most PDF pages are scanned images. `build_dataset.py` uses Tesseract OCR and writes derived data under `data/processed/`; it never edits the source PDFs. OCR quality can vary, especially for equations, diagrams, and tables. The current embedding model and default OCR language are English-only. Urdu books require Urdu Tesseract trained data plus a multilingual embedding model before they can be searched reliably.

The 283-file OpenStax College Physics corpus is supplemental, licensed CC BY 3.0 as recorded in its attribution and manifest files. It is not a Balochistan Board textbook and is excluded from the default API index. Set `ED05_INCLUDE_OPENSTAX=true` to include it in searches.

## Install on Windows

Use Python 3.11 or newer. From PowerShell in this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
winget install --id UB-Mannheim.TesseractOCR --exact
```

If PowerShell blocks environment activation, run the venv Python executable directly instead: `\.venv\Scripts\python.exe`.

## Build and validate the corpus

The builder OCRs every local PDF in the manifest. It may take a while because the source books contain thousands of scanned pages.

```powershell
python build_dataset.py --workers 2 --dpi 144 --languages eng
python validate_dataset.py
```

Use `--workers 1` on a low-memory machine. Use `--languages eng+urd` only after installing both Tesseract language packs. The builder writes:

- `data/processed/passages.jsonl`: full page-cited corpus used by the app.
- `data/processed/train.jsonl` and `test.jsonl`: deterministic seed-42 split grouped by whole book to prevent source-document leakage. The exact record percentages are written to the statistics file and may differ from 80/20 because books have different lengths.
- `data/processed/dataset_statistics.json`: OCR counts, split membership, and source hashes.
- `data/processed/evaluation_results.json`: corpus-processing results. Retrieval accuracy metrics remain null until a human-reviewed question set exists.

This is a retrieval system, not a supervised deep learning training project. No synthetic QA labels or model accuracy are claimed. The API uses the pretrained `BAAI/bge-small-en-v1.5` sentence embedding model through FastEmbed/ONNX and caches the vector index under `models/`. The first run downloads the model and encodes local board-book passages; later runs reuse cached vectors when the corpus hash is unchanged. TF-IDF is built only as a fallback if the model cannot be loaded. Train/test passage files are source-aware dataset/evaluation partitions, not neural model training labels; the production app indexes `passages.jsonl` so it can answer from every processed local book.

To validate the source manifest without an OCR rebuild, run `python validate_dataset.py` after a corpus has been built. To add a textbook, obtain a copy you are allowed to use, put its relative path and metadata in `textbook_manifest.csv`, then rebuild and validate.

## Run the app

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

For a repeatable startup, double-click `start_server.cmd`. It checks whether a healthy server is already listening on port 8000, opens the app when it is, and otherwise starts the API. PowerShell users can run `.\start_server.ps1` when script execution is permitted.

Open the app at <http://127.0.0.1:8000/> and the interactive API docs at <http://127.0.0.1:8000/docs>. Keep the server terminal open while using the app. The interface is served by FastAPI; do not open `index.html` with `file://`.

For a separate frontend development server, set `ED05_CORS_ORIGINS` to a comma-separated list of trusted origins before starting FastAPI. Defaults allow localhost ports 8000 and 5500. The frontend accepts `?api=http://127.0.0.1:8000` to override the API address.

The previous Streamlit UI remains available with `streamlit run app.py`.

## API

- `GET /health`: API status, indexed passage count, and distinct source count.
- `GET /textbooks`: source filters for the frontend.
- `POST /ask`: accepts `{"question":"...", "class_level":"11", "subject":"Physics", "book_id":"..."}`; filters are optional. Returns an answer excerpt, evidence status, and citations.
- `GET /pdf/{book_id}`: streams a configured local source PDF.

## Tests

```powershell
python -m unittest discover -s tests
python validate_dataset.py
```
