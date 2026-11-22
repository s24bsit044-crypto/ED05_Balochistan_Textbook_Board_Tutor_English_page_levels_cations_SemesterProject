# ED-05 Textbook Tutor — Model Documentation

## 1. System summary

ED-05 is a local, citation-first textbook question-answering system. It does not generate new explanations with a chat language model. Instead, it finds the most relevant passages from the indexed textbooks and returns the best passage together with supporting citations.

The system is therefore a retrieval-augmented search application:

```text
Textbook PDFs / optional OpenStax Markdown
        ↓
OCR and text extraction
        ↓
Page-aware passage chunks
        ↓
Embedding index (semantic search)
        ↓
Question embedding + cosine similarity
        ↓
Relevant textbook passages and page citations
```

## 2. Model identity

| Item | Configuration |
|---|---|
| Primary retrieval model | `BAAI/bge-small-en-v1.5` through FastEmbed/ONNX |
| Embedding type | Dense text embeddings |
| Similarity measure | Cosine similarity, using normalized vectors |
| Passage size | Up to 1,200 characters |
| Passage overlap | 180 characters |
| Results returned | Up to 4 passages per question |
| Production index | `models/retrieval_index.npz` |
| Fallback model | Scikit-learn TF-IDF with word unigrams and bigrams |
| Answer style | Extractive: the highest-ranked indexed passage |

The embedding model is pretrained. This project does not train or fine-tune a neural model. The generated `train.jsonl` and `test.jsonl` files are source-aware dataset partitions for evaluation and are not supervised question-answer labels.

## 3. Knowledge sources

The default corpus is made from local textbook PDFs listed in `textbook_manifest.csv`. The current indexed corpus contains 6,420 passages from 11 textbooks.

Each passage stores:

- textbook and book ID;
- board, class, and subject;
- source filename;
- page citation;
- chapter, when available;
- source type (`PDF`, `PDF OCR`, or supplemental Markdown);
- extracted passage text.

OpenStax College Physics is available as an optional supplemental source. It is excluded by default and can be enabled with:

```powershell
$env:ED05_INCLUDE_OPENSTAX = "true"
```

Supplemental OpenStax passages are labelled separately and are not presented as Balochistan Board textbook content.

## 4. Data preparation

The dataset builder reads the manifest and extracts text from each configured PDF. Scanned pages are processed with OCR. Text is normalized and split without crossing the original citation unit, normally a PDF page.

The chunking algorithm:

1. collapses repeated whitespace;
2. limits a passage to 1,200 characters;
3. prefers sentence or word boundaries;
4. carries forward 180 characters of overlap;
5. preserves the page number for citation.

Processed passages are stored in `data/processed/passages.jsonl`. The embedding matrix is cached in `models/retrieval_index.npz`, together with a fingerprint of the model name and passage text. If the corpus or model changes, the cache is rebuilt.

## 5. Search process

For a user question, the API performs the following steps:

1. validates the question and optional class, subject, and textbook filters;
2. embeds the question with `BAAI/bge-small-en-v1.5`;
3. normalizes the question vector;
4. computes cosine similarity against all indexed passage vectors;
5. removes passages outside the requested filters;
6. ranks the remaining passages by score;
7. applies an evidence threshold;
8. returns the highest-ranked passage as the answer and up to three additional citations.

The semantic answer threshold is `0.42`. If semantic embeddings are unavailable, the application automatically falls back to TF-IDF and uses a threshold of `0.035`.

## 6. API interface

Start the application with:

```powershell
.\start_server.cmd
```

The main endpoints are:

| Endpoint | Purpose |
|---|---|
| `GET /` | Serves the web interface |
| `GET /health` | Reports server state, passage count, textbook count, and active backend |
| `GET /textbooks` | Returns available textbook filters |
| `POST /ask` | Searches the corpus and returns an answer with citations |
| `GET /pdf/{book_id}` | Streams an available original textbook PDF |
| `GET /docs` | Opens FastAPI interactive API documentation |

Example request:

```json
{
  "question": "What is the function of the cell membrane?",
  "class_level": "11",
  "subject": "Biology",
  "book_id": null
}
```

Example response fields:

```json
{
  "status": "answered",
  "answer": "...retrieved textbook passage...",
  "citations": [
    {
      "book_id": "...",
      "pdf_page": 42,
      "passage": "..."
    }
  ],
  "latency_ms": 72.22
}
```

Possible answer statuses are `answered` and `insufficient_evidence`. A missing corpus returns HTTP 503, and invalid input returns HTTP 422.

## 7. Reliability and recovery

The server startup issue was caused by the semantic search path checking for a TF-IDF matrix that does not exist when semantic embeddings are active. The retriever now initializes both backend attributes consistently and only requires the matrix belonging to the selected backend.

The startup scripts provide a repeatable recovery path:

- `start_server.cmd` can be double-clicked from Windows;
- it checks `/health` first;
- it opens the existing healthy server instead of starting a duplicate;
- otherwise it launches Uvicorn on `127.0.0.1:8000`.

## 8. Limitations

- It is an extractive retrieval system, not a generative tutor.
- OCR errors can affect equations, tables, diagrams, and scanned text.
- The default embedding model is English-oriented; Urdu search requires Urdu OCR data and a multilingual embedding model.
- A high similarity score is not a guarantee that the passage is correct.
- Textbook provenance and usage rights should be independently verified.
- The system does not claim measured answer accuracy until a human-reviewed question set is created.

## 9. Rebuilding and validation

To rebuild the processed corpus after adding or changing PDFs:

```powershell
.\.venv\Scripts\python.exe build_dataset.py --workers 2 --dpi 144 --languages eng
.\.venv\Scripts\python.exe validate_dataset.py
```

To run automated tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

The implementation is primarily in `retrieval.py`, while the FastAPI adapter is in `backend/main.py`.
