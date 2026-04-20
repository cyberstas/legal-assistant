# Legal Assistant – Backend Processing Engine

A Python/FastAPI backend for processing, organising, and analysing family court legal documents.
The system ingests documents of many types, extracts text, embeds them into a vector store for
RAG-powered semantic search, and uses an LLM to automatically extract timeline events, key facts,
and generate evidence briefs and cross-examination plans.

---

## Features

| Capability | Details |
|---|---|
| **Document ingestion** | PDFs, images (OCR), emails (`.eml`), plain text, transcripts, police reports, iMessage exports |
| **RAG search** | ChromaDB vector store + OpenAI embeddings for semantic retrieval over your entire corpus |
| **Timeline builder** | LLM-powered extraction of dated events from every document, stored and queryable |
| **Fact extractor** | Discrete legal facts extracted per document, categorised by topic and relevance |
| **Evidence analyser** | Gather and synthesise evidence for a specific argument or motion topic |
| **Cross-exam planner** | Generate sequenced cross-examination questions with supporting evidence for a witness |
| **REST API** | Full FastAPI backend with auto-generated OpenAPI/Swagger docs |

---

## Architecture

```
legal-assistant/
├── src/
│   ├── config.py                  # Settings loaded from .env
│   ├── main.py                    # FastAPI application
│   ├── models/
│   │   ├── schemas.py             # SQLAlchemy ORM + Pydantic schemas
│   │   └── database.py            # DB engine / session factory
│   ├── processors/
│   │   ├── dispatcher.py          # Routes files to correct processor
│   │   ├── pdf_processor.py       # PyMuPDF + pdfplumber
│   │   ├── image_processor.py     # Pillow + pytesseract OCR
│   │   ├── email_processor.py     # .eml parsing
│   │   └── text_processor.py      # Plain text / transcripts
│   ├── storage/
│   │   ├── vector_store.py        # ChromaDB via LangChain
│   │   └── document_store.py      # SQLAlchemy CRUD + vector coordination
│   └── analysis/
│       ├── timeline.py            # LLM event extraction
│       ├── fact_extractor.py      # LLM fact extraction
│       └── evidence_analyzer.py   # RAG query, evidence brief, cross-exam plan
└── tests/
    ├── test_processors.py
    ├── test_storage.py
    ├── test_api.py
    └── test_analysis.py
```

**Storage:**
- **SQLite** (default) or any SQLAlchemy-compatible DB for structured metadata, timeline events,
  and extracted facts.
- **ChromaDB** (persistent on disk) for vector embeddings used in semantic search.
- Raw uploaded files are stored in `./data/uploads/`.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For OCR on images you also need Tesseract installed on your system:

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

The most important setting is `OPENAI_API_KEY`.  Document ingestion (text extraction) works
without it, but all LLM features (timeline extraction, fact extraction, RAG queries, evidence
analysis, cross-examination planning) require it.

### 3. Run the server

```bash
uvicorn src.main:app --reload
```

The API is now running at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

---

## API Reference

### Upload a document

```http
POST /documents/upload
Content-Type: multipart/form-data

file: <binary>
doc_type_override: pdf|image|email|text|transcript|police_report|imessage  (optional)
```

Supported file types: `.pdf`, `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`, `.eml`, `.txt`, `.csv`

The document type is auto-detected from the filename if `doc_type_override` is not provided
(e.g. a file named `hearing_transcript.pdf` will be typed as `transcript`).

### List / get documents

```http
GET /documents                      # list all (filter by ?doc_type=&status=)
GET /documents/{id}                 # full detail including extracted text
DELETE /documents/{id}              # delete document + vectors
GET /documents/{id}/timeline        # events extracted from this document
GET /documents/{id}/facts           # facts extracted from this document
```

### Timeline

```http
GET /timeline                       # full chronological timeline
POST /timeline                      # filtered timeline
{
  "document_ids": [1, 2, 3],        # optional: restrict to documents
  "start_date": "2023-01-01",       # optional ISO date
  "end_date": "2024-12-31",
  "category": "hearing"             # optional category filter
}
```

### Analysis (requires OpenAI API key)

#### Free-form RAG query

```http
POST /analysis/query
{
  "query": "When did the last custody exchange take place?",
  "top_k": 5,
  "doc_type_filter": "transcript"   # optional
}
```

#### Gather evidence for a motion

```http
POST /analysis/evidence
{
  "topic": "repeated violations of the custody order",
  "top_k": 10
}
```

Returns a structured evidence brief with supporting facts, weaknesses, and presentation strategy.

#### Plan cross-examination

```http
POST /analysis/cross-examination
{
  "witness_name": "Jane Doe",
  "topics": [
    "missed visitation appointments",
    "communications with children",
    "substance abuse incidents"
  ]
}
```

Returns a sequenced cross-examination plan with questions, expected answers, and document citations.

#### All extracted facts

```http
GET /analysis/facts?category=custody&relevance=high
```

---

## Supported Document Types

| File type | How processed |
|---|---|
| `.pdf` | PyMuPDF (text layer) with pdfplumber fallback |
| `.jpg` / `.png` / `.tiff` / etc. | Pillow + pytesseract OCR |
| `.eml` | mail-parser with stdlib `email` fallback |
| `.txt` / `.csv` | Direct text read, split into page-sized chunks |
| Transcripts (PDF/TXT) | Same as PDF/text; LLM extracts hearing events |
| Police reports (PDF) | Same as PDF; LLM extracts incidents and facts |
| iMessage exports (PDF) | Same as PDF; LLM extracts communication timeline |

---

## Running Tests

```bash
pytest tests/ -v
```

Tests use an in-memory SQLite database and do not require an OpenAI API key.
LLM-dependent tests are automatically skipped when `langchain_core` is not installed.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required for LLM features)* | OpenAI API key |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | ChromaDB storage directory |
| `UPLOAD_DIR` | `./data/uploads` | Uploaded file storage |
| `DATABASE_URL` | `sqlite:///./data/legal_assistant.db` | SQLAlchemy DB URL |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | `gpt-4o` | OpenAI chat model for analysis |
| `TESSERACT_CMD` | `/usr/bin/tesseract` | Path to tesseract binary |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
