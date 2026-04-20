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
| **RAG search** | PostgreSQL + pgvector for vector embeddings, with OpenAI embeddings for semantic retrieval over your entire corpus |
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
│   │   ├── vector_store.py        # pgvector via LangChain (langchain-postgres)
│   │   └── document_store.py      # SQLAlchemy CRUD + vector coordination
│   └── analysis/
│       ├── timeline.py            # LLM event extraction
│       ├── fact_extractor.py      # LLM fact extraction
│       └── evidence_analyzer.py   # RAG query, evidence brief, cross-exam plan
├── tests/
│   ├── test_processors.py
│   ├── test_storage.py
│   ├── test_api.py
│   └── test_analysis.py
├── Makefile                       # Common dev tasks (install, dev, test, lint, clean)
└── pyproject.toml                 # Project metadata and dependencies (uv)
```

**Storage:**
- **PostgreSQL** for both structured metadata (timeline events, extracted facts) and vector
  embeddings (via the `pgvector` extension).  A single `DATABASE_URL` is used for both the
  SQLAlchemy ORM and the LangChain `PGVector` store.
- Raw uploaded files are stored in `./data/uploads/`.

---

## Quick Start

### 0. Install uv

If you don't have [uv](https://docs.astral.sh/uv/) installed yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. Start PostgreSQL

Use Docker Compose to spin up a PostgreSQL instance with the `pgvector` extension pre-installed:

```bash
docker compose up -d
```

This starts a PostgreSQL 16 container (image `pgvector/pgvector:pg16`) on port `5432` with the
`legal_assistant` database ready to use.

### 2. Install dependencies

```bash
make install
```

This runs `uv sync --extra dev`, which creates a virtual environment (`.venv/`) and installs
all runtime and development dependencies declared in `pyproject.toml`.

For OCR on images you also need Tesseract installed on your system:

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY (and DATABASE_URL if not using the Docker Compose default)
```

The most important settings are `OPENAI_API_KEY` and `DATABASE_URL`.  Document ingestion
(text extraction) works without `OPENAI_API_KEY`, but all LLM features (timeline extraction,
fact extraction, RAG queries, evidence analysis, cross-examination planning) require it.

### 4. Run the server

```bash
make dev
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
make test
```

Or directly:

```bash
uv run pytest tests/ -v
```

Tests use an in-memory SQLite database and do not require an OpenAI API key.
LLM-dependent tests are automatically skipped when `langchain_core` is not installed.

---

## Makefile Targets

| Target | Description |
|---|---|
| `make install` | Install all runtime + dev dependencies via `uv sync --extra dev` |
| `make dev` | Start the FastAPI dev server with hot-reload on `http://localhost:8000` |
| `make test` | Run the full test suite with pytest |
| `make lint` | Run ruff linting checks on `src/` and `tests/` |
| `make clean` | Remove `__pycache__`, `.pyc` files, and build artifacts |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required for LLM features)* | OpenAI API key |
| `DATABASE_URL` | `postgresql+psycopg://postgres:postgres@localhost:5432/legal_assistant` | PostgreSQL connection string (used for both ORM tables and pgvector embeddings) |
| `UPLOAD_DIR` | `./data/uploads` | Uploaded file storage |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | `gpt-4o` | OpenAI chat model for analysis |
| `TESSERACT_CMD` | `/usr/bin/tesseract` | Path to tesseract binary |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
