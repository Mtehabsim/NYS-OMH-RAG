# NYS OMH Policy RAG Application

A lightweight Retrieval-Augmented Generation (RAG) pipeline for querying New York State Office of Mental Health (OMH) policy manuals. Ask natural language questions, get answers grounded in the actual documents with citations.

## Design Decisions

- **`pdfplumber`** over PyPDF2: supports spatial bounding-box cropping to strip repetitive headers/footers that pollute embeddings. Validated empirically against all 4 PDFs — see the [LLD document](lld.md) for details.
- **`ChromaDB`** over FAISS/Qdrant: runs natively in Python with zero infrastructure. Persists to a local SQLite folder — no Docker required.
- **`Gemini 3.1 Flash Lite`** at `temperature=0.2`: free tier (no credit card), fast, and deterministic for grounded Q&A.
- **Decoupled ingestion/inference**: ingestion runs once (~15s), queries are real-time (~2s). Mirrors production ETL patterns.

## Project Structure

```
├── src/
│   ├── ingest.py         # Entry point: build the vector database
│   ├── rag_cli.py         # Entry point: interactive CLI
│   ├── evaluate.py        # Entry point: run evaluation harness
│   ├── constants.py       # All configuration constants
│   ├── models.py          # Pydantic data models
│   ├── prompts.py         # System prompt templates
│   ├── document_fetcher.py# PDF download logic
│   ├── inference.py       # Retrieval + generation engine
│   └── utils.py           # Environment and validation helpers
├── tst/                   # Unit tests (fully mocked)
└── data/                  # Source PDFs (ignored)
```

## Setup

### Prerequisites
- Python 3.9+
- A Google Gemini API key (free — no credit card required)

### Get Your API Key (Free)
1. Go to [aistudio.google.com](https://aistudio.google.com/)
2. Sign in with any Google account
3. Click **"Get API key"** in the left sidebar
4. Click **"Create API key"** and select a project
5. Copy the generated key

### Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### API Key Configuration
```bash
cp .env.example .env
# Edit .env and add your key: GOOGLE_API_KEY=your_key_here
```

## Usage

### 1. Build the Database
```bash
python -m src.ingest
```
Downloads the 4 OMH PDFs (or uses local copies), cleans them, and stores embeddings in `.chroma_db/`. Takes ~15 seconds.

### 2. Query
```bash
python -m src.rag_cli
python -m src.rag_cli --verbose   # also shows retrieved chunks
```

**Example questions:**
- "Does the email policy apply to student interns?"
- "Can clients take shared iPads into their private rooms?"
- "Who is the policy owner for the Acceptable Use of Internet directive?"
- "What happens if someone violates the internet policy?"

### 3. Evaluate
```bash
python -m src.evaluate           # run 7 ground-truth Q&A cases
python -m src.evaluate --verbose  # show per-case details
```

## Running Tests
```bash
PYTHONPATH=. pytest tst/ -v
```
Tests are fully mocked — no API key or network access required.
