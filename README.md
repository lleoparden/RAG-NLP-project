# Local RAG System — NLP Final Project

A fully containerized Retrieval-Augmented Generation (RAG) system built with FastAPI, ChromaDB, and Ollama. The system ingests unstructured documents (PDFs, DOCX), embeds them using `sentence-transformers`, stores them in a ChromaDB vector database, and answers natural language queries using a local LLM (gemma:2b via Ollama) — all running inside Docker.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| PDF Parsing | PyMuPDF (fitz) |
| DOCX Parsing | python-docx |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB (PersistentClient) |
| LLM | Ollama (gemma:2b) |
| Containerization | Docker + docker-compose |

---

## Project Structure

```
rag-project/
├── app/
│   ├── ingestion/
│   │   ├── parser.py        # PDF + DOCX parsing
│   │   └── chunker.py       # Text chunking
│   ├── models/
│   │   └── schemas.py       # Pydantic models
│   ├── routers/
│   │   └── query.py         # API endpoints
│   ├── services/
│   │   ├── embedder.py      # Sentence-transformers embeddings
│   │   ├── retriever.py     # ChromaDB vector store
│   │   └── rag_service.py   # RAG pipeline logic
│   └── main.py              # FastAPI entry point
├── chroma_db/               # Persisted vector database
├── data/raw/                # Input documents (PDFs, DOCX)
├── docker-compose.yml
├── Dockerfile
├── ingest.py                # One-time ingestion script
└── requirements.txt
```

---

## Quickstart (Docker — Recommended)

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd rag-project
```

### 2. Start the system
```bash
docker-compose up --build
```

This will:
- Build the FastAPI image
- Start the Ollama container and automatically pull `gemma:2b`
- Mount the pre-ingested `chroma_db/` volume
- Expose the API at `http://localhost:8000`

> First run takes a few minutes while gemma:2b downloads (~1.7GB). Subsequent runs are instant.

### 3. Query the API

**Option A — Swagger UI:**
```
http://localhost:8000/docs
```

**Option B — curl:**
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "what is the candidate's experience", "top_k": 3}'
```

**Option C — Python:**
```python
import requests
r = requests.post("http://localhost:8000/api/query", json={
    "query": "what certifications does the candidate have",
    "top_k": 3
})
print(r.json())
```

---

## API Reference

### `POST /api/query`

Accepts a natural language query and returns an LLM-generated answer with retrieved source chunks.

**Request:**
```json
{
  "query": "what company does the candidate work at",
  "top_k": 3
}
```

**Response:**
```json
{
  "query": "what company does the candidate work at",
  "answer": "The candidate currently works at Instabug, Cairo.",
  "chunks": [
    {
      "text": "Backend Engineer — AI Products\nInstabug, Cairo (Remote-hybrid)...",
      "source": "test_cv_messy.pdf",
      "page": 1,
      "score": 0.3389
    }
  ]
}
```

### `GET /api/health`

Returns API health status.

**Response:**
```json
{ "status": "ok" }
```

---

## Running Locally (Without Docker)

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed and running

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Pull the LLM
```bash
ollama pull gemma:2b
```

### 3. Ingest documents
Place your PDFs/DOCX files in `data/raw/`, then run:
```bash
python ingest.py
```

### 4. Start the server
```bash
python -m uvicorn app.main:app --reload
```

API available at `http://127.0.0.1:8000`

---

## Chunking Strategy

Documents are chunked using `RecursiveCharacterTextSplitter` with:
- **Chunk size:** 500 tokens
- **Overlap:** 50 tokens

**Why 500 tokens?** Large enough to preserve semantic context within a single chunk (a full paragraph or work experience entry), but small enough that retrieved chunks remain focused and don't dilute the LLM prompt with irrelevant content.

**Why 50 token overlap?** Prevents information loss at chunk boundaries — key facts that span two chunks (e.g., a job title on one line, the company name on the next) are still captured in at least one chunk.

---

## Embedding Model

**Model:** `all-MiniLM-L6-v2` (sentence-transformers)

- Lightweight (80MB) — runs on CPU without a GPU
- Optimized for semantic similarity tasks
- Produces 384-dimensional normalized vectors suitable for cosine similarity search in ChromaDB
- Strong performance on English document retrieval benchmarks

---

## Adding New Documents

1. Place new PDF or DOCX files in `data/raw/`
2. Re-run ingestion:
```bash
# Locally
python ingest.py

# Inside Docker
docker-compose exec rag-api python ingest.py
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint |

In Docker, `OLLAMA_URL` is automatically set to `http://ollama:11434/api/generate` via `docker-compose.yml`.
