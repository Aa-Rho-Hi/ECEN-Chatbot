# TAMU ECE Chatbot — Setup Guide

## Architecture

```
engineering.tamu.edu/electrical
        │
        ▼
   [Crawler]  crawler/crawler.py
        │  extracts text per page
        ▼
   [Chunker]  crawler/chunker.py
        │  semantic splits
        ▼
  [Embedder]  crawler/ingest.py
        │  text-embedding-3-small
        ▼
   [Qdrant]   vector DB (Docker)
        │
        ▼
 [FastAPI]    backend/main.py
   ├─ hybrid retrieval (dense + BM25)
   ├─ cross-encoder re-ranker
   └─ TAMU LLM API (generation)
        │
        ▼
 [Next.js]   frontend/
   └─ streaming chat UI
```

---

## Prerequisites

- Docker + Docker Compose
- Python 3.11+
- Node.js 20+
- TAMU AI API key (from [ai.tamu.edu](https://ai.tamu.edu))
- OpenAI API key (for embeddings, or use TAMU's embedding endpoint if available)

---

## Step 1 — Environment

```bash
cp .env.example .env
```

Edit `.env` and fill in:

| Variable | Description |
|---|---|
| `TAMU_API_URL` | Your TAMU AI gateway URL (e.g. `https://ai.tamu.edu/v1`) |
| `TAMU_API_KEY` | Your TAMU API key |
| `TAMU_MODEL` | Model name exposed by TAMU gateway (e.g. `gpt-4o`) |
| `EMBEDDING_API_KEY` | OpenAI key (or TAMU key if they expose embeddings) |
| `EMBEDDING_API_URL` | Leave as `https://api.openai.com/v1` unless using TAMU embeddings |

---

## Step 2 — Start Qdrant

```bash
docker-compose up qdrant -d
```

Verify it's running: http://localhost:6333/dashboard

---

## Step 3 — Run the Crawler & Ingest

```bash
cd crawler
pip install -r requirements.txt
python ingest.py
```

This will:
1. Crawl up to 500 pages under `engineering.tamu.edu/electrical/`
2. Chunk them semantically
3. Embed with `text-embedding-3-small`
4. Upsert into Qdrant

Takes ~10–20 minutes on first run. Subsequent runs with `--diff` are much faster:

```bash
python ingest.py --diff
```

---

## Step 4 — Start the Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
```

API available at http://localhost:8000
Docs at http://localhost:8000/docs

Test it:
```bash
curl -X POST http://localhost:8000/chat/sync \
  -H "Content-Type: application/json" \
  -d '{"question": "Who are the faculty working on machine learning?"}'
```

---

## Step 5 — Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

---

## Step 6 — Run Everything with Docker

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Qdrant dashboard | http://localhost:6333/dashboard |

---

## Re-indexing

The backend auto-re-indexes at 2 AM daily (configurable via `REINDEX_CRON` in `.env`).

To trigger manually:
```bash
curl -X POST http://localhost:8000/admin/reindex
```

---

## API Reference

### `POST /chat` — Streaming (SSE)
```json
{ "question": "What PhD programs are available?", "section_filter": "academics" }
```
Returns Server-Sent Events. First event is `sources`, then text tokens, then `[DONE]`.

### `POST /chat/sync` — Synchronous
Same request body. Returns:
```json
{
  "answer": "...",
  "sources": [{ "url": "...", "title": "...", "section": "academics" }]
}
```

### Section filters
`people` | `research` | `academics` | `admissions` | `news` | `events` | `about`
