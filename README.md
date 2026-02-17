# Chat with Your Book — v2 (Text + Images + Eval)

This project allows you to **upload a PDF or textbook** and interact with it using both **text** and **images**. It retrieves relevant passages and diagrams using **multi-modal search** and processes everything locally with **Ollama**.

**v2 adds**: observability tracing, hierarchical chunking, conversation memory, and an LLM-as-Judge evaluation agent.

---

## Features

### Core
- **Upload PDF/TXT** from the browser.
- **Automatic ingestion**: text chunking + CLIP image embeddings.
- **Interactive chat** with multi-turn conversation memory.
- **Multi-modal retrieval**: text + image search.
- **Runs locally** using Ollama (llama3.1:8b).
- **One-click Docker Compose deployment**.

### v2 Improvements

#### Observability
- **Structured JSON logging** across all services.
- **Span-based tracing** — every request generates a trace with timing for each phase (retrieval, LLM, eval).
- **Metrics dashboard** — avg/p95 latency, chunk counts, similarity scores.
- **Span waterfall view** — visualise where time is spent per request.
- Endpoints: `GET /traces`, `GET /traces/{id}`, `GET /metrics`.

#### Better Chunking
- **Hierarchical chunking**: parent chunks (2000 chars) + child chunks (500 chars).
- **Semantic-aware splitting** — respects paragraph and section boundaries.
- **Section header detection** — auto-detects chapters and section titles.
- **Enriched metadata** — chunk_id, parent_id, section, chunk_type on every chunk.
- **200-char overlap** to avoid mid-sentence cuts.

#### Better Context Window
- **Sliding-window conversation memory** (configurable, default 10 turns).
- **History summarisation** — older turns are compressed via LLM to save tokens.
- **Parent-chunk expansion** — when a small child chunk matches, the full parent is included for richer context.
- **Context budget** — total prompt capped at 12,000 chars (configurable) so the model never overflows.
- **Augmented prompt builder** — assembles history + context + question intelligently.

#### Eval Agent / Judge
- **LLM-as-a-Judge** evaluates every response on four dimensions:
  - **Faithfulness** — does the answer stick to the retrieved context?
  - **Relevance** — does it address the user's question?
  - **Completeness** — does it cover key information from context?
  - **Hallucination-free** — does it avoid stating ungrounded facts?
- Scores are **0.0 – 1.0**, displayed inline as colored badges.
- **Eval Dashboard** — aggregate quality metrics across all queries.
- **Manual evaluation** — paste any Q/A pair to test the judge.
- **Batch evaluation** API for testing at scale.
- Uses the same local Ollama model — **zero extra cost**.

---

## Architecture

```
┌──────────────┐     ┌─────────────────────────────────────────────────┐
│  Streamlit   │────▶│               Agent API (v2)                   │
│   UI (v2)    │     │  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│              │     │  │ Context  │  │ Observ-  │  │  Eval Agent  │  │
│ • Chat       │     │  │ Window   │  │ ability  │  │  (Judge)     │  │
│ • Dashboard  │     │  │ Manager  │  │ Tracer   │  │              │  │
│ • Eval View  │     │  └────┬─────┘  └────┬─────┘  └──────┬───────┘  │
└──────────────┘     │       │             │               │          │
                     │  ┌────▼─────────────▼───────────────▼───────┐  │
                     │  │          LangChain Agent                 │  │
                     │  └────┬─────────────────────────────┬───────┘  │
                     └───────┼─────────────────────────────┼──────────┘
                             │                             │
                     ┌───────▼───────┐           ┌─────────▼────────┐
                     │ mcp_vectordb  │           │ mcp_image_retriever│
                     │ (text search) │           │ (CLIP search)     │
                     └───────┬───────┘           └─────────┬────────┘
                             │                             │
                     ┌───────▼─────────────────────────────▼────────┐
                     │                   Neo4j                      │
                     │  • Text chunks (parent + child)              │
                     │  • Image embeddings (CLIP)                   │
                     └──────────────────────────────────────────────┘
                                           │
                     ┌─────────────────────▼──────────────────────┐
                     │              Ollama (local LLM)            │
                     │  • llama3.1:8b (generation + eval judge)   │
                     └────────────────────────────────────────────┘
```

---

## Services Overview

| Service             | Tech Stack                | Purpose                           |
|---------------------|---------------------------|-----------------------------------|
| ui                  | Streamlit                 | Chat + Observability + Eval UI    |
| agent_api           | FastAPI + LangChain       | Orchestration, tracing, eval      |
| mcp_vectordb        | FastAPI + Neo4j/Chroma    | Text retrieval with scores        |
| mcp_image_retriever | FastAPI + CLIP + Neo4j    | Image retrieval                   |
| neo4j               | Neo4j 5.20                | Vector + graph storage            |
| ingest_book         | Python + LangChain + CLIP | Hierarchical chunking + indexing  |

### New Modules

| Module              | Description                                          |
|---------------------|------------------------------------------------------|
| `observability.py`  | Structured logging, span tracing, metrics store      |
| `chunking.py`       | Hierarchical parent-child chunking with metadata     |
| `context_window.py` | Conversation memory, prompt builder, summarisation   |
| `eval_agent.py`     | LLM-as-Judge evaluation (4 quality dimensions)       |

---

## Installation

### 1. Prerequisites
- Docker: https://docs.docker.com/get-docker/
- Ollama installed locally:
```bash
ollama pull llama3.1:8b
```

### 2. Clone the repo
```bash
git clone https://github.com/Shreyas2409/book-rag.git
cd book-rag
```

### 3. Create .env file
```bash
cp .env.example .env
```

Key settings in `.env`:
```bash
NEO4J_URI=neo4j://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASS=pass
AGENT_API=http://localhost:7005
EVAL_ENABLED=true              # enable/disable eval judge
MAX_HISTORY_TURNS=10           # conversation memory window
CONTEXT_BUDGET_CHARS=12000     # max context chars in prompt
LOG_LEVEL=INFO                 # DEBUG for verbose logging
```

### 4. Start all services
```bash
docker compose -f docker.yaml up --build
```

---

## Ingesting a Book

Upload via the UI sidebar, or via terminal:
```bash
docker compose -f docker.yaml run ingest_book python ingestion.py /books/mybook.pdf
```

The improved ingestion pipeline will:
1. Load the PDF/TXT
2. Create **hierarchical chunks** (parent 2000 chars + child 500 chars)
3. Detect **section headers** and enrich metadata
4. Store everything in Neo4j (or Chroma fallback)
5. Extract + embed images with CLIP

---

## Chatting

Visit: **http://localhost:8501**

The chat now supports:
- Multi-turn conversations with memory
- Inline eval scores (colored badges) on every response
- Click the sidebar tabs to switch between Chat, Observability, and Eval Dashboard

---

## Observability

Access via the **Observability** tab in the UI, or directly:
- `GET http://localhost:7005/metrics` — aggregate stats
- `GET http://localhost:7005/traces?n=20` — recent traces
- `GET http://localhost:7005/traces/{trace_id}` — single trace detail

Each trace includes:
- Span waterfall (retrieval, prompt, LLM, eval timings)
- Chunks retrieved + similarity scores
- Eval scores
- Answer preview

---

## Eval Dashboard

Access via the **Eval Dashboard** tab. Features:
- **Aggregate scores** across all queries (faithfulness, relevance, etc.)
- **Per-query breakdown** with judge reasoning
- **Manual evaluation** — paste any Q/A to test the judge independently

API endpoint for programmatic evaluation:
```bash
curl -X POST http://localhost:7005/eval \
  -H "Content-Type: application/json" \
  -d '{"question": "What is DNA?", "context_chunks": ["DNA is ..."], "answer": "DNA is ..."}'
```

---

## Known Issues
- Neo4j plugin `graph-data-science` must be enabled for image cosine similarity.
- Ensure Ollama is running before starting services.
- First query may be slow as models load into memory.
- Eval adds ~2-5s latency per query (set `EVAL_ENABLED=false` to disable).

---

## License
MIT License
