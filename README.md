# Chat with Your Book — v2 (Text + Images + Eval + ReAct)

This project allows you to **upload a PDF or textbook** and interact with it using both **text** and **images**. It retrieves relevant passages and diagrams using **multi-modal search** and processes everything locally with **Ollama**.

The agent uses the **ReAct (Reasoning + Acting)** paradigm — the LLM explicitly reasons step-by-step, decides which tools to call, observes the results, and iterates before producing a final grounded answer.

**v2 adds**: ReAct agent, observability tracing, hierarchical chunking, conversation memory, and an LLM-as-Judge evaluation agent.

---

## Features

### Core
- **ReAct Agent** — step-by-step Thought/Action/Observation reasoning loop.
- **Upload PDF/TXT** from the browser.
- **Automatic ingestion**: hierarchical text chunking + CLIP image embeddings.
- **Interactive chat** with multi-turn conversation memory.
- **Multi-modal retrieval**: text + image search.
- **Runs locally** using Ollama (llama3.1:8b).
- **One-click Docker Compose deployment**.

### ReAct Agent

The agent follows the ReAct paradigm (Yao et al., 2022) instead of simple tool-calling. Each query triggers an explicit reasoning loop:

```
Question: What is the structure of DNA?

Thought:  I need to find information about DNA structure in the book.
Action:   book_text_retriever
Input:    DNA structure double helix
Observation: [Passage 1 | Page 42] DNA consists of two polynucleotide
             strands that wind around each other to form a double helix...

Thought:  I have text but the user might benefit from a diagram reference.
Action:   book_image_retriever
Input:    DNA double helix structure diagram
Observation: Image ID: img_042, Page: 43, Similarity: 0.82

Thought:  I now have both text and a visual reference. I can answer.
Final Answer: DNA has a double helix structure consisting of two
              complementary strands... (see diagram on page 43)
```

**Why ReAct over simple tool-calling:**
- **Transparent reasoning** — you can see WHY the agent chose each tool.
- **Multi-step refinement** — if the first search is not enough, the agent reasons about it and tries a better query.
- **Better grounding** — the explicit Observation step forces the model to base its answer on retrieved content.
- **Debuggable** — the full reasoning chain is captured in the observability trace.

Configurable via `MAX_REACT_STEPS` (default: 6 loops per query).

### Observability
- **Structured JSON logging** across all services.
- **Span-based tracing** — every request generates a trace with timing for each phase (retrieval, LLM, eval).
- **Metrics dashboard** — avg/p95 latency, chunk counts, similarity scores.
- **Span waterfall view** — visualise where time is spent per request.
- Endpoints: `GET /traces`, `GET /traces/{id}`, `GET /metrics`.

### Hierarchical Chunking
- **Parent chunks** (2000 chars) + **child chunks** (500 chars).
- **Semantic-aware splitting** — respects paragraph and section boundaries.
- **Section header detection** — auto-detects chapters and section titles.
- **Enriched metadata** — chunk_id, parent_id, section, chunk_type on every chunk.
- **200-char overlap** to avoid mid-sentence cuts.

### Context Window
- **Sliding-window conversation memory** (configurable, default 10 turns).
- **History summarisation** — older turns are compressed via LLM to save tokens.
- **Parent-chunk expansion** — when a child chunk matches, the full parent is included for richer context.
- **Context budget** — total prompt capped at 12,000 chars (configurable) so the model never overflows.

### Eval Agent / Judge
- **LLM-as-a-Judge** evaluates every response on four dimensions:
  - **Faithfulness** — does the answer stick to the retrieved context?
  - **Relevance** — does it address the user's question?
  - **Completeness** — does it cover key information from context?
  - **Hallucination-free** — does it avoid stating ungrounded facts?
- Scores are **0.0 - 1.0**, displayed inline as colored badges.
- **Eval Dashboard** — aggregate quality metrics across all queries.
- **Manual evaluation** — paste any Q/A pair to test the judge.
- Uses the same local Ollama model — **zero extra cost**.

---

## Architecture

```
                          ┌──────────────────────────────────────────────┐
┌──────────────┐          │            Agent API (ReAct)                 │
│  Streamlit   │──POST──▶ │                                              │
│     UI       │  /chat   │   Question                                   │
│              │          │      |                                       │
│  - Chat      │          │      v                                       │
│  - Metrics   │          │   Thought: "I need to search the book..."    │
│  - Eval      │          │      |                                       │
└──────────────┘          │      v                                       │
                          │   Action: book_text_retriever ──────────┐    │
                          │      |                                  |    │
                          │   Observation: [passages...]   ◄────────┘    │
                          │      |                                       │
                          │      v                                       │
                          │   Thought: "I should check for diagrams..."  │
                          │      |                                       │
                          │      v                                       │
                          │   Action: book_image_retriever ─────────┐   │
                          │      |                                  |   │
                          │   Observation: [images...]     ◄────────┘   │
                          │      |                                       │
                          │      v                                       │
                          │   Thought: "I have enough info."             │
                          │      |                                       │
                          │      v                                       │
                          │   Final Answer ──▶ Eval Judge ──▶ Response   │
                          └──────────────────────────────────────────────┘
                                  |                       |
                          ┌───────▼───────┐       ┌───────▼────────┐
                          │ mcp_vectordb  │       │ mcp_image_     │
                          │ (text search) │       │ retriever      │
                          └───────┬───────┘       │ (CLIP search)  │
                                  |               └───────┬────────┘
                          ┌───────▼───────────────────────▼────────┐
                          │                Neo4j                    │
                          │  - Text chunks (parent + child)        │
                          │  - Image embeddings (CLIP)             │
                          └────────────────────────────────────────┘
                                          |
                          ┌───────────────▼────────────────────────┐
                          │           Ollama (local LLM)           │
                          │  llama3.1:8b (generation + eval)       │
                          └────────────────────────────────────────┘
```

---

## Services Overview

| Service             | Tech Stack                | Purpose                            |
|---------------------|---------------------------|------------------------------------|
| ui                  | Streamlit                 | Chat + Observability + Eval UI     |
| agent_api           | FastAPI + LangChain ReAct | ReAct orchestration, tracing, eval |
| mcp_vectordb        | FastAPI + Neo4j/Chroma    | Text retrieval with scores         |
| mcp_image_retriever | FastAPI + CLIP + Neo4j    | Image retrieval                    |
| neo4j               | Neo4j 5.20                | Vector + graph storage             |
| ingest_book         | Python + LangChain + CLIP | Hierarchical chunking + indexing   |

### Modules

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
MAX_REACT_STEPS=6              # max reasoning loops per query
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

The ingestion pipeline will:
1. Load the PDF/TXT
2. Create **hierarchical chunks** (parent 2000 chars + child 500 chars)
3. Detect **section headers** and enrich metadata
4. Store everything in Neo4j (or Chroma fallback)
5. Extract + embed images with CLIP

---

## Chatting

Visit: **http://localhost:8501**

The chat supports:
- Multi-turn conversations with memory
- Inline eval scores (colored badges) on every response
- ReAct step count showing how many reasoning loops the agent took
- Sidebar tabs to switch between Chat, Observability, and Eval Dashboard

---

## Observability

Access via the **Observability** tab in the UI, or directly:
- `GET http://localhost:7005/metrics` — aggregate stats
- `GET http://localhost:7005/traces?n=20` — recent traces
- `GET http://localhost:7005/traces/{trace_id}` — single trace detail

Each trace includes:
- ReAct reasoning chain (Thought/Action/Observation steps)
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

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Send a question, get a ReAct-generated answer with eval scores |
| GET | `/traces` | List recent traces (query param: `n`) |
| GET | `/traces/{id}` | Get a single trace with full reasoning chain |
| GET | `/metrics` | Aggregate performance and quality metrics |
| POST | `/eval` | Run the eval judge on a custom Q/A pair |
| GET | `/health` | Service status, agent type, config |

---

## Known Issues
- Neo4j plugin `graph-data-science` must be enabled for image cosine similarity.
- Ensure Ollama is running before starting services.
- First query may be slow as models load into memory.
- Eval adds ~2-5s latency per query (set `EVAL_ENABLED=false` to disable).
- ReAct occasionally produces malformed output; `handle_parsing_errors=True` recovers gracefully.

---

## License
MIT License
