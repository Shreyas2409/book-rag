"""
Agent API – orchestrates retrieval, LLM generation, context management,
observability tracing, and eval-judge scoring.

Improvements:
  • Full observability: every request gets a trace with span timings
  • Conversation memory with sliding window + summarisation
  • Parent-chunk expansion for richer context
  • Eval agent scores every response (async-friendly fallback)
  • /traces, /metrics, /eval endpoints for the dashboard
"""

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain.tools import BaseTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
import requests, os, traceback, uuid
from typing import List, Optional, Dict

# New modules – try direct import first (Docker), then parent dir (local dev)
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from observability import get_logger, Tracer, TraceStore
from eval_agent import evaluate_response
from context_window import (
    get_or_create_session,
    build_augmented_prompt,
    summarise_history,
    ConversationMemory,
)

logger = get_logger("agent_api")

# ── Service URLs ──────────────────────────────────────────────────────
VECTOR_URL = os.getenv("MCP_VECTOR_URL", "http://localhost:7001")
IMAGE_URL  = os.getenv("MCP_IMAGE_URL",  "http://localhost:7002")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EVAL_ENABLED = os.getenv("EVAL_ENABLED", "true").lower() == "true"

# ── Tools ─────────────────────────────────────────────────────────────

class MCPTool(BaseTool):
    endpoint: str
    name: str
    description: str

    def _run(self, query: str, top_k: int = 8):
        r = requests.post(
            f"{self.endpoint}/invoke",
            json={"query": query, "top_k": top_k},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()


vector_tool = MCPTool(
    endpoint=VECTOR_URL,
    name="book_text_retriever",
    description=(
        "Fetch relevant text passages from the book. Use for any "
        "textual or factual questions about book content."
    ),
)

image_tool = MCPTool(
    endpoint=IMAGE_URL,
    name="book_image_retriever",
    description=(
        "Fetch relevant diagrams, pictures, or figures from the book. "
        "Use when the user asks about visual content."
    ),
)

# ── LLM ───────────────────────────────────────────────────────────────
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
    base_url=OLLAMA_URL,
)

try:
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert search assistant helping users find information "
         "in books. You have access to text and image retrieval tools. "
         "Always cite page numbers when available. "
         "Use the conversation history for context in follow-up questions."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, tools=[vector_tool, image_tool], prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=[vector_tool, image_tool], verbose=True)
except Exception:
    logger.warning("Tool-calling agent unavailable, will use fallback path")
    agent = None
    executor = None

# ── FastAPI ───────────────────────────────────────────────────────────
app = FastAPI(title="chat_agent", version="2.0")


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    answer: str
    trace_id: str
    eval_scores: Optional[Dict[str, float]] = None
    chunks_used: int = 0


class EvalRequest(BaseModel):
    question: str
    context_chunks: List[str]
    answer: str


# ── Chat endpoint (the main one) ─────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    tracer = Tracer(question=req.question)
    memory = get_or_create_session(req.session_id)
    memory.add("user", req.question)

    # Summarise older turns if needed
    if memory.needs_summarisation():
        with tracer.span("summarise_history"):
            summarise_history(memory, ollama_url=OLLAMA_URL)

    context_chunks: List[Dict] = []
    parent_chunks: List[Dict] = []
    answer = ""

    # ── Step 1: Retrieve ──────────────────────────────────────────────
    with tracer.span("retrieve_text") as span:
        try:
            vec_results = vector_tool._run(req.question, top_k=8)
            if isinstance(vec_results, list):
                context_chunks = vec_results
                # Separate parents from children for expansion
                for c in context_chunks:
                    if c.get("chunk_type") == "parent":
                        parent_chunks.append(c)
            span.set_metadata(chunks=len(context_chunks))
        except Exception as e:
            logger.warning(f"Text retrieval failed: {e}")
            span.set_metadata(error=str(e))

    with tracer.span("retrieve_images") as span:
        try:
            img_results = image_tool._run(req.question, top_k=4)
            if isinstance(img_results, list):
                tracer.set_retrieval_stats(
                    chunks=len(context_chunks),
                    images=len(img_results),
                    scores=[
                        c.get("score", 0.0) for c in context_chunks
                        if isinstance(c, dict) and "score" in c
                    ],
                )
            span.set_metadata(images=len(img_results) if isinstance(img_results, list) else 0)
        except Exception as e:
            logger.warning(f"Image retrieval failed: {e}")
            img_results = []

    # ── Step 2: Build augmented prompt ────────────────────────────────
    with tracer.span("build_prompt") as span:
        augmented_input = build_augmented_prompt(
            question=req.question,
            retrieved_chunks=context_chunks,
            parent_chunks=parent_chunks if parent_chunks else None,
            memory=memory,
        )
        span.set_metadata(prompt_chars=len(augmented_input))

    # ── Step 3: Generate answer ───────────────────────────────────────
    with tracer.span("llm_generate") as span:
        try:
            if executor is not None:
                result = executor.invoke({"input": augmented_input})
                answer = result.get("output", "")
            else:
                raise RuntimeError("No executor")
        except Exception:
            traceback.print_exc()
            # Fallback: direct LLM call with context
            try:
                from langchain_core.messages import HumanMessage
                resp = llm.invoke([HumanMessage(content=augmented_input)])
                answer = resp.content
            except Exception as e2:
                logger.error(f"LLM generation failed: {e2}")
                # Last resort: return raw context
                text_bits = [c.get("text", "") for c in context_chunks if isinstance(c, dict)]
                img_bits = [
                    f"Image {i.get('id','?')} on page {i.get('page','?')}"
                    for i in (img_results if isinstance(img_results, list) else [])
                    if isinstance(i, dict)
                ]
                answer = "\n\n".join([
                    "Relevant text: " + ("\n".join(text_bits) if text_bits else "<none>"),
                    "Relevant images: " + ("\n".join(img_bits) if img_bits else "<none>"),
                ])
        span.set_metadata(answer_chars=len(answer))

    memory.add("assistant", answer)
    tracer.set_answer(answer)

    # ── Step 4: Eval judge ────────────────────────────────────────────
    eval_scores = None
    if EVAL_ENABLED and answer:
        with tracer.span("eval_judge") as span:
            try:
                chunk_texts = [
                    c.get("text", "") for c in context_chunks if isinstance(c, dict)
                ]
                eval_scores = evaluate_response(
                    question=req.question,
                    context_chunks=chunk_texts,
                    answer=answer,
                )
                if eval_scores:
                    tracer.set_eval(eval_scores)
                    span.set_metadata(scores=eval_scores)
            except Exception as e:
                logger.warning(f"Eval failed: {e}")

    # ── Finish trace ──────────────────────────────────────────────────
    trace = tracer.finish()

    return ChatResponse(
        answer=answer,
        trace_id=trace.trace_id,
        eval_scores={k: v for k, v in eval_scores.items() if k != "reasoning"}
        if eval_scores else None,
        chunks_used=len(context_chunks),
    )


# ── Observability endpoints ───────────────────────────────────────────

@app.get("/traces")
def get_traces(n: int = 20):
    """Return the last N traces."""
    return TraceStore.last(n)


@app.get("/traces/{trace_id}")
def get_trace(trace_id: str):
    """Return a single trace by ID."""
    t = TraceStore.get(trace_id)
    return t or {"error": "not found"}


@app.get("/metrics")
def get_metrics():
    """Aggregate metrics across all stored traces."""
    return TraceStore.summary()


# ── Eval endpoint (for batch / manual evaluation) ─────────────────────

@app.post("/eval")
def eval_single(req: EvalRequest):
    """Run the eval judge on a single Q/A pair (for testing or batch)."""
    scores = evaluate_response(
        question=req.question,
        context_chunks=req.context_chunks,
        answer=req.answer,
    )
    return {"scores": scores}


@app.get("/health")
def health():
    return {"status": "ok", "eval_enabled": EVAL_ENABLED}
