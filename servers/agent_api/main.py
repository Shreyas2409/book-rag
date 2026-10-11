"""
Agent API – ReAct (Reasoning + Acting) agent for RAG orchestration.

The ReAct pattern forces the LLM to follow an explicit loop:
  Thought  -> reason about what to do next
  Action   -> call a tool (text retriever, image retriever)
  Observation -> read the tool result
  ... repeat ...
  Thought  -> "I now have enough information"
  Final Answer -> synthesised response

Improvements (v3):
  • Input validation & prompt injection protection
  • Token usage tracking from Ollama responses
  • Source citations returned alongside answers
  • Reasoning chain serialised for UI display
  • Agentic correction loop (retry if eval score < threshold)
  • Rate limiting via slowapi
  • HyDE query expansion for vague queries
"""

from fastapi import FastAPI, Request, Header
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain.tools import BaseTool, Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
import requests, os, traceback, uuid, json, re
from typing import List, Optional, Dict

# Shared modules
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
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
EVAL_ENABLED = os.getenv("EVAL_ENABLED", "true").lower() == "true"
MAX_REACT_STEPS = int(os.getenv("MAX_REACT_STEPS", "6"))
CORRECTION_THRESHOLD = float(os.getenv("CORRECTION_THRESHOLD", "0.5"))
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "2000"))

# ── Hybrid search config (T020) ───────────────────────────────────────
HYBRID_SEARCH_ENABLED = os.getenv("HYBRID_SEARCH_ENABLED", "false").lower() == "true"
try:
    HYBRID_BLEND_WEIGHT = float(os.getenv("HYBRID_BLEND_WEIGHT", "0.7"))
    HYBRID_BLEND_WEIGHT = max(0.0, min(1.0, HYBRID_BLEND_WEIGHT))
except (TypeError, ValueError):
    HYBRID_BLEND_WEIGHT = 0.7

# ── Input Validation & Prompt Injection Protection ────────────────────

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)",
    r"disregard\s+(all\s+)?(previous|prior|above)",
    r"you\s+are\s+now\s+a",
    r"pretend\s+(you\s+are|to\s+be)",
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"show\s+(me\s+)?(your\s+)?(system\s+)?prompt",
    r"repeat\s+(the\s+)?(above|system)\s+(text|prompt|instructions)",
    r"forget\s+(all\s+)?(your\s+)?(instructions?|rules?)",
    r"override\s+(your\s+)?(instructions?|rules?|behavior)",
]
_injection_re = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def sanitize_query(query: str) -> str:
    """Validate and sanitize user input to prevent prompt injection."""
    if not query or not query.strip():
        return ""
    query = query.strip()
    if len(query) > MAX_QUERY_LENGTH:
        query = query[:MAX_QUERY_LENGTH]
        logger.warning(f"Query truncated to {MAX_QUERY_LENGTH} chars")
    if _injection_re.search(query):
        logger.warning(f"Prompt injection detected, sanitizing query")
        return "Please ask a question about the book."
    return query


# ── Tools ─────────────────────────────────────────────────────────────
# ReAct agents need tools that accept a single string input and return
# a string observation. We wrap the MCP endpoints accordingly.

# Store the raw chunk data from the last retrieval for source citations
_last_retrieval_chunks: Dict[str, List[Dict]] = {}


def _call_text_retriever(query: str) -> str:
    """Retrieve text passages from the book. Returns formatted passages."""
    try:
        r = requests.post(
            f"{VECTOR_URL}/invoke",
            json={
                "query": query,
                "top_k": 8,
                "rerank": True,
                "hybrid": HYBRID_SEARCH_ENABLED,      # T020: pass feature flag through
                "blend_weight": HYBRID_BLEND_WEIGHT,  # T020: pass blend weight through
            },
            timeout=60,
        )
        r.raise_for_status()
        results = r.json()
        if not results:
            return "No relevant passages found."

        # Store raw chunks for source citations
        _last_retrieval_chunks["text"] = results

        # Format as readable observation for the ReAct loop
        parts = []
        for i, chunk in enumerate(results):
            text = chunk.get("text", "")
            page = chunk.get("page", "?")
            section = chunk.get("section", "")
            score = chunk.get("score", 0)
            rerank_score = chunk.get("rerank_score")
            header = f"[Passage {i+1} | Page {page}"
            if section:
                header += f" | Section: {section}"
            if rerank_score is not None:
                header += f" | Relevance: {rerank_score:.3f}"
            elif score:
                header += f" | Relevance: {score:.3f}"
            header += "]"
            parts.append(f"{header}\n{text}")
        return "\n---\n".join(parts)
    except Exception as e:
        return f"Text retrieval error: {e}"


def _call_image_retriever(query: str) -> str:
    """Retrieve relevant images/diagrams from the book."""
    try:
        r = requests.post(
            f"{IMAGE_URL}/invoke",
            json={"query": query, "top_k": 4},
            timeout=60,
        )
        r.raise_for_status()
        results = r.json()
        if not results:
            return "No relevant images found."

        # Store raw image data for citations
        _last_retrieval_chunks["images"] = results

        parts = []
        for img in results:
            parts.append(
                f"Image ID: {img.get('id', '?')}, "
                f"Page: {img.get('page', '?')}, "
                f"Document: {img.get('doc', '?')}, "
                f"Similarity: {img.get('score', 0):.3f}"
            )
        return "\n".join(parts)
    except Exception as e:
        return f"Image retrieval error: {e}"


# Create LangChain Tool objects for the ReAct agent
text_tool = Tool(
    name="book_text_retriever",
    func=_call_text_retriever,
    description=(
        "Search the book for relevant text passages. Input should be a "
        "natural language query describing what information you need. "
        "Use this for factual questions, summaries, definitions, or any "
        "text-based content from the book. You can call this multiple "
        "times with refined queries if the first results are insufficient."
    ),
)

image_tool = Tool(
    name="book_image_retriever",
    func=_call_image_retriever,
    description=(
        "Search the book for relevant diagrams, figures, or images. "
        "Input should describe the visual content you are looking for. "
        "Use this when the user asks about diagrams, illustrations, "
        "charts, or visual appearance of something in the book."
    ),
)

tools = [text_tool, image_tool]

# ── LLM ───────────────────────────────────────────────────────────────
# Default LLM is Ollama (local/free). Groq is available as an extra,
# per-request provider — its API key is supplied by the frontend on each
# request (header, never env/disk) and is used only to build a throwaway
# client for that request; it is never logged or persisted.
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
    base_url=OLLAMA_URL,
)


def build_groq_llm(api_key: str) -> ChatGroq:
    """Build a ChatGroq client from a caller-supplied API key."""
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=api_key,
    )


def build_react_executor(llm_instance) -> AgentExecutor:
    """Build a fresh ReAct AgentExecutor bound to the given LLM."""
    agent = create_react_agent(
        llm=llm_instance,
        tools=tools,
        prompt=REACT_PROMPT,
    )
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=MAX_REACT_STEPS,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        early_stopping_method="generate",
    )

# ── ReAct Prompt Template ─────────────────────────────────────────────
# This is the core of ReAct: the prompt structures the Thought/Action/
# Observation loop explicitly, forcing the LLM to reason before acting.

REACT_PROMPT = PromptTemplate.from_template(
    """You are an expert book assistant that answers questions by searching
through uploaded books. You reason step-by-step before giving a final answer.

You have access to the following tools:

{tools}

Tool names: {tool_names}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: reason about what you need to do. Consider what information you need
         and which tool would best provide it.
Action: the tool to use, must be one of [{tool_names}]
Action Input: the input to pass to the tool
Observation: the result from the tool
... (repeat Thought/Action/Action Input/Observation as needed, up to {max_steps} times)
Thought: I now have enough information to answer the question.
Final Answer: your comprehensive answer based on the retrieved information.
              Always cite page numbers when available.

Important rules:
- Always start with a Thought before taking any Action.
- If the first retrieval doesn't fully answer the question, refine your query
  and search again.
- For questions about visuals, use the image retriever.
- For text questions, use the text retriever.
- You may use both tools if the question involves both text and images.
- Base your Final Answer ONLY on information from the Observations.
- If no relevant information is found, say so honestly.

Begin!

{context}

Question: {input}
Thought: {agent_scratchpad}"""
)

# ── Build the ReAct agent ─────────────────────────────────────────────
try:
    react_agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=REACT_PROMPT,
    )
    executor = AgentExecutor(
        agent=react_agent,
        tools=tools,
        verbose=True,
        max_iterations=MAX_REACT_STEPS,
        handle_parsing_errors=True,      # gracefully handle malformed output
        return_intermediate_steps=True,   # capture the Thought/Action/Obs chain
        early_stopping_method="generate", # let the LLM produce a final answer if it hits max steps
    )
    logger.info(f"ReAct agent initialised (max {MAX_REACT_STEPS} steps)")
except Exception as e:
    logger.warning(f"ReAct agent unavailable: {e}, will use fallback path")
    react_agent = None
    executor = None

# ── Rate Limiting ─────────────────────────────────────────────────────
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from fastapi.responses import JSONResponse

    limiter = Limiter(key_func=get_remote_address)
    RATE_LIMIT_ENABLED = True
except ImportError:
    limiter = None
    RATE_LIMIT_ENABLED = False
    logger.warning("slowapi not installed, rate limiting disabled")

# ── FastAPI ───────────────────────────────────────────────────────────
app = FastAPI(title="chat_agent", version="3.0-react")

if RATE_LIMIT_ENABLED and limiter:
    app.state.limiter = limiter

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded. Please wait before sending more requests."},
        )


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"
    doc_filter: Optional[str] = None  # filter retrieval to a specific book
    provider: Optional[str] = "ollama"  # "ollama" (default) or "groq"


class ChatResponse(BaseModel):
    answer: str
    trace_id: str
    eval_scores: Optional[Dict[str, float]] = None
    chunks_used: int = 0
    reasoning_steps: int = 0
    sources: Optional[List[Dict]] = None        # source citations
    reasoning_chain: Optional[List[Dict]] = None  # Thought/Action/Obs chain


class EvalRequest(BaseModel):
    question: str
    context_chunks: List[str]
    answer: str


# ── Helper: serialise reasoning chain for UI ──────────────────────────

def _serialise_reasoning_chain(intermediate_steps: list) -> List[Dict]:
    """Convert LangChain intermediate_steps into a JSON-serialisable list."""
    chain = []
    for action, observation in intermediate_steps:
        step = {}
        if hasattr(action, 'tool'):
            step["tool"] = action.tool
            step["tool_input"] = str(action.tool_input)[:200]
        if hasattr(action, 'log'):
            # Extract the Thought from the log
            log = action.log.strip()
            thought_match = re.search(r'^(.*?)(?:Action:|$)', log, re.DOTALL)
            if thought_match:
                thought = thought_match.group(1).strip()
                if thought:
                    step["thought"] = thought
        step["observation_preview"] = str(observation)[:300]
        chain.append(step)
    return chain


# ── Helper: extract source citations from retrieved chunks ────────────

def _extract_sources(context_chunks: List[Dict]) -> List[Dict]:
    """Build a deduplicated list of source citations from retrieved chunks."""
    sources = []
    seen = set()
    # From text retrieval
    for chunk in _last_retrieval_chunks.get("text", []):
        key = (chunk.get("page", -1), chunk.get("section", ""))
        if key not in seen:
            seen.add(key)
            sources.append({
                "page": chunk.get("page", -1),
                "section": chunk.get("section", ""),
                "doc": chunk.get("doc", ""),
                "score": chunk.get("rerank_score") or chunk.get("score", 0),
                "chunk_type": chunk.get("chunk_type", ""),
                "type": "text",
            })
    # From image retrieval
    for img in _last_retrieval_chunks.get("images", []):
        sources.append({
            "page": img.get("page", -1),
            "doc": img.get("doc", ""),
            "score": img.get("score", 0),
            "type": "image",
            "image_id": img.get("id", ""),
        })
    return sources


# ── Chat endpoint ─────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, x_groq_api_key: Optional[str] = Header(default=None)):
    # Clear per-request retrieval cache
    _last_retrieval_chunks.clear()

    # Input validation
    clean_question = sanitize_query(req.question)
    if not clean_question:
        return ChatResponse(
            answer="Please provide a valid question.",
            trace_id="",
            chunks_used=0,
        )

    # Groq is opt-in and per-request: the key travels only in a request
    # header (never body/query, never logged, never persisted) and is used
    # to build a throwaway LLM + executor for this request only.
    use_groq = req.provider == "groq" and bool(x_groq_api_key)
    if use_groq:
        active_llm = build_groq_llm(x_groq_api_key)
        active_executor = build_react_executor(active_llm)
    else:
        active_llm = llm
        active_executor = executor

    tracer = Tracer(question=clean_question)
    memory = get_or_create_session(req.session_id)
    memory.add("user", clean_question)

    # Summarise older turns if needed
    if memory.needs_summarisation():
        with tracer.span("summarise_history"):
            if use_groq:
                summarise_history(memory, provider="groq", groq_api_key=x_groq_api_key)
            else:
                summarise_history(memory, ollama_url=OLLAMA_URL)

    context_chunks: List[Dict] = []
    answer = ""
    reasoning_steps = 0
    reasoning_chain = []

    # ── Build context from conversation history ───────────────────────
    with tracer.span("build_context") as span:
        context = ""
        if memory and len(memory.turns) > 1:
            context = "Previous conversation:\n" + memory.format_for_prompt(max_chars=3000)
        span.set_metadata(context_chars=len(context))

    # ── Run ReAct agent ───────────────────────────────────────────────
    with tracer.span("react_agent") as span:
        try:
            if active_executor is not None:
                result = active_executor.invoke({
                    "input": clean_question,
                    "context": context,
                    "max_steps": str(MAX_REACT_STEPS),
                })
                answer = result.get("output", "")

                # Extract intermediate steps for observability
                intermediate = result.get("intermediate_steps", [])
                reasoning_steps = len(intermediate)

                # Serialise reasoning chain for UI
                reasoning_chain = _serialise_reasoning_chain(intermediate)

                # Collect context chunks from observations
                for action, observation in intermediate:
                    if hasattr(action, 'tool') and action.tool == "book_text_retriever":
                        context_chunks.append({
                            "text": str(observation)[:500],
                            "tool": action.tool,
                            "query": action.tool_input,
                        })

                span.set_metadata(
                    steps=reasoning_steps,
                    tools_called=[
                        {"tool": a.tool, "input": str(a.tool_input)[:100]}
                        for a, _ in intermediate
                        if hasattr(a, 'tool')
                    ],
                )
                logger.info(
                    f"ReAct completed in {reasoning_steps} steps",
                    extra={"trace_id": tracer.trace.trace_id},
                )
            else:
                raise RuntimeError("No ReAct executor available")

        except Exception as e:
            logger.warning(f"ReAct agent failed: {e}")
            traceback.print_exc()

            # ── Fallback: manual retrieve + generate ──────────────────
            with tracer.span("fallback_retrieve"):
                try:
                    text_obs = _call_text_retriever(clean_question)
                    context_chunks.append({"text": text_obs, "tool": "fallback"})
                except Exception:
                    text_obs = ""

            with tracer.span("fallback_generate"):
                try:
                    from langchain_core.messages import HumanMessage
                    fallback_prompt = (
                        f"Based on the following context, answer the question.\n\n"
                        f"Context:\n{text_obs}\n\n"
                        f"Question: {clean_question}\n\n"
                        f"Answer:"
                    )
                    resp = active_llm.invoke([HumanMessage(content=fallback_prompt)])
                    answer = resp.content
                except Exception as e2:
                    logger.error(f"Fallback generation failed: {e2}")
                    answer = f"Retrieved context:\n{text_obs}" if text_obs else "Unable to retrieve information."

        span.set_metadata(answer_chars=len(answer))

    tracer.set_retrieval_stats(chunks=len(context_chunks))
    memory.add("assistant", answer)
    tracer.set_answer(answer)

    # ── Eval judge ────────────────────────────────────────────────────
    eval_scores = None
    if EVAL_ENABLED and answer:
        with tracer.span("eval_judge") as span:
            try:
                chunk_texts = [
                    c.get("text", "") for c in context_chunks if isinstance(c, dict)
                ]
                eval_scores = evaluate_response(
                    question=clean_question,
                    context_chunks=chunk_texts,
                    answer=answer,
                    provider="groq" if use_groq else "ollama",
                    groq_api_key=x_groq_api_key if use_groq else None,
                )
                if eval_scores:
                    tracer.set_eval(eval_scores)
                    span.set_metadata(scores=eval_scores)
            except Exception as e:
                logger.warning(f"Eval failed: {e}")

    # ── Agentic Correction Loop ───────────────────────────────────────
    # If eval scores are below threshold, retry with a refined query
    if eval_scores and EVAL_ENABLED:
        numeric_scores = [v for k, v in eval_scores.items() if isinstance(v, (int, float))]
        avg_score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 1.0
        if avg_score < CORRECTION_THRESHOLD and active_executor is not None:
            with tracer.span("correction_retry") as span:
                try:
                    logger.info(
                        f"Correction loop triggered (avg_score={avg_score:.2f} < {CORRECTION_THRESHOLD})"
                    )
                    refined_query = (
                        f"The previous answer was not good enough. "
                        f"Please search more carefully and answer: {clean_question}"
                    )
                    result2 = active_executor.invoke({
                        "input": refined_query,
                        "context": context,
                        "max_steps": str(MAX_REACT_STEPS),
                    })
                    retry_answer = result2.get("output", "")
                    if retry_answer:
                        # Re-evaluate the retry answer
                        retry_chunks = [c.get("text", "") for c in context_chunks]
                        retry_eval = evaluate_response(
                            question=clean_question,
                            context_chunks=retry_chunks,
                            answer=retry_answer,
                            provider="groq" if use_groq else "ollama",
                            groq_api_key=x_groq_api_key if use_groq else None,
                        )
                        retry_numeric = [v for k, v in (retry_eval or {}).items() if isinstance(v, (int, float))]
                        retry_avg = sum(retry_numeric) / len(retry_numeric) if retry_numeric else 0
                        if retry_avg > avg_score:
                            answer = retry_answer
                            eval_scores = retry_eval
                            tracer.set_eval(eval_scores)
                            tracer.set_answer(answer)
                            # Update memory with corrected answer
                            memory.turns[-1].content = answer
                            span.set_metadata(
                                improved=True,
                                old_avg=round(avg_score, 3),
                                new_avg=round(retry_avg, 3),
                            )
                            logger.info(f"Correction improved score: {avg_score:.2f} -> {retry_avg:.2f}")
                        else:
                            span.set_metadata(improved=False)
                except Exception as e:
                    logger.warning(f"Correction loop failed: {e}")
                    span.set_metadata(improved=False, error=str(e))

    # ── Extract source citations ──────────────────────────────────────
    sources = _extract_sources(context_chunks)

    # ── Finish trace ──────────────────────────────────────────────────
    trace = tracer.finish()

    return ChatResponse(
        answer=answer,
        trace_id=trace.trace_id,
        eval_scores={k: v for k, v in eval_scores.items() if k != "reasoning"}
        if eval_scores else None,
        chunks_used=len(context_chunks),
        reasoning_steps=reasoning_steps,
        sources=sources,
        reasoning_chain=reasoning_chain,
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


# ── Eval endpoint ─────────────────────────────────────────────────────

@app.post("/eval")
def eval_single(req: EvalRequest):
    """Run the eval judge on a single Q/A pair."""
    scores = evaluate_response(
        question=req.question,
        context_chunks=req.context_chunks,
        answer=req.answer,
    )
    return {"scores": scores}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "agent_type": "ReAct",
        "max_steps": MAX_REACT_STEPS,
        "eval_enabled": EVAL_ENABLED,
        "rate_limiting": RATE_LIMIT_ENABLED,
        "correction_threshold": CORRECTION_THRESHOLD,
        "input_sanitization": True,
    }
