"""
Agent API – ReAct (Reasoning + Acting) agent for RAG orchestration.

The ReAct pattern forces the LLM to follow an explicit loop:
  Thought  -> reason about what to do next
  Action   -> call a tool (text retriever, image retriever)
  Observation -> read the tool result
  ... repeat ...
  Thought  -> "I now have enough information"
  Final Answer -> synthesised response

This produces more transparent, traceable reasoning compared to
simple tool-calling, and lets the LLM decide dynamically whether
to retrieve more context, search for images, or refine its query.

Also includes: observability, conversation memory, eval judge.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import ChatOllama
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
EVAL_ENABLED = os.getenv("EVAL_ENABLED", "true").lower() == "true"
MAX_REACT_STEPS = int(os.getenv("MAX_REACT_STEPS", "6"))

# ── Tools ─────────────────────────────────────────────────────────────
# ReAct agents need tools that accept a single string input and return
# a string observation. We wrap the MCP endpoints accordingly.

def _call_text_retriever(query: str) -> str:
    """Retrieve text passages from the book. Returns formatted passages."""
    try:
        r = requests.post(
            f"{VECTOR_URL}/invoke",
            json={"query": query, "top_k": 8},
            timeout=60,
        )
        r.raise_for_status()
        results = r.json()
        if not results:
            return "No relevant passages found."
        # Format as readable observation for the ReAct loop
        parts = []
        for i, chunk in enumerate(results):
            text = chunk.get("text", "")
            page = chunk.get("page", "?")
            section = chunk.get("section", "")
            score = chunk.get("score", 0)
            header = f"[Passage {i+1} | Page {page}"
            if section:
                header += f" | Section: {section}"
            if score:
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
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
    base_url=OLLAMA_URL,
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

# ── FastAPI ───────────────────────────────────────────────────────────
app = FastAPI(title="chat_agent", version="2.1-react")


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    answer: str
    trace_id: str
    eval_scores: Optional[Dict[str, float]] = None
    chunks_used: int = 0
    reasoning_steps: int = 0


class EvalRequest(BaseModel):
    question: str
    context_chunks: List[str]
    answer: str


# ── Chat endpoint ─────────────────────────────────────────────────────

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
    answer = ""
    reasoning_steps = 0

    # ── Build context from conversation history ───────────────────────
    with tracer.span("build_context") as span:
        context = ""
        if memory and len(memory.turns) > 1:
            context = "Previous conversation:\n" + memory.format_for_prompt(max_chars=3000)
        span.set_metadata(context_chars=len(context))

    # ── Run ReAct agent ───────────────────────────────────────────────
    with tracer.span("react_agent") as span:
        try:
            if executor is not None:
                result = executor.invoke({
                    "input": req.question,
                    "context": context,
                    "max_steps": str(MAX_REACT_STEPS),
                })
                answer = result.get("output", "")

                # Extract intermediate steps for observability
                intermediate = result.get("intermediate_steps", [])
                reasoning_steps = len(intermediate)

                # Collect context chunks from observations
                for action, observation in intermediate:
                    if hasattr(action, 'tool') and action.tool == "book_text_retriever":
                        # Parse observation back to track chunks used
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
                    text_obs = _call_text_retriever(req.question)
                    context_chunks.append({"text": text_obs, "tool": "fallback"})
                except Exception:
                    text_obs = ""

            with tracer.span("fallback_generate"):
                try:
                    from langchain_core.messages import HumanMessage
                    fallback_prompt = (
                        f"Based on the following context, answer the question.\n\n"
                        f"Context:\n{text_obs}\n\n"
                        f"Question: {req.question}\n\n"
                        f"Answer:"
                    )
                    resp = llm.invoke([HumanMessage(content=fallback_prompt)])
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
        reasoning_steps=reasoning_steps,
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
    }
