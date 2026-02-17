"""
Observability module – structured logging, tracing, and metrics collection.

Every service imports `get_logger(name)` and `Tracer` to automatically
record span timings, similarity scores, chunk counts, and errors.
All trace data is stored in-memory and exposed via a `/traces` API so
the Streamlit dashboard can pull it in real-time.
"""

import time, uuid, json, logging, threading, os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

# ── Structured JSON logger ────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Attach extra fields if present (e.g. trace_id, span_name)
        for key in ("trace_id", "span_id", "span_name", "duration_ms",
                     "chunk_count", "similarity_scores", "tokens_used",
                     "eval_scores", "error"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val
        return json.dumps(log_entry)


def get_logger(name: str) -> logging.Logger:
    """Return a logger with JSON-formatted output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    return logger


# ── Span & Trace dataclasses ──────────────────────────────────────────

@dataclass
class Span:
    span_id: str
    name: str
    start: float
    end: Optional[float] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: Optional[str] = None


@dataclass
class Trace:
    trace_id: str
    question: str
    created_at: str
    spans: List[Span] = field(default_factory=list)
    total_duration_ms: Optional[float] = None
    # Retrieval stats
    chunks_retrieved: int = 0
    similarity_scores: List[float] = field(default_factory=list)
    images_retrieved: int = 0
    # LLM stats
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Eval scores (filled later by the judge)
    eval_scores: Optional[Dict[str, float]] = None
    answer: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["spans"] = [asdict(s) for s in self.spans]
        return d


# ── Tracer (manages in-flight trace + spans) ─────────────────────────

class Tracer:
    """Lightweight tracer that collects spans for a single request."""

    def __init__(self, question: str, trace_id: Optional[str] = None):
        self.trace = Trace(
            trace_id=trace_id or uuid.uuid4().hex[:16],
            question=question,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._start = time.perf_counter()
        self._logger = get_logger("tracer")

    # context-manager for individual spans
    class _SpanCtx:
        def __init__(self, tracer: "Tracer", name: str):
            self.tracer = tracer
            self.span = Span(
                span_id=uuid.uuid4().hex[:8],
                name=name,
                start=time.perf_counter(),
            )

        def set_metadata(self, **kw):
            self.span.metadata.update(kw)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.span.end = time.perf_counter()
            self.span.duration_ms = round(
                (self.span.end - self.span.start) * 1000, 2
            )
            if exc_type:
                self.span.status = "error"
                self.span.error = str(exc_val)
            self.tracer.trace.spans.append(self.span)
            self.tracer._logger.info(
                f"span:{self.span.name}",
                extra={
                    "trace_id": self.tracer.trace.trace_id,
                    "span_id": self.span.span_id,
                    "span_name": self.span.name,
                    "duration_ms": self.span.duration_ms,
                },
            )
            return False  # don't suppress exceptions

    def span(self, name: str) -> "_SpanCtx":
        return self._SpanCtx(self, name)

    def set_retrieval_stats(self, chunks: int = 0,
                            scores: List[float] = None,
                            images: int = 0):
        self.trace.chunks_retrieved = chunks
        self.trace.similarity_scores = scores or []
        self.trace.images_retrieved = images

    def set_llm_stats(self, prompt_tokens: int = 0,
                      completion_tokens: int = 0):
        self.trace.prompt_tokens = prompt_tokens
        self.trace.completion_tokens = completion_tokens

    def set_eval(self, scores: Dict[str, float]):
        self.trace.eval_scores = scores

    def set_answer(self, answer: str):
        self.trace.answer = answer

    def finish(self) -> Trace:
        self.trace.total_duration_ms = round(
            (time.perf_counter() - self._start) * 1000, 2
        )
        TraceStore.add(self.trace)
        self._logger.info(
            "trace_complete",
            extra={
                "trace_id": self.trace.trace_id,
                "duration_ms": self.trace.total_duration_ms,
                "chunk_count": self.trace.chunks_retrieved,
                "similarity_scores": self.trace.similarity_scores,
            },
        )
        return self.trace


# ── In-memory trace store (thread-safe, ring buffer) ──────────────────

class TraceStore:
    _lock = threading.Lock()
    _traces: List[Trace] = []
    _max = 200  # keep last 200 traces

    @classmethod
    def add(cls, t: Trace):
        with cls._lock:
            cls._traces.append(t)
            if len(cls._traces) > cls._max:
                cls._traces = cls._traces[-cls._max:]

    @classmethod
    def all(cls) -> List[dict]:
        with cls._lock:
            return [t.to_dict() for t in cls._traces]

    @classmethod
    def last(cls, n: int = 20) -> List[dict]:
        with cls._lock:
            return [t.to_dict() for t in cls._traces[-n:]]

    @classmethod
    def get(cls, trace_id: str) -> Optional[dict]:
        with cls._lock:
            for t in cls._traces:
                if t.trace_id == trace_id:
                    return t.to_dict()
            return None

    @classmethod
    def summary(cls) -> dict:
        """Aggregate metrics across all stored traces."""
        with cls._lock:
            if not cls._traces:
                return {"total_queries": 0}
            durations = [t.total_duration_ms for t in cls._traces if t.total_duration_ms]
            chunks = [t.chunks_retrieved for t in cls._traces]
            all_scores = []
            for t in cls._traces:
                all_scores.extend(t.similarity_scores)
            eval_agg: Dict[str, List[float]] = {}
            for t in cls._traces:
                if t.eval_scores:
                    for k, v in t.eval_scores.items():
                        eval_agg.setdefault(k, []).append(v)
            return {
                "total_queries": len(cls._traces),
                "avg_latency_ms": round(sum(durations) / len(durations), 1) if durations else 0,
                "p95_latency_ms": round(sorted(durations)[int(len(durations) * 0.95)] if durations else 0, 1),
                "avg_chunks_retrieved": round(sum(chunks) / len(chunks), 1) if chunks else 0,
                "avg_similarity": round(sum(all_scores) / len(all_scores), 4) if all_scores else 0,
                "eval_averages": {
                    k: round(sum(v) / len(v), 3) for k, v in eval_agg.items()
                },
            }
