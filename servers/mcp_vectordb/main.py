"""
Text retriever MCP server – returns top-k text chunks with enriched metadata.

Improvements (v4):
  • Hybrid search: in-process BM25 index (rank-bm25) fused with vector search via RRF
  • BM25IndexState enum: UNBUILT / READY / REBUILDING
  • POST /bm25/refresh endpoint to rebuild the index on demand
  • GET /health extended with bm25_index and bm25_chunk_count fields
  • HybridQuery model replaces Query (backward-compatible; hybrid=False by default)
  • HybridResultChunk response model adds vector_score, bm25_score, retrieval_path
  • Four trace spans: bm25_retrieval, vector_retrieval, hybrid_merge, rerank
  • Configurable blend weight (HYBRID_BLEND_WEIGHT env var, default 0.7)
  • Graceful degradation: UNBUILT/empty BM25 → vector-only, no error

Previous improvements (v3):
  • Returns chunk_type, parent_id, section, chunk_id alongside text
  • Returns similarity scores so the agent/dashboard can display them
  • Structured logging via observability module
  • Supports parent-chunk expansion via direct Cypher query (O(1) not O(n))
  • Cross-encoder reranking for higher precision
  • Metadata filtering (filter by document name)
"""

from __future__ import annotations

import os
import pathlib
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

# Add paths for shared modules (Docker: same dir, local: grandparent dir)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from observability import get_logger

logger = get_logger("mcp_vectordb")

# ── Environment ───────────────────────────────────────────────────────
NEO4J_URI  = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")

RERANK_ENABLED       = os.getenv("RERANK_ENABLED", "true").lower() == "true"
HYBRID_SEARCH_ENABLED = os.getenv("HYBRID_SEARCH_ENABLED", "false").lower() == "true"

def _parse_blend_weight(raw: str | None, source: str = "env") -> float:
    """Parse and clamp blend_weight from a string value; log WARNING if out of range."""
    try:
        val = float(raw) if raw is not None else 0.7
    except (TypeError, ValueError):
        logger.warning(f"Invalid HYBRID_BLEND_WEIGHT={raw!r}, defaulting to 0.7")
        return 0.7
    if val < 0.0:
        logger.warning(f"HYBRID_BLEND_WEIGHT={val} from {source} < 0.0; clamping to 0.0")
        return 0.0
    if val > 1.0:
        logger.warning(f"HYBRID_BLEND_WEIGHT={val} from {source} > 1.0; clamping to 1.0")
        return 1.0
    return val

DEFAULT_BLEND_WEIGHT = _parse_blend_weight(os.getenv("HYBRID_BLEND_WEIGHT", "0.7"))

# ── Rate Limiting ─────────────────────────────────────────────────────
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address)
    RATE_LIMIT_ENABLED = True
except ImportError:
    limiter = None
    RATE_LIMIT_ENABLED = False
    logger.warning("slowapi not installed, rate limiting disabled")

# ── FastAPI app ────────────────────────────────────────────────────────
app = FastAPI(
    title="book_text_retriever",
    description="Returns top-k text chunks with metadata, reranking, hybrid BM25+vector search, and similarity scores.",
    version="4.0",
)

if RATE_LIMIT_ENABLED and limiter:
    app.state.limiter = limiter

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded. Please wait before sending more requests."},
        )

# ── Pydantic models ────────────────────────────────────────────────────

class HybridQuery(BaseModel):
    """Unified query model (replaces Query). Backward-compatible: hybrid=False by default."""
    query: str
    top_k: int = 8
    doc_filter: Optional[str] = None
    rerank: Optional[bool] = True
    hybrid: bool = False                      # FR: enable BM25 + vector fusion
    blend_weight: float = DEFAULT_BLEND_WEIGHT  # α: 1.0=pure vector, 0.0=pure BM25

    @field_validator("blend_weight", mode="before")
    @classmethod
    def clamp_blend_weight(cls, v):
        val = float(v)
        if val < 0.0:
            logger.warning(f"blend_weight={val} < 0.0 in request; clamping to 0.0")
            return 0.0
        if val > 1.0:
            logger.warning(f"blend_weight={val} > 1.0 in request; clamping to 1.0")
            return 1.0
        return val


class HybridResultChunk(BaseModel):
    """Response element for /invoke. Extends chunk with per-path score attribution."""
    text: str
    score: float
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    page: int = -1
    doc: str = ""
    chunk_id: str = ""
    chunk_type: str = ""
    parent_id: str = ""
    section: str = ""
    chunk_index: int = -1
    rerank_score: Optional[float] = None
    retrieval_path: Optional[str] = None   # "vector" | "bm25" | "both"; None when hybrid=False


class BM25RefreshResponse(BaseModel):
    status: str          # "ok" | "error"
    chunks_indexed: int
    duration_ms: float


class ParentQuery(BaseModel):
    parent_ids: List[str]


# ── BM25 Index Singleton ───────────────────────────────────────────────

class BM25IndexState(Enum):
    UNBUILT    = "unbuilt"
    READY      = "ready"
    REBUILDING = "rebuilding"


class BM25Index:
    """In-process BM25 keyword index over all child chunks. Not persisted to disk (v1)."""

    def __init__(self):
        self.corpus_chunk_ids: list[str] = []
        self.corpus_texts: list[str] = []
        self.corpus_meta: list[dict] = []   # page/doc/chunk_type/parent_id/section/chunk_index, parallel to corpus_texts
        self.bm25 = None            # BM25Okapi instance or None
        self.built_at: datetime | None = None
        self.chunk_count: int = 0
        self.state: BM25IndexState = BM25IndexState.UNBUILT
        self._building: bool = False  # concurrency sentinel


_bm25_index = BM25Index()


def _load_bm25_index() -> None:
    """Build/rebuild the BM25 index from Neo4j child chunks.

    Sets state to READY on success, leaves UNBUILT on Neo4j error.
    """
    global _bm25_index

    if not NEO4J_URI:
        logger.warning("NEO4J_URI not set; BM25 index left UNBUILT")
        return

    try:
        from neo4j import GraphDatabase
        from rank_bm25 import BM25Okapi
    except ImportError as e:
        logger.warning(f"BM25 dependencies unavailable ({e}); index left UNBUILT")
        return

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        with driver.session() as sess:
            records = sess.run(
                "MATCH (c:Chunk {chunk_type:'child'}) RETURN c.chunk_id AS chunk_id, "
                "c.text AS text, c.page AS page, c.doc AS doc, c.chunk_type AS chunk_type, "
                "c.parent_id AS parent_id, c.section AS section, c.chunk_index AS chunk_index"
            ).data()
        driver.close()
    except Exception as e:
        logger.warning(f"BM25 index build failed (Neo4j unavailable): {e}; index left UNBUILT")
        return

    if not records:
        logger.warning("BM25 index: no child chunks found in Neo4j; index left UNBUILT")
        return

    records   = [r for r in records if r.get("chunk_id")]
    chunk_ids = [r["chunk_id"] for r in records]
    texts     = [r["text"] or "" for r in records]
    # Carried through to _bm25_search() results so BM25-only hits (not
    # matched by the vector path) still have full citation metadata.
    metas = [
        {
            "page": r.get("page", -1),
            "doc": r.get("doc", ""),
            "chunk_type": r.get("chunk_type", ""),
            "parent_id": r.get("parent_id", ""),
            "section": r.get("section", ""),
            "chunk_index": r.get("chunk_index", -1),
        }
        for r in records
    ]

    # Tokenise: split on whitespace (rank_bm25 BM25Okapi handles punctuation internally)
    tokenised = [t.lower().split() for t in texts]

    bm25 = BM25Okapi(tokenised)

    _bm25_index.corpus_chunk_ids = chunk_ids
    _bm25_index.corpus_texts     = texts
    _bm25_index.corpus_meta      = metas
    _bm25_index.bm25             = bm25
    _bm25_index.chunk_count      = len(chunk_ids)
    _bm25_index.built_at         = datetime.now(timezone.utc)
    _bm25_index.state            = BM25IndexState.READY

    logger.info(f"BM25 index built: {len(chunk_ids)} child chunks indexed")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Build BM25 index at service start (non-fatal if Neo4j is unavailable)."""
    logger.info("Building BM25 index at startup …")
    _load_bm25_index()
    if _bm25_index.state == BM25IndexState.READY:
        logger.info(f"BM25 index ready ({_bm25_index.chunk_count} chunks)")
    else:
        logger.warning("BM25 index is UNBUILT at startup (Neo4j may not be ready yet)")
    yield


# `app` was constructed before this module's BM25 helpers existed, so the
# lifespan context is attached here rather than via FastAPI(lifespan=...).
app.router.lifespan_context = _lifespan


# ── RRF Merge Algorithm ────────────────────────────────────────────────

def rrf_merge(
    vector_results: list[dict],
    bm25_results: list[dict],
    blend_weight: float = 0.7,
    k: int = 60,
) -> list[dict]:
    """Weighted Reciprocal Rank Fusion.

    blend_weight α: 1.0 = pure vector, 0.0 = pure BM25.
    Deduplicates by chunk_id, sets retrieval_path to "vector"/"bm25"/"both".
    """
    scores: dict[str, dict] = {}

    # Vector path contribution
    for rank, chunk in enumerate(vector_results):
        cid = chunk.get("chunk_id", "")
        if not cid:
            continue
        contribution = blend_weight * (1 / (k + rank + 1))
        if cid not in scores:
            scores[cid] = {"rrf": 0.0, "meta": chunk, "vector_score": None, "bm25_score": None}
        scores[cid]["rrf"] += contribution
        scores[cid]["vector_score"] = chunk.get("score")

    # BM25 path contribution
    for rank, chunk in enumerate(bm25_results):
        cid = chunk.get("chunk_id", "")
        if not cid:
            continue
        contribution = (1 - blend_weight) * (1 / (k + rank + 1))
        if cid not in scores:
            scores[cid] = {"rrf": 0.0, "meta": chunk, "vector_score": None, "bm25_score": None}
        scores[cid]["rrf"] += contribution
        scores[cid]["bm25_score"] = chunk.get("bm25_score")

    # Build merged, deduplicated list sorted descending by RRF score
    merged = []
    for cid, data in sorted(scores.items(), key=lambda x: -x[1]["rrf"]):
        entry = {**data["meta"], "score": round(data["rrf"], 6)}
        if data["vector_score"] is not None:
            entry["vector_score"] = data["vector_score"]
        if data["bm25_score"] is not None:
            entry["bm25_score"] = data["bm25_score"]

        v = data["vector_score"] is not None
        b = data["bm25_score"] is not None
        entry["retrieval_path"] = "both" if (v and b) else ("vector" if v else "bm25")
        merged.append(entry)

    return merged


# ── BM25 Search Helper ─────────────────────────────────────────────────

def _bm25_search(query: str, top_k: int = 8) -> list[dict]:
    """Run BM25 keyword search against the in-memory index.

    Returns list of dicts with chunk_id, text, bm25_score, and the same
    page/doc/section metadata vector results carry (sorted descending).
    Returns [] if index is UNBUILT or empty.
    """
    if _bm25_index.state != BM25IndexState.READY or _bm25_index.bm25 is None:
        return []

    tokens = query.lower().split()
    raw_scores = _bm25_index.bm25.get_scores(tokens)

    # Zip with metadata, sort descending
    hits = sorted(
        zip(
            _bm25_index.corpus_chunk_ids,
            _bm25_index.corpus_texts,
            _bm25_index.corpus_meta,
            raw_scores,
        ),
        key=lambda x: -x[3],
    )

    results = []
    for chunk_id, text, meta, bm25_score in hits[:top_k]:
        if bm25_score <= 0.0:
            break  # rank_bm25 scores can be 0 for non-matching docs — stop early
        results.append({
            "chunk_id": chunk_id,
            "text": text,
            "bm25_score": round(float(bm25_score), 6),
            "score": round(float(bm25_score), 6),  # fallback 'score' for rrf_merge compat
            **meta,
        })

    return results


# ── Cross-encoder reranker ─────────────────────────────────────────────
_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker
    try:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Cross-encoder reranker loaded")
    except Exception as e:
        logger.warning(f"Reranker unavailable: {e}")
        _reranker = False
    return _reranker


def _rerank(query: str, results: list, top_k: int) -> list:
    """Re-rank results using cross-encoder for higher precision."""
    reranker = _get_reranker()
    if not reranker or not results:
        return results[:top_k]
    try:
        pairs = [(query, r["text"]) for r in results]
        scores = reranker.predict(pairs)
        for r, s in zip(results, scores):
            r["rerank_score"] = round(float(s), 4)
        results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return results[:top_k]
    except Exception as e:
        logger.warning(f"Reranking failed, returning original order: {e}")
        return results[:top_k]


# ── Neo4j vector store ────────────────────────────────────────────────
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector, Chroma

EMB = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
)

DB = None
CHROMA_DIR = os.getenv(
    "CHROMA_DIR",
    str(pathlib.Path(__file__).resolve().parents[2] / "chroma" / "books"),
)


def init_db():
    global DB
    if DB is not None:
        return DB
    try:
        # Ingestion already embeds and writes the "books" vector index via
        # from_documents(), so we attach to it with from_existing_index()
        # rather than from_existing_graph() (which requires node_label /
        # embedding_node_property / text_node_properties positional args
        # and would otherwise raise TypeError on every call).
        DB = Neo4jVector.from_existing_index(
            EMB,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASS,
            index_name="books",
        )
        logger.info("Connected to Neo4j vector store")

    except Exception as e:
        logger.warning(f"Neo4j unavailable, falling back to Chroma: {e}")
        try:
            os.makedirs(CHROMA_DIR, exist_ok=True)
        except Exception:
            pass
        try:
            DB = Chroma(persist_directory=CHROMA_DIR, embedding_function=EMB)
            logger.info("Connected to local Chroma store")
        except Exception:
            DB = None
    return DB


# ── Neo4j driver for direct Cypher queries ─────────────────────────────
_neo4j_driver = None


def _get_neo4j_driver():
    global _neo4j_driver
    if _neo4j_driver is not None:
        return _neo4j_driver
    if not NEO4J_URI:
        return None
    try:
        from neo4j import GraphDatabase
        _neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        logger.info("Direct Neo4j driver connected")
    except Exception as e:
        logger.warning(f"Neo4j driver unavailable: {e}")
        _neo4j_driver = False
    return _neo4j_driver


# ── /invoke endpoint ───────────────────────────────────────────────────

@app.post("/invoke")
def invoke(body: HybridQuery):
    """Retrieve top-k chunks. hybrid=True fuses BM25 + vector via RRF.
    When hybrid=False (default) the behaviour is identical to v3.
    """
    db = init_db()
    if db is None:
        logger.error("No vector store available")
        return []

    # Over-fetch candidates for reranking (rerank top-fetch_k → return top-k)
    fetch_k = body.top_k * 3 if (RERANK_ENABLED and body.rerank) else body.top_k

    # ── Tracer import (optional; graceful if unavailable) ─────────────
    try:
        from observability import Tracer
        tracer = Tracer(question=body.query)
        _tracing = True
    except Exception:
        tracer = None
        _tracing = False

    def _span(name):
        """Context manager for an optional trace span."""
        class _NullSpan:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def set_metadata(self, **_): pass

        if _tracing and tracer:
            return tracer.span(name)
        return _NullSpan()

    output: list[dict] = []

    # ── Hybrid branch ──────────────────────────────────────────────────
    if body.hybrid:
        # --- Vector retrieval span ---
        vector_results: list[dict] = []
        with _span("vector_retrieval") as vs:
            try:
                if body.doc_filter:
                    raw = db.similarity_search_with_score(
                        body.query, k=fetch_k, filter={"doc": body.doc_filter}
                    )
                else:
                    raw = db.similarity_search_with_score(body.query, k=fetch_k)
            except Exception:
                try:
                    docs = db.similarity_search(body.query, k=fetch_k)
                    raw = [(d, 0.0) for d in docs]
                except Exception:
                    raw = []

            for doc, score in raw:
                vector_results.append({
                    "text": doc.page_content,
                    "score": round(float(score), 4),
                    "page": doc.metadata.get("page", -1),
                    "doc": doc.metadata.get("doc", ""),
                    "chunk_id": doc.metadata.get("chunk_id", ""),
                    "chunk_type": doc.metadata.get("chunk_type", ""),
                    "parent_id": doc.metadata.get("parent_id", ""),
                    "section": doc.metadata.get("section", ""),
                    "chunk_index": doc.metadata.get("chunk_index", -1),
                })
            vs.set_metadata(candidate_count=len(vector_results))

        # --- BM25 retrieval span ---
        bm25_results: list[dict] = []
        with _span("bm25_retrieval") as bs:
            if _bm25_index.state == BM25IndexState.UNBUILT:
                logger.warning(
                    "BM25 index is UNBUILT at query time; falling back to vector-only"
                )
            else:
                bm25_results = _bm25_search(body.query, top_k=fetch_k)
            bs.set_metadata(
                candidate_count=len(bm25_results),
                index_state=_bm25_index.state.value,
            )

        # --- Graceful fallbacks (FR-008, FR-009) ---
        if not bm25_results and not vector_results:
            return []
        if not bm25_results:
            # FR-008: BM25 returned nothing → vector-only
            output = vector_results
            for r in output:
                r["retrieval_path"] = "vector"
        elif not vector_results:
            # FR-009: vector returned nothing → BM25-only
            output = bm25_results
            for r in output:
                r["retrieval_path"] = "bm25"
        else:
            # --- Hybrid merge span ---
            with _span("hybrid_merge") as hm:
                pre_count = len(vector_results) + len(bm25_results)
                output = rrf_merge(
                    vector_results,
                    bm25_results,
                    blend_weight=body.blend_weight,
                )
                hm.set_metadata(
                    pre_merge_candidates=pre_count,
                    post_dedup_count=len(output),
                    blend_weight=body.blend_weight,
                )

        # --- Rerank span ---
        with _span("rerank") as rs:
            if RERANK_ENABLED and body.rerank and output:
                output = _rerank(body.query, output, body.top_k)
                rs.set_metadata(reranked=True, top_k=body.top_k)
            else:
                output = output[:body.top_k]
                rs.set_metadata(reranked=False)

    # ── Vector-only branch (hybrid=False — unchanged from v3) ──────────
    else:
        try:
            if body.doc_filter:
                results = db.similarity_search_with_score(
                    body.query, k=fetch_k, filter={"doc": body.doc_filter}
                )
            else:
                results = db.similarity_search_with_score(body.query, k=fetch_k)
        except Exception:
            try:
                docs = db.similarity_search(body.query, k=fetch_k)
                results = [(d, 0.0) for d in docs]
            except Exception:
                return []

        for doc, score in results:
            output.append({
                "text": doc.page_content,
                "score": round(float(score), 4),
                "page": doc.metadata.get("page", -1),
                "doc": doc.metadata.get("doc", ""),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "chunk_type": doc.metadata.get("chunk_type", ""),
                "parent_id": doc.metadata.get("parent_id", ""),
                "section": doc.metadata.get("section", ""),
                "chunk_index": doc.metadata.get("chunk_index", -1),
            })
        # note: retrieval_path intentionally OMITTED when hybrid=False (API contract)

        if RERANK_ENABLED and body.rerank and output:
            output = _rerank(body.query, output, body.top_k)
        else:
            output = output[:body.top_k]

    logger.info(
        f"Retrieved {len(output)} chunks for query: {body.query[:80]}",
        extra={
            "chunk_count": len(output),
            "reranked": RERANK_ENABLED and bool(body.rerank),
            "hybrid": body.hybrid,
        },
    )
    return output


# ── /bm25/refresh endpoint ─────────────────────────────────────────────

def _do_bm25_refresh() -> BM25RefreshResponse:
    """Core rebuild logic shared by startup and the /bm25/refresh endpoint."""
    global _bm25_index

    if _bm25_index._building:
        # Concurrent rebuild in progress — wait and return current state
        logger.info("BM25 refresh: rebuild already in progress, waiting …")
        deadline = time.monotonic() + 30
        while _bm25_index._building and time.monotonic() < deadline:
            time.sleep(0.1)
        return BM25RefreshResponse(
            status="ok" if _bm25_index.state == BM25IndexState.READY else "error",
            chunks_indexed=_bm25_index.chunk_count,
            duration_ms=0.0,
        )

    _bm25_index._building = True
    _bm25_index.state = BM25IndexState.REBUILDING
    t0 = time.monotonic()

    try:
        _load_bm25_index()
        duration_ms = round((time.monotonic() - t0) * 1000, 2)

        if _bm25_index.state == BM25IndexState.READY:
            logger.info(
                f"BM25 refresh complete: {_bm25_index.chunk_count} chunks in {duration_ms}ms"
            )
            return BM25RefreshResponse(
                status="ok",
                chunks_indexed=_bm25_index.chunk_count,
                duration_ms=duration_ms,
            )
        else:
            return BM25RefreshResponse(
                status="error",
                chunks_indexed=0,
                duration_ms=duration_ms,
            )
    except Exception as e:
        duration_ms = round((time.monotonic() - t0) * 1000, 2)
        logger.error(f"BM25 refresh failed: {e}")
        _bm25_index.state = BM25IndexState.UNBUILT
        return BM25RefreshResponse(status="error", chunks_indexed=0, duration_ms=duration_ms)
    finally:
        _bm25_index._building = False


if RATE_LIMIT_ENABLED and limiter:
    @app.post("/bm25/refresh", response_model=BM25RefreshResponse)
    @limiter.limit("5/minute")
    async def bm25_refresh(request: Request):
        """Rebuild the in-memory BM25 index from the current Neo4j corpus.
        Rate-limited to 5 req/min (Constitution §IV Security-by-Default).
        """
        return _do_bm25_refresh()
else:
    @app.post("/bm25/refresh", response_model=BM25RefreshResponse)
    async def bm25_refresh():  # type: ignore[misc]
        """Rebuild the in-memory BM25 index from the current Neo4j corpus."""
        return _do_bm25_refresh()


# ── /parents endpoint ──────────────────────────────────────────────────

@app.post("/parents")
def get_parents(body: ParentQuery):
    """Fetch parent chunks by their IDs using direct Cypher query (O(1) lookup)."""
    if not body.parent_ids:
        return []

    driver = _get_neo4j_driver()
    if driver and driver is not False:
        try:
            results = []
            with driver.session() as sess:
                for pid in body.parent_ids:
                    records = sess.run(
                        "MATCH (c:Chunk {chunk_id: $id}) "
                        "WHERE c.chunk_type = 'parent' "
                        "RETURN c.text AS text, c.chunk_id AS chunk_id, "
                        "c.section AS section, c.page AS page",
                        id=pid,
                    ).data()
                    for r in records:
                        results.append({
                            "text": r.get("text", ""),
                            "chunk_id": r.get("chunk_id", pid),
                            "chunk_type": "parent",
                            "section": r.get("section", ""),
                            "page": r.get("page", -1),
                        })
            logger.info(f"Parent lookup via Cypher: found {len(results)}/{len(body.parent_ids)}")
            return results
        except Exception as e:
            logger.warning(f"Cypher parent lookup failed, trying fallback: {e}")

    db = init_db()
    if db is None:
        return []

    results = []
    try:
        all_docs = db.similarity_search("", k=200)
        parent_map = {}
        for doc in all_docs:
            cid = doc.metadata.get("chunk_id", "")
            if cid and doc.metadata.get("chunk_type") == "parent":
                parent_map[cid] = doc
        for pid in body.parent_ids:
            if pid in parent_map:
                pdoc = parent_map[pid]
                results.append({
                    "text": pdoc.page_content,
                    "chunk_id": pid,
                    "chunk_type": "parent",
                    "section": pdoc.metadata.get("section", ""),
                    "page": pdoc.metadata.get("page", -1),
                })
    except Exception as e:
        logger.warning(f"Parent lookup failed: {e}")

    return results


# ── /documents endpoint ────────────────────────────────────────────────

@app.get("/documents")
def list_documents():
    """List all distinct document names in the vector store."""
    driver = _get_neo4j_driver()
    if driver and driver is not False:
        try:
            with driver.session() as sess:
                records = sess.run(
                    "MATCH (c:Chunk) RETURN DISTINCT c.doc AS doc"
                ).data()
                return [r["doc"] for r in records if r.get("doc")]
        except Exception as e:
            logger.warning(f"Document list failed: {e}")
    return []


# ── /health endpoint ───────────────────────────────────────────────────

@app.get("/health")
def health():
    """Return service health including BM25 index status."""
    db = init_db()
    reranker = _get_reranker()
    return {
        "status": "ok" if db else "no_store",
        "backend": "neo4j" if NEO4J_URI else "chroma",
        "reranker": "active" if reranker and reranker is not False else "disabled",
        "bm25_index": _bm25_index.state.value,      # "ready" | "unbuilt" | "rebuilding"
        "bm25_chunk_count": _bm25_index.chunk_count,
    }
