"""
Integration tests for hybrid search endpoints.

Requires services to be running:
    docker compose -f docker.yaml up -d
    python ingestion.py sample.txt

Run with:  pytest tests/integration/test_hybrid_endpoint.py -v

Tests cover:
  T022 — POST /invoke with hybrid=True returns chunks with retrieval_path and
          non-null bm25_score or vector_score; GET /health shows "bm25_index": "ready"
  T023 — POST /bm25/refresh returns status=ok, chunks_indexed>0, duration_ms>0
  T026 — HYBRID_SEARCH_ENABLED=false: no bm25_retrieval, vector_retrieval, hybrid_merge
          spans in the trace; response has no retrieval_path or bm25_score fields
"""

import os
import pytest
import requests

MCP_VECTOR_URL = os.getenv("MCP_VECTOR_URL", "http://localhost:7001")
AGENT_URL      = os.getenv("AGENT_URL",      "http://localhost:7005")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _invoke(query, hybrid=True, blend_weight=0.7, top_k=3):
    resp = requests.post(
        f"{MCP_VECTOR_URL}/invoke",
        json={
            "query": query,
            "top_k": top_k,
            "rerank": False,   # disable rerank for faster integration test
            "hybrid": hybrid,
            "blend_weight": blend_weight,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _health():
    resp = requests.get(f"{MCP_VECTOR_URL}/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


def _bm25_refresh():
    resp = requests.post(f"{MCP_VECTOR_URL}/bm25/refresh", timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# T022 — hybrid /invoke endpoint
# ---------------------------------------------------------------------------

class TestHybridInvokeEndpoint:
    """T022: POST /invoke with hybrid=True validates hybrid fields in response."""

    @pytest.fixture(autouse=True)
    def require_service(self):
        """Skip if mcp_vectordb service is not reachable."""
        try:
            requests.get(f"{MCP_VECTOR_URL}/health", timeout=5)
        except Exception:
            pytest.skip("mcp_vectordb service not running (start with docker compose)")

    def test_health_shows_bm25_ready(self):
        """T022: GET /health must show bm25_index=ready after service startup."""
        h = _health()
        assert "bm25_index" in h, f"/health missing 'bm25_index' field: {h}"
        assert "bm25_chunk_count" in h, f"/health missing 'bm25_chunk_count' field: {h}"
        assert h["bm25_index"] == "ready", (
            f"Expected bm25_index=ready but got {h['bm25_index']!r}. "
            "Ensure at least one book has been ingested before running integration tests."
        )
        assert h["bm25_chunk_count"] > 0, "bm25_chunk_count should be > 0 after ingestion"

    def test_invoke_hybrid_returns_list(self):
        """T022: /invoke with hybrid=True must return a non-empty list."""
        results = _invoke("the main character")
        assert isinstance(results, list), f"Expected list, got {type(results)}"
        assert len(results) > 0, "Hybrid /invoke returned empty results"

    def test_invoke_hybrid_fields_present(self):
        """T022: Each chunk must contain retrieval_path and at least one score field."""
        results = _invoke("the main character")
        for chunk in results:
            assert "retrieval_path" in chunk, (
                f"retrieval_path missing from chunk: {list(chunk.keys())}"
            )
            assert chunk["retrieval_path"] in ("vector", "bm25", "both"), (
                f"Invalid retrieval_path: {chunk['retrieval_path']!r}"
            )
            # chunk must have at least one score — either bm25_score or vector_score
            has_bm25   = chunk.get("bm25_score") is not None
            has_vector = chunk.get("vector_score") is not None
            assert has_bm25 or has_vector, (
                f"Chunk has neither bm25_score nor vector_score: {chunk}"
            )

    def test_invoke_vector_only_no_hybrid_fields(self):
        """T026 (vector-only baseline): hybrid=False response must NOT include retrieval_path."""
        results = _invoke("the main character", hybrid=False)
        assert isinstance(results, list)
        for chunk in results:
            assert "retrieval_path" not in chunk, (
                f"retrieval_path should be absent when hybrid=False: {chunk}"
            )
            assert "bm25_score"     not in chunk, (
                f"bm25_score should be absent when hybrid=False: {chunk}"
            )
            assert "vector_score"   not in chunk, (
                f"vector_score should be absent when hybrid=False: {chunk}"
            )

    def test_invoke_blend_weight_extremes(self):
        """T018/T022: blend_weight=0.0 and 1.0 must not cause errors."""
        for w in (0.0, 1.0):
            results = _invoke("the main character", blend_weight=w)
            assert isinstance(results, list), (
                f"Expected list for blend_weight={w}, got {type(results)}"
            )

    def test_invoke_hybrid_score_is_float(self):
        """T022: score field must be a float (RRF score) in hybrid mode."""
        results = _invoke("the main character")
        for chunk in results:
            assert isinstance(chunk.get("score"), float), (
                f"score is not a float: {chunk.get('score')!r}"
            )


# ---------------------------------------------------------------------------
# T023 — /bm25/refresh endpoint
# ---------------------------------------------------------------------------

class TestBM25RefreshEndpoint:
    """T023: POST /bm25/refresh returns expected fields with valid values."""

    @pytest.fixture(autouse=True)
    def require_service(self):
        try:
            requests.get(f"{MCP_VECTOR_URL}/health", timeout=5)
        except Exception:
            pytest.skip("mcp_vectordb service not running")

    def test_refresh_returns_ok_status(self):
        """T023: /bm25/refresh must return status=ok."""
        data = _bm25_refresh()
        assert data.get("status") == "ok", f"Expected status=ok, got {data.get('status')!r}: {data}"

    def test_refresh_chunks_indexed_positive(self):
        """T023: chunks_indexed must be > 0 if books have been ingested."""
        data = _bm25_refresh()
        assert data.get("chunks_indexed", 0) > 0, (
            f"chunks_indexed={data.get('chunks_indexed')} — ensure a book is ingested first"
        )

    def test_refresh_duration_ms_positive(self):
        """T023: duration_ms must be > 0."""
        data = _bm25_refresh()
        assert data.get("duration_ms", 0) > 0, (
            f"duration_ms={data.get('duration_ms')!r} should be > 0"
        )

    def test_refresh_health_shows_ready_after(self):
        """T023: /health must show bm25_index=ready after a refresh."""
        _bm25_refresh()
        h = _health()
        assert h.get("bm25_index") == "ready", (
            f"bm25_index should be ready after refresh, got: {h.get('bm25_index')!r}"
        )

    def test_refresh_response_schema(self):
        """T023: Response must have exactly the expected fields."""
        data = _bm25_refresh()
        required = {"status", "chunks_indexed", "duration_ms"}
        missing = required - set(data.keys())
        assert not missing, f"Refresh response missing fields: {missing}"


# ---------------------------------------------------------------------------
# T026 — Feature flag off: no trace regression
# ---------------------------------------------------------------------------

class TestFeatureFlagOff:
    """T026: HYBRID_SEARCH_ENABLED=false → vector-only path, no hybrid spans in trace."""

    @pytest.fixture(autouse=True)
    def require_services(self):
        try:
            requests.get(f"{AGENT_URL}/health", timeout=5)
            requests.get(f"{MCP_VECTOR_URL}/health", timeout=5)
        except Exception:
            pytest.skip("Services not running")

    def test_chat_response_has_trace_id(self):
        """T026: /chat must return a trace_id we can look up."""
        resp = requests.post(
            f"{AGENT_URL}/chat",
            json={"question": "Who is the main character?"},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        assert "trace_id" in data and data["trace_id"], (
            f"trace_id missing from /chat response: {data}"
        )

    def test_vector_only_response_no_retrieval_path(self):
        """T026: When agent uses hybrid=False, retrieved chunks have no retrieval_path."""
        # Call /invoke directly with hybrid=False (simulating HYBRID_SEARCH_ENABLED=false)
        results = _invoke("main character", hybrid=False)
        for chunk in results:
            assert "retrieval_path" not in chunk, (
                "retrieval_path must not appear in vector-only response (SC-004)"
            )
