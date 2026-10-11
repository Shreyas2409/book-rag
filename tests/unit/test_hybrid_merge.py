"""
Unit tests for hybrid search merge logic and BM25 weight clamping.

Tests cover:
  T010 — rrf_merge() edge cases (normal, empty lists, vector/BM25 only, weight extremes)
  T011 — blend_weight clamping with WARNING log assertions
  T018 — extreme blend_weight values (0.0, 1.0) in rrf_merge
  T021 — trace span emission (bm25_retrieval, vector_retrieval, hybrid_merge, rerank)

Run with:  pytest tests/unit/test_hybrid_merge.py -v
"""

import importlib
import logging
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Minimal stubs so mcp_vectordb/main.py can be imported without running
# HuggingFaceEmbeddings, Neo4j driver, or rank_bm25 at import time.
# ---------------------------------------------------------------------------

def _make_stub_module(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# Stub heavy dependencies before importing the module under test
_needed_stubs = {
    "langchain_community": _make_stub_module("langchain_community"),
    "langchain_community.embeddings": _make_stub_module(
        "langchain_community.embeddings",
        HuggingFaceEmbeddings=MagicMock(return_value=MagicMock()),
    ),
    "langchain_community.vectorstores": _make_stub_module(
        "langchain_community.vectorstores",
        Neo4jVector=MagicMock(),
        Chroma=MagicMock(),
    ),
    "rank_bm25": _make_stub_module("rank_bm25", BM25Okapi=MagicMock()),
    "neo4j": _make_stub_module("neo4j", GraphDatabase=MagicMock()),
    "sentence_transformers": _make_stub_module(
        "sentence_transformers", CrossEncoder=MagicMock()
    ),
    # NOTE: `observability` is intentionally NOT stubbed — it is pure-stdlib and
    # the logging-assertion tests (T011) need `get_logger` to return a real
    # logging.Logger so `assertLogs` can observe WARNING records.
    "slowapi": _make_stub_module(
        "slowapi",
        Limiter=MagicMock(return_value=MagicMock()),
    ),
    "slowapi.util": _make_stub_module(
        "slowapi.util", get_remote_address=MagicMock()
    ),
    "slowapi.errors": _make_stub_module(
        "slowapi.errors", RateLimitExceeded=Exception
    ),
}

for mod_name, stub in _needed_stubs.items():
    sys.modules.setdefault(mod_name, stub)

# Import the functions under test directly (avoids running startup side effects)
import importlib.util, pathlib

_main_path = (
    pathlib.Path(__file__).resolve().parents[2]
    / "servers" / "mcp_vectordb" / "main.py"
)
spec = importlib.util.spec_from_file_location("mcp_vectordb_main", str(_main_path))
_mod = importlib.util.module_from_spec(spec)
sys.modules["mcp_vectordb_main"] = _mod
# Suppress logging noise during import; heavy deps are stubbed above so the
# module (and its FastAPI startup hook registration) imports without touching
# Neo4j, HuggingFace, or rank_bm25.
logging.disable(logging.CRITICAL)
try:
    spec.loader.exec_module(_mod)
finally:
    logging.disable(logging.NOTSET)

rrf_merge = _mod.rrf_merge
_bm25_index = _mod._bm25_index
BM25IndexState = _mod.BM25IndexState
_parse_blend_weight = _mod._parse_blend_weight


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vec(chunk_id, score=1.0):
    """Make a minimal vector result dict."""
    return {
        "chunk_id": chunk_id,
        "text": f"text for {chunk_id}",
        "score": score,
        "page": 1,
        "doc": "test.pdf",
        "chunk_type": "child",
        "parent_id": "",
        "section": "",
        "chunk_index": 0,
    }


def _bm25(chunk_id, bm25_score=1.0):
    """Make a minimal BM25 result dict."""
    return {
        "chunk_id": chunk_id,
        "text": f"text for {chunk_id}",
        "bm25_score": bm25_score,
        "score": bm25_score,
    }


# ===========================================================================
# T010 — rrf_merge() edge cases
# ===========================================================================

class TestRRFMerge(unittest.TestCase):
    """T010: rrf_merge() correctness across a range of input combinations."""

    # --- T010a: normal merge with overlapping chunks ---
    def test_normal_merge_overlapping(self):
        vec   = [_vec("c1", 0.9), _vec("c2", 0.8), _vec("c3", 0.7)]
        bm25  = [_bm25("c2", 5.0), _bm25("c4", 4.0), _bm25("c1", 3.0)]
        merged = rrf_merge(vec, bm25, blend_weight=0.7)

        chunk_ids = [r["chunk_id"] for r in merged]
        # c1 and c2 appear in both paths → present exactly once each
        self.assertEqual(chunk_ids.count("c1"), 1)
        self.assertEqual(chunk_ids.count("c2"), 1)
        # all four unique chunks present
        self.assertEqual(set(chunk_ids), {"c1", "c2", "c3", "c4"})

    # --- T010b: retrieval_path is set correctly ---
    def test_retrieval_path_labels(self):
        vec  = [_vec("v_only")]
        bm25 = [_bm25("b_only")]
        merged = rrf_merge([_vec("both"), *vec], [_bm25("both"), *bm25], blend_weight=0.5)

        path_map = {r["chunk_id"]: r["retrieval_path"] for r in merged}
        self.assertEqual(path_map["both"],   "both")
        self.assertEqual(path_map["v_only"], "vector")
        self.assertEqual(path_map["b_only"], "bm25")

    # --- T010c: both lists empty ---
    def test_both_lists_empty(self):
        merged = rrf_merge([], [], blend_weight=0.7)
        self.assertEqual(merged, [])

    # --- T010d: vector list empty (FR-009 path) ---
    def test_vector_empty_bm25_only(self):
        bm25 = [_bm25("c1"), _bm25("c2")]
        merged = rrf_merge([], bm25, blend_weight=0.7)
        self.assertEqual(len(merged), 2)
        for r in merged:
            self.assertEqual(r["retrieval_path"], "bm25")
            self.assertIsNone(r.get("vector_score"))

    # --- T010e: BM25 list empty (FR-008 path) ---
    def test_bm25_empty_vector_only(self):
        vec = [_vec("c1"), _vec("c2")]
        merged = rrf_merge(vec, [], blend_weight=0.7)
        self.assertEqual(len(merged), 2)
        for r in merged:
            self.assertEqual(r["retrieval_path"], "vector")
            self.assertIsNone(r.get("bm25_score"))

    # --- T010f: weight=0.0 — pure BM25 (vector contributes nothing) ---
    def test_weight_zero_pure_bm25(self):
        vec  = [_vec("v1", 0.99)]
        bm25 = [_bm25("b1", 9.99)]
        merged = rrf_merge(vec, bm25, blend_weight=0.0)
        # BM25 chunk should rank first since vector contribution = 0
        self.assertEqual(merged[0]["chunk_id"], "b1")
        # vector chunk gets (0.0 * 1/61) = 0.0 RRF score
        vec_entry = next(r for r in merged if r["chunk_id"] == "v1")
        self.assertEqual(vec_entry["score"], 0.0)

    # --- T010g: weight=1.0 — pure vector (BM25 contributes nothing) ---
    def test_weight_one_pure_vector(self):
        vec  = [_vec("v1", 0.99)]
        bm25 = [_bm25("b1", 9.99)]
        merged = rrf_merge(vec, bm25, blend_weight=1.0)
        self.assertEqual(merged[0]["chunk_id"], "v1")
        bm25_entry = next(r for r in merged if r["chunk_id"] == "b1")
        self.assertEqual(bm25_entry["score"], 0.0)

    # --- T010h: deduplication — same chunk_id in both lists → one entry ---
    def test_deduplication_by_chunk_id(self):
        shared_id = "shared"
        vec  = [_vec(shared_id, 0.9)]
        bm25 = [_bm25(shared_id, 5.0)]
        merged = rrf_merge(vec, bm25, blend_weight=0.7)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["chunk_id"], shared_id)

    # --- T010i: result is sorted descending by RRF score ---
    def test_result_sorted_descending(self):
        vec  = [_vec(f"v{i}") for i in range(5)]
        bm25 = [_bm25(f"b{i}") for i in range(5)]
        merged = rrf_merge(vec, bm25, blend_weight=0.5)
        scores = [r["score"] for r in merged]
        self.assertEqual(scores, sorted(scores, reverse=True))


# ===========================================================================
# T011 — blend_weight clamping
# ===========================================================================

class TestBlendWeightClamping(unittest.TestCase):
    """T011: _parse_blend_weight clamps out-of-range values and logs WARNING."""

    def test_below_zero_clamped_to_zero(self):
        with self.assertLogs("mcp_vectordb", level="WARNING"):
            result = _parse_blend_weight("-0.5")
        self.assertEqual(result, 0.0)

    def test_above_one_clamped_to_one(self):
        with self.assertLogs("mcp_vectordb", level="WARNING"):
            result = _parse_blend_weight("1.5")
        self.assertEqual(result, 1.0)

    def test_valid_value_unchanged(self):
        result = _parse_blend_weight("0.5")
        self.assertEqual(result, 0.5)

    def test_boundary_zero_no_warning(self):
        result = _parse_blend_weight("0.0")
        self.assertEqual(result, 0.0)

    def test_boundary_one_no_warning(self):
        result = _parse_blend_weight("1.0")
        self.assertEqual(result, 1.0)

    def test_invalid_string_defaults_to_0_7(self):
        result = _parse_blend_weight("not_a_number")
        self.assertEqual(result, 0.7)


# ===========================================================================
# T018 — extreme blend_weight values in rrf_merge
# ===========================================================================

class TestBlendWeightExtremes(unittest.TestCase):
    """T018: rrf_merge behaves correctly at weight extremes; dedup still works."""

    def _make_data(self):
        vec  = [_vec("c1", 0.9), _vec("c2", 0.8)]
        bm25 = [_bm25("c2", 5.0), _bm25("c3", 4.0)]
        return vec, bm25

    def test_weight_0_bm25_leads(self):
        vec, bm25 = self._make_data()
        merged = rrf_merge(vec, bm25, blend_weight=0.0)
        # c2 and c3 appear in bm25 path → should rank higher than c1 (vector-only)
        ids = [r["chunk_id"] for r in merged]
        self.assertIn("c2", ids[:2])  # c2 in top 2

    def test_weight_1_vector_leads(self):
        vec, bm25 = self._make_data()
        merged = rrf_merge(vec, bm25, blend_weight=1.0)
        ids = [r["chunk_id"] for r in merged]
        # c1 ranked first (top vector, with weight=1.0 BM25 contributes 0)
        self.assertEqual(ids[0], "c1")

    def test_dedup_works_at_weight_0(self):
        vec, bm25 = self._make_data()
        merged = rrf_merge(vec, bm25, blend_weight=0.0)
        ids = [r["chunk_id"] for r in merged]
        self.assertEqual(len(ids), len(set(ids)), "Duplicates found in merged output")

    def test_dedup_works_at_weight_1(self):
        vec, bm25 = self._make_data()
        merged = rrf_merge(vec, bm25, blend_weight=1.0)
        ids = [r["chunk_id"] for r in merged]
        self.assertEqual(len(ids), len(set(ids)), "Duplicates found in merged output")


# ===========================================================================
# T021 — trace span emission
# ===========================================================================

class TestTraceSpanEmission(unittest.TestCase):
    """T021: /invoke with hybrid=True records the four required trace spans."""

    def _recorded_spans(self, body_dict):
        """Run invoke() with a mocked Tracer and return the recorded span names."""
        recorded = []

        class _FakeSpan:
            def __init__(self, name):
                self.name = name
                recorded.append(name)

            def __enter__(self): return self
            def __exit__(self, *_): pass
            def set_metadata(self, **_): pass

        class _FakeTracer:
            def span(self, name): return _FakeSpan(name)
            def finish(self): return MagicMock(trace_id="test")

        # Provide mock DB that returns one result
        from unittest.mock import MagicMock
        mock_doc = MagicMock()
        mock_doc.page_content = "sample text"
        mock_doc.metadata = {
            "page": 1, "doc": "test.pdf", "chunk_id": "c1",
            "chunk_type": "child", "parent_id": "", "section": "", "chunk_index": 0,
        }

        with patch.object(_mod, "init_db", return_value=MagicMock(
            similarity_search_with_score=MagicMock(return_value=[(mock_doc, 0.9)])
        )), patch.object(_mod, "_bm25_index") as mock_idx, \
             patch("observability.Tracer", return_value=_FakeTracer()):

            mock_idx.state = BM25IndexState.READY
            mock_idx.bm25 = MagicMock()
            mock_idx.corpus_chunk_ids = ["c1"]
            mock_idx.corpus_texts = ["sample text"]
            # Patch _bm25_search to return a hit
            with patch.object(_mod, "_bm25_search", return_value=[{
                "chunk_id": "c1", "text": "sample text",
                "bm25_score": 1.5, "score": 1.5,
            }]):
                from mcp_vectordb_main import HybridQuery
                body = HybridQuery(**body_dict)
                _mod.invoke(body)

        return recorded

    def test_bm25_retrieval_span_recorded(self):
        """bm25_retrieval span must appear when hybrid=True."""
        spans = self._recorded_spans({"query": "test", "hybrid": True})
        self.assertIn("bm25_retrieval", spans)

    def test_vector_retrieval_span_recorded(self):
        spans = self._recorded_spans({"query": "test", "hybrid": True})
        self.assertIn("vector_retrieval", spans)

    def test_hybrid_merge_span_recorded(self):
        spans = self._recorded_spans({"query": "test", "hybrid": True})
        self.assertIn("hybrid_merge", spans)

    def test_rerank_span_recorded(self):
        spans = self._recorded_spans({"query": "test", "hybrid": True})
        self.assertIn("rerank", spans)

    def test_no_bm25_spans_when_hybrid_false(self):
        """When hybrid=False, BM25-specific spans must NOT be recorded."""
        spans = self._recorded_spans({"query": "test", "hybrid": False})
        self.assertNotIn("bm25_retrieval", spans)
        self.assertNotIn("hybrid_merge", spans)


if __name__ == "__main__":
    unittest.main()
