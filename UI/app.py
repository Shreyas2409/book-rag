"""
Streamlit UI – Chat with your Book + Observability Dashboard + Eval Scores.

Improvements:
  • Shows eval scores (faithfulness, relevance, etc.) for each response
  • Observability sidebar with live metrics and recent traces
  • Trace detail view with span waterfall
  • Session-aware chat with conversation memory
  • Cleaner layout with status indicators
"""

import streamlit as st
import requests
import tempfile
import pathlib
import os
import uuid
import json
import time
from dotenv import load_dotenv

load_dotenv()

AGENT_API  = os.getenv("AGENT_API", "http://localhost:7005")
NEO4J_URI  = os.getenv("NEO4J_URI", "neo4j://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "pass")

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat with your Book",
    page_icon="book",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for a polished look ────────────────────────────────────
st.markdown("""
<style>
/* Score badges */
.score-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 16px;
    font-size: 0.85em;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 4px;
}
.score-high { background: #d4edda; color: #155724; }
.score-mid  { background: #fff3cd; color: #856404; }
.score-low  { background: #f8d7da; color: #721c24; }

/* Trace span bars */
.span-bar {
    height: 22px;
    border-radius: 4px;
    margin: 2px 0;
    display: flex;
    align-items: center;
    padding-left: 8px;
    font-size: 0.8em;
    color: white;
    font-weight: 500;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 16px;
    border-radius: 12px;
    color: white;
    text-align: center;
    margin-bottom: 8px;
}
.metric-card h3 { margin: 0; font-size: 1.8em; }
.metric-card p  { margin: 0; font-size: 0.85em; opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Chat"

# ── Helper functions ──────────────────────────────────────────────────

def score_badge(label: str, value: float) -> str:
    """Return HTML for a colored score badge."""
    if value >= 0.7:
        cls = "score-high"
    elif value >= 0.4:
        cls = "score-mid"
    else:
        cls = "score-low"
    return f'<span class="score-badge {cls}">{label}: {value:.2f}</span>'


def fetch_metrics():
    try:
        r = requests.get(f"{AGENT_API}/metrics", timeout=5)
        return r.json()
    except Exception:
        return None


def fetch_traces(n=20):
    try:
        r = requests.get(f"{AGENT_API}/traces", params={"n": n}, timeout=5)
        return r.json()
    except Exception:
        return []


def fetch_trace(trace_id):
    try:
        r = requests.get(f"{AGENT_API}/traces/{trace_id}", timeout=5)
        return r.json()
    except Exception:
        return None


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Book RAG v2")
    st.markdown("---")

    # File upload
    up = st.file_uploader("Upload a book (PDF/TXT)", type=["pdf", "txt"])
    if up and st.button("Ingest"):
        tmp = pathlib.Path(tempfile.mkdtemp()) / up.name
        tmp.write_bytes(up.getbuffer())
        st.info("Running ingestion … this may take a minute.")
        ingest_script = pathlib.Path(__file__).resolve().parents[0]
        # Try to find ingestion.py relative to the app
        for candidate in [
            pathlib.Path(__file__).resolve().parent / "ingestion.py",
            pathlib.Path(__file__).resolve().parents[1] / "ingestion.py",
        ]:
            if candidate.exists():
                ingest_script = candidate
                break
        code = os.system(f"python3 '{ingest_script}' '{tmp}'")
        if code == 0:
            st.success("Ingest complete!")
        else:
            st.error("Ingestion failed. Check logs.")

    st.markdown("---")

    # Navigation
    st.session_state.active_tab = st.radio(
        "Navigation",
        ["Chat", "Observability", "Eval Dashboard"],
        index=0,
    )

    st.markdown("---")
    st.caption(f"Session: `{st.session_state.session_id}`")

    if st.button("Clear Chat"):
        st.session_state.history = []
        st.session_state.session_id = uuid.uuid4().hex[:12]
        st.rerun()


# ── Tab: Chat ─────────────────────────────────────────────────────────
if st.session_state.active_tab == "Chat":
    st.title("Chat with your Book")
    st.caption("Ask questions about your uploaded books. Answers are evaluated in real-time.")

    # Display chat history
    for item in st.session_state.history:
        role = item["role"]
        content = item["content"]
        with st.chat_message(role):
            st.markdown(content)
            # Show eval scores if available
            if role == "assistant" and "eval_scores" in item and item["eval_scores"]:
                scores = item["eval_scores"]
                badges = " ".join(
                    score_badge(k.replace("_", " ").title(), v)
                    for k, v in scores.items()
                    if isinstance(v, (int, float))
                )
                st.markdown(badges, unsafe_allow_html=True)
                if "trace_id" in item:
                    st.caption(f"Trace: `{item['trace_id']}` | Chunks: {item.get('chunks_used', '?')}")

    # Chat input
    query = st.chat_input("Ask me anything about your uploaded books …")
    if query:
        # Show user message
        st.session_state.history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking …"):
                try:
                    r = requests.post(
                        f"{AGENT_API}/chat",
                        json={
                            "question": query,
                            "session_id": st.session_state.session_id,
                        },
                        timeout=180,
                    )
                    data = r.json()
                    answer = data.get("answer", "Sorry, something went wrong.")
                    eval_scores = data.get("eval_scores")
                    trace_id = data.get("trace_id", "")
                    chunks_used = data.get("chunks_used", 0)
                except Exception as e:
                    answer = f"Error: {e}"
                    eval_scores = None
                    trace_id = ""
                    chunks_used = 0

            st.markdown(answer)

            # Show eval scores
            if eval_scores:
                badges = " ".join(
                    score_badge(k.replace("_", " ").title(), v)
                    for k, v in eval_scores.items()
                    if isinstance(v, (int, float))
                )
                st.markdown(badges, unsafe_allow_html=True)
                st.caption(f"Trace: `{trace_id}` | Chunks: {chunks_used}")

        st.session_state.history.append({
            "role": "assistant",
            "content": answer,
            "eval_scores": eval_scores,
            "trace_id": trace_id,
            "chunks_used": chunks_used,
        })


# ── Tab: Observability ────────────────────────────────────────────────
elif st.session_state.active_tab == "Observability":
    st.title("Observability Dashboard")

    metrics = fetch_metrics()
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Queries", metrics.get("total_queries", 0))
        with col2:
            st.metric("Avg Latency", f"{metrics.get('avg_latency_ms', 0):.0f}ms")
        with col3:
            st.metric("P95 Latency", f"{metrics.get('p95_latency_ms', 0):.0f}ms")
        with col4:
            st.metric("Avg Chunks", f"{metrics.get('avg_chunks_retrieved', 0):.1f}")

        # Eval averages
        eval_avgs = metrics.get("eval_averages", {})
        if eval_avgs:
            st.markdown("### Evaluation Averages")
            ecols = st.columns(len(eval_avgs))
            for i, (k, v) in enumerate(eval_avgs.items()):
                with ecols[i]:
                    st.metric(k.replace("_", " ").title(), f"{v:.3f}")
    else:
        st.info("No metrics available yet. Start chatting to generate data!")

    st.markdown("---")

    # Recent traces
    st.markdown("### Recent Traces")
    traces = fetch_traces(20)
    if traces:
        for t in reversed(traces):
            tid = t.get("trace_id", "?")
            q = t.get("question", "")[:60]
            dur = t.get("total_duration_ms", 0)
            chunks = t.get("chunks_retrieved", 0)
            evals = t.get("eval_scores")

            with st.expander(f"`{tid}` -- {q}... ({dur:.0f}ms)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Question:** {t.get('question', '')}")
                    st.markdown(f"**Duration:** {dur:.0f}ms")
                    st.markdown(f"**Chunks retrieved:** {chunks}")
                    st.markdown(f"**Images retrieved:** {t.get('images_retrieved', 0)}")
                with col2:
                    if evals:
                        st.markdown("**Eval Scores:**")
                        for k, v in evals.items():
                            if isinstance(v, (int, float)):
                                badge = score_badge(k.replace("_", " ").title(), v)
                                st.markdown(badge, unsafe_allow_html=True)

                # Span waterfall
                spans = t.get("spans", [])
                if spans:
                    st.markdown("**Span Waterfall:**")
                    total = t.get("total_duration_ms", 1) or 1
                    colors = [
                        "#6366f1", "#8b5cf6", "#a855f7",
                        "#d946ef", "#ec4899", "#f43f5e",
                        "#f97316", "#eab308",
                    ]
                    for i, s in enumerate(spans):
                        dur_s = s.get("duration_ms", 0)
                        pct = min(100, max(5, (dur_s / total) * 100))
                        color = colors[i % len(colors)]
                        name = s.get("name", "?")
                        st.markdown(
                            f'<div class="span-bar" style="width:{pct}%;background:{color}">'
                            f'{name} ({dur_s:.0f}ms)</div>',
                            unsafe_allow_html=True,
                        )

                # Answer preview
                ans = t.get("answer", "")
                if ans:
                    st.markdown("**Answer preview:**")
                    st.text(ans[:500] + ("…" if len(ans) > 500 else ""))
    else:
        st.info("No traces yet.")


# ── Tab: Eval Dashboard ──────────────────────────────────────────────
elif st.session_state.active_tab == "Eval Dashboard":
    st.title("Eval Agent Dashboard")
    st.caption("Quality scores from the LLM-as-Judge evaluation agent")

    traces = fetch_traces(50)
    evaluated = [t for t in traces if t.get("eval_scores")]

    if evaluated:
        # Aggregate scores
        score_keys = ["faithfulness", "relevance", "completeness", "hallucination_free"]
        agg = {k: [] for k in score_keys}
        for t in evaluated:
            for k in score_keys:
                v = t["eval_scores"].get(k)
                if v is not None:
                    agg[k].append(float(v))

        # Summary cards
        cols = st.columns(4)
        for i, k in enumerate(score_keys):
            vals = agg[k]
            avg = sum(vals) / len(vals) if vals else 0
            with cols[i]:
                if avg >= 0.7:
                    color = "[HIGH]"
                elif avg >= 0.4:
                    color = "[MID]"
                else:
                    color = "[LOW]"
                st.metric(
                    f"{color} {k.replace('_', ' ').title()}",
                    f"{avg:.3f}",
                    f"n={len(vals)}",
                )

        st.markdown("---")

        # Per-query breakdown
        st.markdown("### Per-Query Scores")
        for t in reversed(evaluated):
            q = t.get("question", "")[:80]
            scores = t.get("eval_scores", {})
            with st.expander(f"{q}"):
                badges = " ".join(
                    score_badge(k.replace("_", " ").title(), v)
                    for k, v in scores.items()
                    if isinstance(v, (int, float))
                )
                st.markdown(badges, unsafe_allow_html=True)
                reasoning = scores.get("reasoning", "")
                if reasoning:
                    st.markdown(f"**Reasoning:** {reasoning}")
                ans = t.get("answer", "")
                if ans:
                    st.markdown(f"**Answer:** {ans[:300]}{'…' if len(ans) > 300 else ''}")

        st.markdown("---")

        # Manual eval
        st.markdown("### Manual Evaluation")
        with st.form("manual_eval"):
            eq = st.text_input("Question")
            ec = st.text_area("Context (paste relevant passages)")
            ea = st.text_area("Answer to evaluate")
            submitted = st.form_submit_button("Run Eval")
            if submitted and eq and ea:
                with st.spinner("Running eval judge …"):
                    try:
                        r = requests.post(
                            f"{AGENT_API}/eval",
                            json={
                                "question": eq,
                                "context_chunks": [ec] if ec else [],
                                "answer": ea,
                            },
                            timeout=120,
                        )
                        result = r.json()
                        scores = result.get("scores")
                        if scores:
                            badges = " ".join(
                                score_badge(k.replace("_", " ").title(), v)
                                for k, v in scores.items()
                                if isinstance(v, (int, float))
                            )
                            st.markdown(badges, unsafe_allow_html=True)
                            if "reasoning" in scores:
                                st.info(scores["reasoning"])
                        else:
                            st.warning("Eval returned no scores.")
                    except Exception as e:
                        st.error(f"Eval failed: {e}")
    else:
        st.info("No evaluated queries yet. Start chatting to generate eval scores!")
        st.markdown("---")
        st.markdown("### Manual Evaluation")
        with st.form("manual_eval_empty"):
            eq = st.text_input("Question")
            ec = st.text_area("Context (paste relevant passages)")
            ea = st.text_area("Answer to evaluate")
            submitted = st.form_submit_button("Run Eval")
            if submitted and eq and ea:
                with st.spinner("Running eval judge …"):
                    try:
                        r = requests.post(
                            f"{AGENT_API}/eval",
                            json={
                                "question": eq,
                                "context_chunks": [ec] if ec else [],
                                "answer": ea,
                            },
                            timeout=120,
                        )
                        result = r.json()
                        scores = result.get("scores")
                        if scores:
                            badges = " ".join(
                                score_badge(k.replace("_", " ").title(), v)
                                for k, v in scores.items()
                                if isinstance(v, (int, float))
                            )
                            st.markdown(badges, unsafe_allow_html=True)
                            if "reasoning" in scores:
                                st.info(scores["reasoning"])
                        else:
                            st.warning("Eval returned no scores.")
                    except Exception as e:
                        st.error(f"Eval failed: {e}")
