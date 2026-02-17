"""
Eval Agent / Judge – LLM-as-a-Judge evaluation for RAG responses.

Scores every answer on four dimensions:
  1. Faithfulness – Does the answer stick to the retrieved context?
  2. Relevance   – Does it actually address the user's question?
  3. Completeness – Does it cover all important points from context?
  4. Hallucination (inverse) – Lower = more hallucination detected.

Works with Ollama (free, local) so there's zero extra cost.
Scores are 0.0 – 1.0 and stored in the trace for the dashboard.
"""

import json, os, re, traceback
from typing import Dict, List, Optional
import requests
from observability import get_logger

logger = get_logger("eval_agent")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EVAL_MODEL = os.getenv("EVAL_MODEL", "llama3.1:8b")


# ── Evaluation prompt templates ───────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
You will be given a user question, the retrieved context passages, and the system's answer.
Your job is to evaluate the answer quality on FOUR dimensions, each scored 0.0 to 1.0.

Score definitions:
- faithfulness: How well does the answer stick to facts present in the context? 1.0 = fully faithful, 0.0 = completely made up.
- relevance: How well does the answer address the user's actual question? 1.0 = perfectly relevant, 0.0 = totally off-topic.
- completeness: How thoroughly does the answer cover the key information available in the context? 1.0 = comprehensive, 0.0 = misses everything.
- hallucination_free: Does the answer avoid stating things NOT in the context as fact? 1.0 = no hallucination, 0.0 = heavy hallucination.

You MUST respond with ONLY a valid JSON object in this exact format, nothing else:
{"faithfulness": <float>, "relevance": <float>, "completeness": <float>, "hallucination_free": <float>, "reasoning": "<brief explanation>"}
"""

JUDGE_USER_TEMPLATE = """## User Question
{question}

## Retrieved Context
{context}

## System Answer
{answer}

Evaluate the answer. Respond with ONLY the JSON object."""


def _call_ollama(system_prompt: str, user_prompt: str,
                 model: str = None) -> Optional[str]:
    """Send a chat completion to Ollama and return the response text."""
    model = model or EVAL_MODEL
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except Exception as e:
        logger.error(f"Ollama eval call failed: {e}")
        return None


def _parse_scores(raw: str) -> Optional[Dict[str, float]]:
    """Extract the JSON scores dict from the LLM response."""
    if not raw:
        return None
    # Try direct parse first
    try:
        data = json.loads(raw)
        return _validate_scores(data)
    except json.JSONDecodeError:
        pass
    # Try to find JSON in the response
    match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return _validate_scores(data)
        except json.JSONDecodeError:
            pass
    logger.warning(f"Could not parse eval scores from: {raw[:200]}")
    return None


def _validate_scores(data: dict) -> Optional[Dict[str, float]]:
    """Ensure all four score keys exist and values are valid floats."""
    required = ["faithfulness", "relevance", "completeness", "hallucination_free"]
    scores = {}
    for key in required:
        val = data.get(key)
        if val is None:
            return None
        try:
            val = float(val)
            scores[key] = max(0.0, min(1.0, val))  # clamp to [0,1]
        except (ValueError, TypeError):
            return None
    # Optionally include reasoning
    if "reasoning" in data:
        scores["reasoning"] = str(data["reasoning"])
    return scores


# ── Public API ────────────────────────────────────────────────────────

def evaluate_response(
    question: str,
    context_chunks: List[str],
    answer: str,
) -> Optional[Dict[str, float]]:
    """
    Run the judge LLM on a single Q-A pair.

    Returns dict with keys: faithfulness, relevance, completeness,
    hallucination_free, reasoning  (or None on failure).
    """
    if not answer or not question:
        return None

    context = "\n---\n".join(context_chunks) if context_chunks else "<no context retrieved>"

    user_prompt = JUDGE_USER_TEMPLATE.format(
        question=question,
        context=context,
        answer=answer,
    )

    logger.info("Running eval judge", extra={"trace_id": ""})
    raw = _call_ollama(JUDGE_SYSTEM_PROMPT, user_prompt)
    scores = _parse_scores(raw)

    if scores:
        logger.info(
            "Eval complete",
            extra={"eval_scores": {k: v for k, v in scores.items() if k != "reasoning"}},
        )
    else:
        logger.warning("Eval returned no valid scores")

    return scores


def batch_evaluate(
    items: List[Dict],
) -> List[Optional[Dict[str, float]]]:
    """
    Evaluate a batch of Q-A pairs.
    Each item: {"question": str, "context_chunks": list[str], "answer": str}
    Returns list of score dicts (same order).
    """
    results = []
    for item in items:
        scores = evaluate_response(
            question=item.get("question", ""),
            context_chunks=item.get("context_chunks", []),
            answer=item.get("answer", ""),
        )
        results.append(scores)
    return results
