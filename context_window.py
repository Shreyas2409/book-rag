"""
Context Window Manager – conversation history + parent-chunk expansion.

Two concerns solved here:
1. **Conversation memory**: sliding window of the last N turns, with
   optional summarisation of older turns so the LLM always has context
   without blowing up the token budget.
2. **Parent-chunk expansion**: when a small child chunk is retrieved,
   fetch the parent chunk so the LLM sees the full surrounding context.
"""

import os, hashlib, json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

# ── Configuration ─────────────────────────────────────────────────────

MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))
CONTEXT_BUDGET_CHARS = int(os.getenv("CONTEXT_BUDGET_CHARS", "12000"))
SUMMARY_TRIGGER = int(os.getenv("SUMMARY_TRIGGER", "6"))  # summarise after N turns


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class Turn:
    role: str        # "user" or "assistant"
    content: str
    turn_index: int = 0


@dataclass
class ConversationMemory:
    """Holds a single session's chat history."""
    session_id: str
    turns: List[Turn] = field(default_factory=list)
    summary: str = ""  # rolling summary of older turns

    def add(self, role: str, content: str):
        t = Turn(role=role, content=content, turn_index=len(self.turns))
        self.turns.append(t)

    def recent(self, n: int = None) -> List[Turn]:
        n = n or MAX_HISTORY_TURNS
        return self.turns[-n:]

    def format_for_prompt(self, max_chars: int = 4000) -> str:
        """
        Build a string of conversation history that fits within max_chars.
        Prepends the rolling summary if available.
        """
        parts: List[str] = []
        if self.summary:
            parts.append(f"[Conversation summary so far]: {self.summary}")

        recent = self.recent()
        for t in recent:
            prefix = "User" if t.role == "user" else "Assistant"
            parts.append(f"{prefix}: {t.content}")

        text = "\n".join(parts)
        # Trim from the front if too long
        if len(text) > max_chars:
            text = "..." + text[-(max_chars - 3):]
        return text

    def needs_summarisation(self) -> bool:
        return len(self.turns) > SUMMARY_TRIGGER and len(self.turns) % SUMMARY_TRIGGER == 0


# ── Session store (in-memory, keyed by session ID) ────────────────────

_sessions: Dict[str, ConversationMemory] = {}


def get_or_create_session(session_id: str) -> ConversationMemory:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationMemory(session_id=session_id)
    return _sessions[session_id]


def list_sessions() -> List[str]:
    return list(_sessions.keys())


# ── Context assembly (combines history + retrieval) ───────────────────

def build_augmented_prompt(
    question: str,
    retrieved_chunks: List[Dict],
    parent_chunks: Optional[List[Dict]] = None,
    memory: Optional[ConversationMemory] = None,
    context_budget: int = None,
) -> str:
    """
    Assemble the final prompt sent to the LLM, comprising:
      1. Conversation history (if multi-turn)
      2. Retrieved context (child chunks + expanded parents)
      3. The current question

    The total context is capped at `context_budget` characters.
    """
    budget = context_budget or CONTEXT_BUDGET_CHARS
    sections: List[str] = []

    # 1) Conversation history
    if memory and memory.turns:
        history = memory.format_for_prompt(max_chars=budget // 4)
        sections.append(f"## Previous Conversation\n{history}")

    # 2) Retrieved context
    ctx_parts: List[str] = []

    # Deduplicate – if we have parent expansions, prefer the parent text
    seen_parents = set()
    if parent_chunks:
        for pc in parent_chunks:
            pid = pc.get("chunk_id") or pc.get("parent_id", "")
            if pid and pid not in seen_parents:
                seen_parents.add(pid)
                section = pc.get("section", "")
                page = pc.get("page", "")
                header = f"[Section: {section} | Page: {page}]" if section or page else ""
                ctx_parts.append(f"{header}\n{pc['text']}")

    # Add child chunks that weren't already covered by a parent
    for c in retrieved_chunks:
        parent_id = c.get("parent_id", "")
        if parent_id and parent_id in seen_parents:
            continue  # parent already included
        section = c.get("section", "")
        page = c.get("page", "")
        header = f"[Section: {section} | Page: {page}]" if section or page else ""
        ctx_parts.append(f"{header}\n{c['text']}")

    if ctx_parts:
        context_text = "\n---\n".join(ctx_parts)
        # Trim context to fit within budget
        remaining = budget - sum(len(s) for s in sections) - len(question) - 200
        if len(context_text) > remaining:
            context_text = context_text[:remaining] + "\n[...context truncated]"
        sections.append(f"## Retrieved Context\n{context_text}")

    # 3) Current question
    sections.append(f"## Current Question\n{question}")

    return "\n\n".join(sections)


# ── Parent-chunk expansion helper ─────────────────────────────────────

def expand_to_parents(
    child_chunks: List[Dict],
    parent_lookup: Dict[str, Dict],
) -> List[Dict]:
    """
    Given a list of retrieved child chunk dicts, look up and return
    their parent chunks for context expansion.
    """
    parent_ids_seen = set()
    parents = []
    for child in child_chunks:
        pid = child.get("parent_id")
        if pid and pid not in parent_ids_seen and pid in parent_lookup:
            parent_ids_seen.add(pid)
            parents.append(parent_lookup[pid])
    return parents


# ── Summarisation helper (uses Ollama) ────────────────────────────────

def summarise_history(
    memory: ConversationMemory,
    ollama_url: str = None,
) -> str:
    """
    Compress older turns into a summary to save tokens.
    Falls back to simple truncation if Ollama isn't available.
    """
    import requests as _req

    ollama_url = ollama_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    older_turns = memory.turns[:-SUMMARY_TRIGGER]
    if not older_turns:
        return memory.summary

    text = "\n".join(
        f"{'User' if t.role == 'user' else 'Assistant'}: {t.content}"
        for t in older_turns
    )

    prompt = (
        "Summarise this conversation fragment in 2-3 concise sentences. "
        "Preserve key questions asked and answers given:\n\n" + text
    )

    try:
        resp = _req.post(
            f"{ollama_url}/api/generate",
            json={"model": "llama3.1:8b", "prompt": prompt, "stream": False,
                   "options": {"temperature": 0.0, "num_predict": 200}},
            timeout=30,
        )
        resp.raise_for_status()
        summary = resp.json().get("response", "").strip()
        if summary:
            memory.summary = summary
            return summary
    except Exception:
        pass

    # Fallback: naive truncation
    memory.summary = text[:500] + "..."
    return memory.summary
