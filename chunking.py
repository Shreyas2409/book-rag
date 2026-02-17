"""
Advanced chunking strategies for the RAG pipeline.

Improvements over the original 500-char naive splitter:
  1. Semantic-aware splitting that respects paragraph / section boundaries
  2. Hierarchical chunking: parent (2000 chars) ⟶ child (500 chars)
     so retrieval can fetch the small child but expand to the parent
     for richer LLM context.
  3. Metadata enrichment: chapter detection, section headers, page nums
  4. Sliding-window overlap (200 chars) to avoid cutting mid-sentence
"""

import re, uuid, hashlib
from typing import List, Dict, Optional, Tuple
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── Configuration ─────────────────────────────────────────────────────

PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100

# Section header patterns (common in textbooks / PDFs)
SECTION_PATTERN = re.compile(
    r'^(?:'
    r'(?:Chapter|CHAPTER)\s+\d+'                   # Chapter 1
    r'|(?:Section|SECTION)\s+[\d.]+'               # Section 1.2
    r'|\d+\.\d+(?:\.\d+)?\s+[A-Z]'                # 1.2 Title
    r'|(?:Part|PART)\s+[IVXLCDM\d]+'              # Part III
    r'|[A-Z][A-Z\s]{4,}$'                          # ALL CAPS LINE (likely heading)
    r')',
    re.MULTILINE,
)


# ── Helper: detect section headers ────────────────────────────────────

def _detect_current_section(text: str) -> str:
    """Return the last section-level heading found in `text`."""
    matches = SECTION_PATTERN.findall(text)
    return matches[-1].strip() if matches else ""


def _stable_id(content: str, source: str) -> str:
    """Deterministic chunk ID from content hash."""
    h = hashlib.sha256(f"{source}::{content[:200]}".encode()).hexdigest()[:12]
    return h


# ── Public API ────────────────────────────────────────────────────────

def create_hierarchical_chunks(
    docs: List[Document],
    source_name: str = "",
) -> Tuple[List[Document], List[Document]]:
    """
    Take raw LangChain Documents (one per page typically) and produce:
      - parent_chunks: large, context-rich chunks (2000 chars)
      - child_chunks : small, retrieval-optimised chunks (500 chars)
    Each child has metadata pointing to its parent_id so the agent
    can expand context at query time.
    """

    # Step 1 – create parent-level chunks
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        length_function=len,
    )
    parent_docs = parent_splitter.split_documents(docs)

    # Step 2 – enrich parent metadata & assign IDs
    running_section = ""
    parent_chunks: List[Document] = []
    for idx, pdoc in enumerate(parent_docs):
        sec = _detect_current_section(pdoc.page_content)
        if sec:
            running_section = sec
        pid = _stable_id(pdoc.page_content, source_name)
        pdoc.metadata.update({
            "chunk_type": "parent",
            "chunk_id": pid,
            "chunk_index": idx,
            "section": running_section,
            "doc": source_name or pdoc.metadata.get("doc", ""),
        })
        parent_chunks.append(pdoc)

    # Step 3 – split each parent into child chunks
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        length_function=len,
    )

    child_chunks: List[Document] = []
    for pdoc in parent_chunks:
        children = child_splitter.split_documents(
            [Document(page_content=pdoc.page_content, metadata=pdoc.metadata.copy())]
        )
        for cidx, cdoc in enumerate(children):
            cid = _stable_id(cdoc.page_content, source_name + f"_c{cidx}")
            cdoc.metadata.update({
                "chunk_type": "child",
                "chunk_id": cid,
                "parent_id": pdoc.metadata["chunk_id"],
                "child_index": cidx,
            })
            child_chunks.append(cdoc)

    return parent_chunks, child_chunks


def create_enhanced_chunks(
    docs: List[Document],
    source_name: str = "",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Single-tier enhanced chunking (no parent-child) but with better
    defaults and metadata.  Drop-in replacement for the original splitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)

    running_section = ""
    for idx, chunk in enumerate(chunks):
        sec = _detect_current_section(chunk.page_content)
        if sec:
            running_section = sec
        cid = _stable_id(chunk.page_content, source_name)
        chunk.metadata.update({
            "chunk_id": cid,
            "chunk_index": idx,
            "section": running_section,
            "doc": source_name or chunk.metadata.get("doc", ""),
            "char_count": len(chunk.page_content),
        })

    return chunks
