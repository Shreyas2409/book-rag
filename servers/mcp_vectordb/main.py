"""
Text retriever MCP server – returns top-k text chunks with enriched metadata.

Improvements:
  • Returns chunk_type, parent_id, section, chunk_id alongside text
  • Returns similarity scores so the agent/dashboard can display them
  • Structured logging via observability module
  • Supports parent-chunk expansion query
"""

from fastapi import FastAPI
from pydantic import BaseModel
import os, pathlib, sys
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector, Chroma
from typing import List, Optional

# Add paths for shared modules (Docker: same dir, local: grandparent dir)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from observability import get_logger

logger = get_logger("mcp_vectordb")

app = FastAPI(
    title="book_text_retriever",
    description="Returns top-k text chunks with metadata and similarity scores.",
    version="2.0",
)

NEO4J_URI  = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")

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
        DB = Neo4jVector.from_existing_graph(
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


class Query(BaseModel):
    query: str
    top_k: int = 8


class ParentQuery(BaseModel):
    parent_ids: List[str]


@app.post("/invoke")
def invoke(body: Query):
    """Retrieve top-k chunks with similarity scores and enriched metadata."""
    db = init_db()
    if db is None:
        logger.error("No vector store available")
        return []

    # Use similarity_search_with_score for ranking visibility
    try:
        results = db.similarity_search_with_score(body.query, k=body.top_k)
    except Exception:
        # Fallback if with_score isn't supported
        docs = db.similarity_search(body.query, k=body.top_k)
        results = [(d, 0.0) for d in docs]

    output = []
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

    logger.info(
        f"Retrieved {len(output)} chunks for query: {body.query[:80]}",
        extra={"chunk_count": len(output)},
    )
    return output


@app.post("/parents")
def get_parents(body: ParentQuery):
    """Fetch parent chunks by their IDs for context expansion."""
    db = init_db()
    if db is None:
        return []

    # Search for parent chunks by ID
    # (This is a simple approach; a production system would use a separate lookup)
    results = []
    try:
        all_docs = db.similarity_search("", k=200)  # get all
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


@app.get("/health")
def health():
    db = init_db()
    return {"status": "ok" if db else "no_store", "backend": "neo4j" if NEO4J_URI else "chroma"}
