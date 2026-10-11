from fastapi import FastAPI
from pydantic import BaseModel
import os, numpy as np, clip, torch
from neo4j import GraphDatabase

app = FastAPI(
    title="book_image_retriever",
    description="Vector-search over Image nodes. Returns IDs + page numbers.",
    version="1.0")

NEO4J_URI  = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")

_driver = None


def _get_driver():
    """Lazily connect to Neo4j; never crash the service on a bad/missing URI."""
    global _driver
    if _driver is not None:
        return _driver
    if not NEO4J_URI:
        return None
    try:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    except Exception:
        _driver = None
    return _driver


try:
    clip_model, preprocess = clip.load("ViT-B/32")
    clip_model.eval()
except Exception:
    clip_model, preprocess = None, None

class Query(BaseModel):
    query: str
    top_k: int = 4

def search(vec, k):
    driver = _get_driver()
    if driver is None:
        return []
    q = """
    MATCH (i:Image)
    WITH i, gds.similarity.cosine(i.embedding, $v) AS score
    ORDER BY score DESC LIMIT $k
    RETURN i.id AS id, i.page AS page, i.doc AS doc, score
    """
    try:
        with driver.session() as sess:
            rows = sess.run(q, v=vec, k=k).data()
        return rows
    except Exception:
        return []

@app.post("/invoke")
def invoke(body: Query):
    if clip_model is None:
        return []
    with torch.no_grad():
        txt_vec = clip_model.encode_text(
                clip.tokenize(body.query)).cpu().numpy()[0]
    return search(txt_vec.tolist(), body.top_k)


@app.get("/health")
def health():
    return {"status": "ok" if _get_driver() else "no_store", "clip": clip_model is not None}
