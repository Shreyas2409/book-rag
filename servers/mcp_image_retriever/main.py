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
driver     = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

try:
    clip_model, preprocess = clip.load("ViT-B/32")
    clip_model.eval()
except Exception:
    clip_model, preprocess = None, None

class Query(BaseModel):
    query: str
    top_k: int = 4

def search(vec, k):
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
