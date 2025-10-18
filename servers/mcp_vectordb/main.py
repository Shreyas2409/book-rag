from fastapi import FastAPI
from pydantic import BaseModel
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector, Chroma
import pathlib

app = FastAPI(
    title="book_text_retriever",
    description="Returns top-k text chunks or captions from all ingested books.",
    version="1.0")

NEO4J_URI  = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")

# Use free HuggingFace embeddings
EMB = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'device': 'cpu'}
)

DB = None
CHROMA_DIR = os.getenv(
    "CHROMA_DIR",
    str(pathlib.Path(__file__).resolve().parents[2] / "chroma" / "books")
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
    except Exception:
        try:
            os.makedirs(CHROMA_DIR, exist_ok=True)
        except Exception:
            pass
        try:
            DB = Chroma(persist_directory=CHROMA_DIR, embedding_function=EMB)
        except Exception:
            DB = None
    return DB
class Query(BaseModel):
    query: str
    top_k: int = 6

@app.post("/invoke")
def invoke(body: Query):
    db = init_db()
    if db is None:
        return []
    docs = db.similarity_search(body.query, k=body.top_k)
    return [{"text": d.page_content,
             "page": d.metadata.get("page", -1),
             "doc":  d.metadata.get("doc", "")} for d in docs]
