from fastapi import FastAPI
from pydantic import BaseModel
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector

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

DB = Neo4jVector(
    url=NEO4J_URI, username=NEO4J_USER,
    password=NEO4J_PASS, index_name="*")
# The embedding model is not passed to Neo4jVector. This will cause an error when trying to use the vector store.
class Query(BaseModel):
    query: str
    top_k: int = 6

@app.post("/invoke")
def invoke(body: Query):
    docs = DB.similarity_search(body.query, k=body.top_k)
    return [{"text": d.page_content,
             "page": d.metadata.get("page", -1),
             "doc":  d.metadata.get("doc", "")} for d in docs]
