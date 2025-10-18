"""
Run this once per PDF / book:
$ python ingestion.py <path-to-pdf>
"""

import os, sys, uuid, pathlib, io
from dotenv import load_dotenv

load_dotenv()

if len(sys.argv) < 2:
    print("Usage: python ingestion.py <path-to-pdf>")
    sys.exit(1)

PDF = pathlib.Path(sys.argv[1]).expanduser()
assert PDF.exists(), f"PDF not found: {PDF}"

# Use service hostname in Docker by default
NEO4J_URI  = os.getenv("NEO4J_URI",  "neo4j://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "pass")

print(f"Connecting to Neo4j at {NEO4J_URI} as {NEO4J_USER}")

# ---------- TEXT CHUNKS --------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

if PDF.suffix.lower() == ".pdf":
    loader = PyPDFLoader(str(PDF))
    docs = loader.load()
else:
    # Simple text ingestion without external dependencies
    text = PDF.read_text(encoding="utf-8", errors="ignore")
    docs = [Document(page_content=text, metadata={"source": str(PDF)})]
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

# add source document name for downstream display
for c in chunks:
    c.metadata["doc"] = PDF.name

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'device': 'cpu'}
)

# Persisted Chroma directory as a local fallback
CHROMA_DIR = os.getenv("CHROMA_DIR", str(pathlib.Path(__file__).resolve().parent / "chroma" / "books"))
os.makedirs(CHROMA_DIR, exist_ok=True)

# Try Neo4j first; fall back to local Chroma if unavailable
try:
    vector_store = Neo4jVector.from_documents(
        chunks, embeddings,
        url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS,
        node_label="Chunk",
        index_name="books",
        text_node_properties=["text"]
    )
    print("Stored text chunks in Neo4j vector index 'books'.")
except Exception as e:
    print(f"Neo4j unavailable, using local Chroma store: {e}")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vector_store.persist()

# ---------- IMAGES -------------------------
import fitz
import clip
from PIL import Image
from neo4j import GraphDatabase
import torch

try:
    clip_model, preprocess = clip.load("ViT-B/32")
    clip_model.eval()
except Exception as e:
    print(f"CLIP unavailable, skipping image embeddings: {e}")
    clip_model, preprocess = None, None

doc = fitz.open(str(PDF))

if clip_model is not None:
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        with driver.session() as sess:
            for page in doc:
                for img in page.get_images(full=True):
                    xref = img[0]
                    base = doc.extract_image(xref)
                    img_bytes = base["image"]
                    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                    image_input = preprocess(image).unsqueeze(0)
                    with torch.no_grad():
                        features = clip_model.encode_image(image_input).tolist()[0]

                    sess.run(
                        "CREATE (i:Image {id: $id, embedding: $embedding, page: $page, doc: $doc})",
                        id=str(uuid.uuid4()), embedding=features, page=page.number + 1, doc=str(PDF.name)
                    )
        print("Stored image embeddings in Neo4j.")
    except Exception as e:
        print(f"Neo4j unavailable for images; skipped image ingestion: {e}")

print("Ingestion complete.")
