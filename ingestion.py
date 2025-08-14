"""
Run this once per PDF / book:
$ python ingestion.py <path-to-pdf>
"""

import os, sys, uuid, pathlib
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
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import HuggingFaceEmbeddings

loader = PyPDFLoader(str(PDF)) if PDF.suffix.lower() == ".pdf" else UnstructuredFileLoader(str(PDF))
docs   = loader.load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'device': 'cpu'}
)

vector_store = Neo4jVector.from_documents(
    chunks, embeddings,
    url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS,
    node_label="Chunk",
    index_name=f"book_{uuid.uuid4().hex[:6]}",
    text_node_properties=["text"]
)

# ---------- IMAGES -------------------------
import fitz
import clip_anytorch as clip
from PIL import Image
from neo4j import GraphDatabase
import torch

clip_model, preprocess = clip.load("ViT-B/32")
clip_model.eval()

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
doc    = fitz.open(str(PDF))

with driver.session() as sess:
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            base = doc.extract_image(xref)
            img_bytes = base["image"]
            image = Image.open(pathlib.BytesIO(img_bytes)).convert("RGB")

            image_input = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                features = clip_model.encode_image(image_input).tolist()[0]

            sess.run(
                "CREATE (i:Image {id: $id, vector: $vector})",
                id=str(uuid.uuid4()), vector=features
            )

print("Ingestion complete.")
