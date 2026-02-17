"""
Ingestion pipeline with improved chunking + observability.

Run:  python ingestion.py <path-to-pdf-or-txt>

Changes from original:
  • Hierarchical chunking (parent 2000 / child 500) with metadata enrichment
  • Structured JSON logging via the observability module
  • Stores both parent and child chunks in the vector store
  • Timing spans for each pipeline phase
"""

import os, sys, uuid, pathlib, io
from dotenv import load_dotenv

load_dotenv()

if len(sys.argv) < 2:
    print("Usage: python ingestion.py <path-to-pdf>")
    sys.exit(1)

PDF = pathlib.Path(sys.argv[1]).expanduser()
assert PDF.exists(), f"File not found: {PDF}"

# ── Observability ─────────────────────────────────────────────────────
from observability import get_logger, Tracer

logger = get_logger("ingestion")
tracer = Tracer(question=f"ingest:{PDF.name}")

# ── Neo4j config ──────────────────────────────────────────────────────
NEO4J_URI  = os.getenv("NEO4J_URI",  "neo4j://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "pass")

logger.info(f"Connecting to Neo4j at {NEO4J_URI} as {NEO4J_USER}")

# ── Load documents ────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import Neo4jVector, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from chunking import create_hierarchical_chunks, create_enhanced_chunks

with tracer.span("load_documents") as span:
    if PDF.suffix.lower() == ".pdf":
        try:
            loader = PyPDFLoader(str(PDF))
            docs = loader.load()
        except Exception as e:
            logger.warning(f"PDF loader failed, falling back to text: {e}")
            text = PDF.read_text(encoding="utf-8", errors="ignore")
            docs = [Document(page_content=text, metadata={"source": str(PDF)})]
    else:
        text = PDF.read_text(encoding="utf-8", errors="ignore")
        docs = [Document(page_content=text, metadata={"source": str(PDF)})]
    span.set_metadata(pages=len(docs), source=str(PDF))

logger.info(f"Loaded {len(docs)} page(s) from {PDF.name}")

# ── Chunk documents (hierarchical) ────────────────────────────────────
with tracer.span("chunk_documents") as span:
    parent_chunks, child_chunks = create_hierarchical_chunks(
        docs, source_name=PDF.name
    )
    # Also add the source-doc name to all chunks
    for c in parent_chunks + child_chunks:
        c.metadata["doc"] = PDF.name
    span.set_metadata(
        parent_chunks=len(parent_chunks),
        child_chunks=len(child_chunks),
    )

logger.info(
    f"Created {len(parent_chunks)} parent chunks + {len(child_chunks)} child chunks"
)

# ── Embeddings ────────────────────────────────────────────────────────
with tracer.span("load_embeddings_model"):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
    )

# ── Store in vector DB ────────────────────────────────────────────────
CHROMA_DIR = os.getenv(
    "CHROMA_DIR",
    str(pathlib.Path(__file__).resolve().parent / "chroma" / "books"),
)
os.makedirs(CHROMA_DIR, exist_ok=True)

# We store child chunks for retrieval (they're more precise) and keep
# parents alongside so we can expand context at query time.
all_chunks = child_chunks + parent_chunks

with tracer.span("store_vectors") as span:
    try:
        vector_store = Neo4jVector.from_documents(
            all_chunks, embeddings,
            url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS,
            node_label="Chunk",
            index_name="books",
            text_node_properties=["text"],
        )
        span.set_metadata(backend="neo4j")
        logger.info("Stored chunks in Neo4j vector index 'books'.")
    except Exception as e:
        logger.warning(f"Neo4j unavailable, using local Chroma: {e}")
        vector_store = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
        )
        vector_store.persist()
        span.set_metadata(backend="chroma")

# ── Image embeddings (unchanged) ──────────────────────────────────────
import fitz
import clip
from PIL import Image
from neo4j import GraphDatabase
import torch

try:
    clip_model, preprocess = clip.load("ViT-B/32")
    clip_model.eval()
except Exception as e:
    logger.warning(f"CLIP unavailable, skipping image embeddings: {e}")
    clip_model, preprocess = None, None

if PDF.suffix.lower() == ".pdf" and clip_model is not None:
    with tracer.span("store_image_embeddings") as span:
        try:
            doc = fitz.open(str(PDF))
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
            img_count = 0
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
                            "CREATE (i:Image {id: $id, embedding: $embedding, "
                            "page: $page, doc: $doc})",
                            id=str(uuid.uuid4()),
                            embedding=features,
                            page=page.number + 1,
                            doc=str(PDF.name),
                        )
                        img_count += 1
            span.set_metadata(images_stored=img_count)
            logger.info(f"Stored {img_count} image embeddings in Neo4j.")
        except Exception as e:
            logger.warning(f"Image ingestion skipped: {e}")

# ── Finalise ──────────────────────────────────────────────────────────
trace = tracer.finish()
logger.info(
    f"Ingestion complete in {trace.total_duration_ms:.0f}ms "
    f"({len(parent_chunks)} parents, {len(child_chunks)} children)"
)
