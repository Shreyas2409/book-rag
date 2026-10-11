"""
Ingestion pipeline with improved chunking + observability.

Run:  python ingestion.py <path-to-pdf-or-txt>

Improvements (v3):
  • Hierarchical chunking (parent 2000 / child 500) with metadata enrichment
  • Structured JSON logging via the observability module
  • Stores both parent and child chunks in the vector store
  • Timing spans for each pipeline phase
  • Deduplication: uses MERGE to avoid duplicate chunks on re-ingestion
  • Creates Neo4j fulltext index for hybrid search support
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
        # ── Deduplication: check existing chunk_ids and skip duplicates ──
        from neo4j import GraphDatabase as _GDB
        existing_ids = set()
        try:
            _dedup_driver = _GDB.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
            with _dedup_driver.session() as sess:
                records = sess.run(
                    "MATCH (c:Chunk) WHERE c.doc = $doc RETURN c.chunk_id AS cid",
                    doc=PDF.name,
                ).data()
                existing_ids = {r["cid"] for r in records if r.get("cid")}
            _dedup_driver.close()
        except Exception:
            pass  # Neo4j not available yet, skip dedup check

        if existing_ids:
            before = len(all_chunks)
            all_chunks = [
                c for c in all_chunks
                if c.metadata.get("chunk_id") not in existing_ids
            ]
            skipped = before - len(all_chunks)
            if skipped > 0:
                logger.info(f"Deduplication: skipped {skipped} existing chunks")
            span.set_metadata(skipped_duplicates=skipped)

        if all_chunks:
            vector_store = Neo4jVector.from_documents(
                all_chunks, embeddings,
                url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS,
                node_label="Chunk",
                index_name="books",
                # Neo4jVector.__init__ takes `text_node_property` (singular
                # str), not `text_node_properties` — the plural kwarg raised
                # TypeError on every call, silently falling back to Chroma.
                text_node_property="text",
            )

            span.set_metadata(backend="neo4j", chunks_stored=len(all_chunks))
            logger.info(f"Stored {len(all_chunks)} chunks in Neo4j vector index 'books'.")
        else:
            logger.info("All chunks already exist in Neo4j, nothing to store.")
            span.set_metadata(backend="neo4j", chunks_stored=0)

    except Exception as e:
        logger.warning(f"Neo4j unavailable, using local Chroma: {e}")
        vector_store = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
        )
        vector_store.persist()
        span.set_metadata(backend="chroma", chunks_stored=len(all_chunks))

# ── Create fulltext index for hybrid search ───────────────────────────
with tracer.span("create_fulltext_index") as span:
    try:
        from neo4j import GraphDatabase as _GDB2
        _idx_driver = _GDB2.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        with _idx_driver.session() as sess:
            # Create fulltext index on Chunk text for keyword/BM25 search
            sess.run(
                "CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS "
                "FOR (c:Chunk) ON EACH [c.text]"
            )
            logger.info("Fulltext index 'chunk_fulltext' created/verified.")
            span.set_metadata(index="chunk_fulltext", status="ok")
        _idx_driver.close()
    except Exception as e:
        logger.warning(f"Fulltext index creation skipped: {e}")
        span.set_metadata(index="chunk_fulltext", status="skipped", error=str(e))

# ── Trigger BM25 index refresh (best-effort, non-blocking) ───────────
# FR-007: after new chunks are stored, notify mcp_vectordb to rebuild
# its in-memory BM25 index. Failures are caught and logged — they do NOT
# fail ingestion (the BM25 index will be rebuilt on the next service restart).
with tracer.span("bm25_refresh_notify") as span:
    try:
        import requests as _requests
        MCP_VECTOR_URL = os.getenv("MCP_VECTOR_URL", "http://localhost:7001")
        resp = _requests.post(f"{MCP_VECTOR_URL}/bm25/refresh", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        span.set_metadata(
            status=data.get("status", "unknown"),
            chunks_indexed=data.get("chunks_indexed", 0),
            duration_ms=data.get("duration_ms", 0),
        )
        logger.info(
            f"BM25 index refreshed: {data.get('chunks_indexed', 0)} chunks "
            f"in {data.get('duration_ms', 0):.1f}ms"
        )
    except Exception as e:
        logger.warning(f"BM25 index refresh notify failed (non-fatal): {e}")
        span.set_metadata(status="error", error=str(e))

# ── Image embeddings ──────────────────────────────────────────────────
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
            img_skipped = 0
            with driver.session() as sess:
                # Get existing image IDs for this doc to avoid duplicates
                existing_img_pages = set()
                try:
                    records = sess.run(
                        "MATCH (i:Image) WHERE i.doc = $doc RETURN i.page AS page",
                        doc=str(PDF.name),
                    ).data()
                    existing_img_pages = {r["page"] for r in records if r.get("page")}
                except Exception:
                    pass

                for page in doc:
                    # Skip if images from this page already exist
                    if (page.number + 1) in existing_img_pages:
                        img_skipped += 1
                        continue

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
            span.set_metadata(images_stored=img_count, images_skipped=img_skipped)
            logger.info(f"Stored {img_count} image embeddings in Neo4j (skipped {img_skipped} existing pages).")
        except Exception as e:
            logger.warning(f"Image ingestion skipped: {e}")

# ── Finalise ──────────────────────────────────────────────────────────
trace = tracer.finish()
logger.info(
    f"Ingestion complete in {trace.total_duration_ms:.0f}ms "
    f"({len(parent_chunks)} parents, {len(child_chunks)} children)"
)
