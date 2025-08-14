# 📚 Chat with Your Book (Text + Images)

This project allows you to **upload a PDF or textbook** and interact with it using both **text** and **images**. It retrieves relevant passages and diagrams using **multi-modal search** and processes everything locally with **Ollama**.

---

## ✨ Features
- **Upload PDF/TXT** from the browser.
- **Automatic ingestion**:
  - Extracts text, chunks it, and stores embeddings in Neo4j.
  - Extracts images, encodes them with CLIP embeddings, and stores them in Neo4j.
- **Interactive chat interface** with memory.
- **Multi-modal retrieval**:
  - **Text retriever** for semantic search.
  - **Image retriever** for diagrams/figures.
- **Runs locally** using Ollama (llama3.1:8b by default).
- **One-click Docker Compose deployment**.

---

## 🏗 Architecture
1. **UI (Streamlit)** → Upload books & ask questions.
2. **Ingestion Service** (ingestion.py) → Parses books and stores embeddings.
3. **Retriever Services**:
   - mcp_vectordb: Text search in Neo4j.
   - mcp_image_retriever: Image search in Neo4j using CLIP.
4. **Agent API** → Uses LangChain + Ollama to call the right retriever.
5. **Neo4j** → Stores both text and image embeddings.

Architecture flow:
Streamlit UI → Agent API → (Text Retriever → Neo4j, Image Retriever → Neo4j) and Agent API → Ollama LLM

---

## 📦 Services Overview
| Service               | Tech Stack                  | Purpose |
|-----------------------|-----------------------------|---------|
| ui                    | Streamlit                   | Frontend for upload & chat |
| ingest_book           | Python + LangChain + CLIP   | Parses books & stores embeddings |
| mcp_vectordb          | FastAPI + Neo4jVector       | Text retrieval |
| mcp_image_retriever   | FastAPI + CLIP + Neo4j      | Image retrieval |
| agent_api             | FastAPI + LangChain Agent   | Orchestrates retrieval |
| neo4j                 | Neo4j DB                    | Vector storage |

---

## 🚀 Installation

### 1. Prerequisites
- Docker: https://docs.docker.com/get-docker/
- Ollama installed locally:
    ollama pull llama3.1:8b

### 2. Clone the repo
    git clone https://github.com/your-org/chat-with-book.git
    cd chat-with-book

### 3. Create .env file
    NEO4J_URI=neo4j://neo4j:7687
    NEO4J_USER=neo4j
    NEO4J_PASS=pass
    AGENT_API=http://localhost:7005

### 4. Start all services
    docker compose -f docker.yaml up --build

---

## 📖 Ingesting a Book
From the UI sidebar, upload a PDF/TXT and click "Ingest".  
Or via terminal:
    docker compose run ingest_book python ingestion.py /books/mybook.pdf

---

## 💬 Chatting with the Book
Visit:
    http://localhost:8501

Example prompts:
- Summarize chapter 2
- Show me diagrams of the solar system

---

## 🔍 Retrieval Logic

**Text Retrieval (mcp_vectordb)**
- Uses BAAI/bge-base-en-v1.5 embeddings.
- Chunks stored in Neo4j.
- Queries by semantic similarity.

**Image Retrieval (mcp_image_retriever)**
- Uses CLIP ViT-B/32.
- Vectors stored as Image nodes in Neo4j.
- Queries by cosine similarity.

---

## 🧠 Agent Orchestration
- LangChain Agent calls book_text_retriever for text queries.
- Calls book_image_retriever for visual queries.
- Runs locally with Ollama for privacy.

---

## ⚠ Known Issues
- mcp_vectordb must pass its embedding model into Neo4jVector or it errors.
- Neo4j plugin graph-data-science must be enabled.
- Ensure consistent image vector property naming (vector vs embedding).

---

## 📜 License
MIT License
