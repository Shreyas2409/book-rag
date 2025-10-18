from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import ChatOllama  # CHANGED: Use Ollama instead of OpenAI
from langchain.tools import BaseTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
import requests, os, traceback

VECTOR_URL = os.getenv("MCP_VECTOR_URL", "http://localhost:7001")
IMAGE_URL  = os.getenv("MCP_IMAGE_URL",  "http://localhost:7002")

class MCPTool(BaseTool):
    endpoint: str
    name: str
    description: str

    def _run(self, query: str, top_k: int = 6):
        r = requests.post(f"{self.endpoint}/invoke",
                          json={"query": query, "top_k": top_k}, timeout=60)
        r.raise_for_status()
        return r.json()

vector_tool = MCPTool(
    endpoint=VECTOR_URL,
    name="book_text_retriever",
    description=("Use to fetch passages from the book "
                 "when the user asks textual questions."))

image_tool = MCPTool(
    endpoint=IMAGE_URL,
    name="book_image_retriever",
    description=("Use when the user asks about diagrams, pictures or visual "
                 "appearance. Returns image IDs you can cite."))

# CHANGED: Use Ollama instead of OpenAI
llm = ChatOllama(
    model="llama3.1:8b",          # or "phi3:mini" for lighter model
    temperature=0,
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)

try:
    agent = create_tool_calling_agent(llm, tools=[vector_tool, image_tool], prompt="You are an expert search bot. Your goal is to find relevant information based on user queries.")
    executor = AgentExecutor(agent=agent, tools=[vector_tool, image_tool])
except Exception:
    agent = None
    executor = None

app = FastAPI(title="chat_agent")

class Q(BaseModel):
    question: str

@app.post("/chat")
def chat(q: Q):
    try:
        if executor is not None:
            result = executor.invoke({"input": q.question})
            return {"answer": result.get("output", "")}
    except Exception:
        traceback.print_exc()

    # Fallback: directly query tools and compose a simple answer
    try:
        vec = vector_tool._run(q.question)
    except Exception:
        vec = []
    try:
        imgs = image_tool._run(q.question)
    except Exception:
        imgs = []

    text_bits = []
    if isinstance(vec, list):
        for d in vec:
            t = d.get("text") if isinstance(d, dict) else None
            if t:
                text_bits.append(t)
    img_bits = []
    if isinstance(imgs, list):
        for i in imgs:
            if isinstance(i, dict):
                img_bits.append(f"Image {i.get('id','?')} on page {i.get('page','?')}")

    answer = "\n\n".join([
        "Relevant text: " + ("\n".join(text_bits) if text_bits else "<none>"),
        "Relevant images: " + ("\n".join(img_bits) if img_bits else "<none>")
    ])
    return {"answer": answer}
