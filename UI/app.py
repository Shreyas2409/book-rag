import streamlit as st, requests, tempfile, pathlib, os
from dotenv import load_dotenv; load_dotenv()

AGENT_API = os.getenv("AGENT_API", "http://localhost:7005")
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://neo4j:7687")
NEO4J_USER= os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS= os.getenv("NEO4J_PASS", "pass")

st.set_page_config(page_title="📚 Chat with your Book", layout="wide")
st.title("📚 Chat with your Book (text + images)")

# ---------- sidebar ingest --------------------------------------------------
with st.sidebar:
    up = st.file_uploader("Upload a book (PDF/TXT)", type=["pdf","txt"])
    if up and st.button("Ingest"):
        tmp = pathlib.Path(tempfile.mkdtemp())/up.name
        tmp.write_bytes(up.getbuffer())
        st.info("Running ingestion job … this may take a minute.")
        ingest_script = pathlib.Path(__file__).resolve().parents[1]/"ingestion.py"
        code = os.system(f"python3 '{ingest_script}' '{tmp}'")
        st.success("Ingest complete ✓" if code==0 else "Ingest failed!")

# ---------- chat ------------------------------------------------------------
if "history" not in st.session_state: st.session_state.history = []

query = st.chat_input("Ask me anything about your uploaded books …")
if query:
    st.session_state.history.append(("user", query))
    with st.spinner("Thinking …"):
        r = requests.post(f"{AGENT_API}/chat", json={"question": query}, timeout=120)
        answer = r.json()["answer"]
    st.session_state.history.append(("assistant", answer))

for role, msg in st.session_state.history:
    st.chat_message(role).markdown(msg)
