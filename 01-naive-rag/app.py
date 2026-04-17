"""
app.py — Streamlit web interface for the RAG system.

Usage:
    streamlit run app.py
"""

import streamlit as st
import os
import sys
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import DATA_DIR, CHROMA_DIR
from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store, load_vector_store, get_retriever
from src.rag_chain import build_rag_chain


# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📚 RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }

    .source-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        border-left: 4px solid #667eea;
        font-size: 0.85rem;
    }

    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .status-ready {
        background: #d4edda;
        color: #155724;
    }

    .status-empty {
        background: #fff3cd;
        color: #856404;
    }

    .sidebar-section {
        background: rgba(102, 126, 234, 0.05);
        border-radius: 10px;
        padding: 16px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "indexed" not in st.session_state:
    st.session_state.indexed = os.path.exists(CHROMA_DIR)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📁 Document Manager")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload documents to be indexed and queried.",
    )

    if uploaded_files:
        os.makedirs(DATA_DIR, exist_ok=True)
        for uploaded_file in uploaded_files:
            filepath = os.path.join(DATA_DIR, uploaded_file.name)
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"✅ {len(uploaded_files)} file(s) saved to data/")

    st.divider()

    # Index button
    if st.button("🔄 Index Documents", use_container_width=True, type="primary"):
        with st.spinner("Indexing documents..."):
            try:
                # Clear old index
                if os.path.exists(CHROMA_DIR):
                    shutil.rmtree(CHROMA_DIR)

                docs = load_documents()
                if not docs:
                    st.error("⚠️ No documents found. Upload files first!")
                else:
                    chunks = split_documents(docs)
                    embedding_model = get_embedding_model()
                    vector_store = create_vector_store(chunks, embedding_model)
                    retriever = get_retriever(vector_store)
                    st.session_state.rag_chain = build_rag_chain(retriever)
                    st.session_state.indexed = True
                    st.success(f"✅ Indexed {len(docs)} page(s) → {len(chunks)} chunks")
            except Exception as e:
                st.session_state.indexed = False
                st.session_state.rag_chain = None
                st.error(f"❌ Error: {e}")

    st.divider()

    # Status
    st.markdown("### 📊 Status")
    if st.session_state.indexed:
        st.markdown('<span class="status-badge status-ready">● Index Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-empty">○ No Index</span>', unsafe_allow_html=True)

    # Data folder contents
    if os.path.exists(DATA_DIR):
        files = os.listdir(DATA_DIR)
        if files:
            st.markdown(f"**{len(files)} file(s) in data/**")
            for f in files:
                size = os.path.getsize(os.path.join(DATA_DIR, f))
                st.markdown(f"- `{f}` ({size / 1024:.1f} KB)")

    st.divider()
    st.caption("Built with LangChain + ChromaDB + Groq")


# ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">📚 RAG System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about your documents — powered by AI</p>', unsafe_allow_html=True)


# Load RAG chain if index exists but chain not in session
if st.session_state.indexed and st.session_state.rag_chain is None:
    try:
        with st.spinner("Loading index..."):
            vector_store = load_vector_store()
            retriever = get_retriever(vector_store)
            st.session_state.rag_chain = build_rag_chain(retriever)
    except Exception as e:
        st.warning(f"⚠️ Could not load index: {e}")
        st.session_state.indexed = False


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        if not st.session_state.indexed or st.session_state.rag_chain is None:
            response = "⚠️ Please upload documents and click **Index Documents** first!"
            st.markdown(response)
        else:
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.rag_chain.invoke({"query": prompt})
                    response = result["result"]
                    st.markdown(response)

                    # Show sources
                    if result.get("source_documents"):
                        with st.expander("📎 View Sources", expanded=False):
                            for i, doc in enumerate(result["source_documents"], 1):
                                source = os.path.basename(doc.metadata.get("source", "Unknown"))
                                page = doc.metadata.get("page", "N/A")
                                snippet = doc.page_content[:200] + "..."
                                st.markdown(
                                    f'<div class="source-card">'
                                    f'<strong>{i}. {source}</strong> (page {page})<br>'
                                    f'<em>{snippet}</em></div>',
                                    unsafe_allow_html=True,
                                )
                except Exception as e:
                    response = f"❌ Error: {e}"
                    st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
