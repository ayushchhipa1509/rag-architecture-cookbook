"""
Vector Store — create, persist, and query a ChromaDB vector database.
"""

import os
from langchain_community.vectorstores import Chroma
from src.config import CHROMA_DIR, RETRIEVER_K
from src.embeddings import get_embedding_model


def create_vector_store(chunks: list, embedding_model=None) -> Chroma:
    """
    Create a ChromaDB vector store from document chunks and persist to disk.

    Args:
        chunks: List of chunked Document objects.
        embedding_model: Embedding model instance. Auto-loaded if None.

    Returns:
        Chroma vector store instance.
    """
    if embedding_model is None:
        embedding_model = get_embedding_model()

    print(f"📦 Creating vector store with {len(chunks)} chunks...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR,
    )
    print(f"✅ Vector store created and persisted to {CHROMA_DIR}")
    return vector_store


def load_vector_store(embedding_model=None) -> Chroma:
    """
    Load an existing ChromaDB vector store from disk.

    Args:
        embedding_model: Embedding model instance. Auto-loaded if None.

    Returns:
        Chroma vector store instance.

    Raises:
        FileNotFoundError: If no vector store exists on disk.
    """
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"No vector store found at {CHROMA_DIR}. "
            "Please index documents first."
        )

    if embedding_model is None:
        embedding_model = get_embedding_model()

    print(f"📂 Loading vector store from {CHROMA_DIR}")
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model,
    )
    print("✅ Vector store loaded successfully")
    return vector_store


def get_retriever(vector_store: Chroma = None):
    """
    Get a retriever from the vector store for similarity search.

    Args:
        vector_store: Chroma instance. Loaded from disk if None.

    Returns:
        LangChain retriever object.
    """
    if vector_store is None:
        vector_store = load_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )
    print(f"🔍 Retriever ready (top-{RETRIEVER_K} chunks)")
    return retriever
