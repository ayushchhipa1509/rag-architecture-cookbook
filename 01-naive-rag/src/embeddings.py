"""
Embeddings — provides the HuggingFace embedding model.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL_NAME


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Create and return a HuggingFace embedding model instance.
    Uses the all-MiniLM-L6-v2 model by default (free, local, no API key).

    Returns:
        HuggingFaceEmbeddings model instance.
    """
    print(f"🧠 Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("✅ Embedding model loaded successfully")
    return embeddings
