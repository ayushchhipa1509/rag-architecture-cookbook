"""
Configuration module — loads environment variables and defines constants.
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── LLM Settings ──────────────────────────────────────────────────────────────
LLM_MODEL_NAME = "llama-3.1-8b-instant"  # Fast Groq model
LLM_TEMPERATURE = 0.2

# ── Embedding Settings ────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Free HuggingFace model

# ── Text Splitting ────────────────────────────────────────────────────────────
CHUNK_SIZE = 1000        # characters per chunk
CHUNK_OVERLAP = 200      # overlap between consecutive chunks

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# ── Retriever Settings ────────────────────────────────────────────────────────
RETRIEVER_K = 4  # Number of chunks to retrieve
