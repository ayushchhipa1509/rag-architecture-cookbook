"""
Document Loader — loads PDF and TXT files from the data directory.
"""

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from src.config import DATA_DIR


def load_documents(directory: str = None) -> list:
    """
    Load all PDF and TXT files from the given directory.

    Args:
        directory: Path to the folder containing documents. Defaults to DATA_DIR.

    Returns:
        List of LangChain Document objects.
    """
    if directory is None:
        directory = DATA_DIR

    documents = []

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created data directory: {directory}")
        print("   Please add PDF or TXT files to this folder and re-run.")
        return documents

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        if filename.lower().endswith(".pdf"):
            print(f"📄 Loading PDF: {filename}")
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())

        elif filename.lower().endswith(".txt"):
            print(f"📝 Loading TXT: {filename}")
            loader = TextLoader(filepath, encoding="utf-8")
            documents.extend(loader.load())

    print(f"✅ Loaded {len(documents)} document page(s) from {directory}")
    return documents
