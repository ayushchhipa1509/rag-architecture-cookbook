"""
main.py — CLI entry point for the RAG system.

Usage:
    python main.py              # Query an existing index
    python main.py --index      # Index documents first, then query
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.vector_store import create_vector_store, load_vector_store, get_retriever
from src.rag_chain import build_rag_chain


def index_documents():
    """Load documents, split them, and create a vector store."""
    print("\n" + "=" * 60)
    print("📚 INDEXING DOCUMENTS")
    print("=" * 60 + "\n")

    # Step 1: Load documents
    documents = load_documents()
    if not documents:
        print("⚠️  No documents found in data/ folder. Please add PDF or TXT files.")
        return None

    # Step 2: Split into chunks
    chunks = split_documents(documents)

    # Step 3: Create vector store
    vector_store = create_vector_store(chunks)
    return vector_store


def query_loop():
    """Interactive Q&A loop."""
    print("\n" + "=" * 60)
    print("💬 RAG QUERY INTERFACE")
    print("=" * 60)
    print("Type your questions below. Type 'quit' or 'exit' to stop.\n")

    # Load vector store and build chain
    try:
        vector_store = load_vector_store()
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("💡 Run with --index flag first: python main.py --index")
        return

    retriever = get_retriever(vector_store)
    rag_chain = build_rag_chain(retriever)

    print("\n" + "-" * 60)

    while True:
        try:
            question = input("\n🟢 Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("\n👋 Goodbye!")
            break

        print("⏳ Thinking...")
        result = rag_chain.invoke({"query": question})

        # Display answer
        print(f"\n🤖 Answer: {result['result']}")

        # Display sources
        if result.get("source_documents"):
            print("\n📎 Sources:")
            for i, doc in enumerate(result["source_documents"], 1):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "N/A")
                print(f"   {i}. {os.path.basename(source)} (page {page})")


def main():
    parser = argparse.ArgumentParser(description="RAG System — Ask questions about your documents")
    parser.add_argument("--index", action="store_true", help="Index documents before querying")
    args = parser.parse_args()

    if args.index:
        result = index_documents()
        if result is None:
            print("\n⚠️  Indexing failed or no documents found. Fix the issue and try again.")
            return

    query_loop()


if __name__ == "__main__":
    main()
