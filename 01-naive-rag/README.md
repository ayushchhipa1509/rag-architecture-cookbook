# 1️⃣ Naive RAG (Basic Retrieval)

This is the foundational Architecture of the RAG ecosystem. Use this when you are just starting out, building a simple proof-of-concept, or querying well-formatted, clean text.

## 🏗️ Architecture

```text
1. Data Pipeline:
Documents → Load → Split into chunks → Embed → Store in Vector DB

2. Query Pipeline:
User Query → Embed query → Retrieve matching chunks → Context + Query → LLM → Answer
```

## 🛠️ Components Used
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Local, Free)
- **Vector Store:** ChromaDB (Local)
- **LLM:** Groq (Fast, Free)
- **Framework:** LangChain

## 🚀 How to Run

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add API Key**
   Create a `.env` file in this folder and add your Groq key:
   ```env
   GROQ_API_KEY=gsk_your_key_here
   ```

3. **Start the UI**
   ```bash
   streamlit run app.py
   ```
   *You can upload PDF or TXT files directly through the Streamlit UI.*

## ⚠️ Limitations of this Pattern
As your document base grows, Naive RAG begins to fail:
- **Low Precision:** It retrieves chunks based purely on semantic similarity, which can bring in noisy, irrelevant data.
- **Lost Context:** Simple chunking cuts sentences in half, losing structural meaning.
- **Solution:** Move to [Pattern 2: Retrieve & Rerank](../02-retrieve-and-rerank-rag/) which adds a reranking step to filter out noise.
