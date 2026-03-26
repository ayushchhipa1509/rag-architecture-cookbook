"""
RAG Chain — wires the retriever + Groq LLM into a question-answering chain.
"""

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.config import GROQ_API_KEY, LLM_MODEL_NAME, LLM_TEMPERATURE


# ── Prompt Template ───────────────────────────────────────────────────────────

RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context.
Use ONLY the context below to answer the question. If the context does not contain enough information
to answer the question, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""

RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def get_llm() -> ChatGroq:
    """
    Create and return a Groq LLM instance.

    Returns:
        ChatGroq model instance.
    """
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        raise ValueError(
            "❌ GROQ_API_KEY not set! "
            "Please add your key to the .env file.\n"
            "   Get a free key at: https://console.groq.com/keys"
        )

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
    )
    print(f"🤖 LLM ready: {LLM_MODEL_NAME} (temp={LLM_TEMPERATURE})")
    return llm


def build_rag_chain(retriever) -> RetrievalQA:
    """
    Build a RetrievalQA chain that combines the retriever with the Groq LLM.

    Args:
        retriever: LangChain retriever instance.

    Returns:
        RetrievalQA chain ready for .invoke() calls.
    """
    llm = get_llm()

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )
    print("⛓️  RAG chain built successfully!")
    return rag_chain
