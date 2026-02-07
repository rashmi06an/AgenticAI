import os
from typing import List, Tuple
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from rag_web_agent.loaders import load_documents_from_folder
from dotenv import load_dotenv

# Optional: keep load_dotenv for other potential needs, 
# although Serper is usually handled in config.py
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

class BM25Retriever:
    def __init__(self, docs: List[Document]):
        self.docs = docs
        # Tokenize documents for BM25
        self.tokenized_corpus = [doc.page_content.lower().split() for doc in docs]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def get_top_k(self, query: str, k: int = 3) -> List[Document]:
        tokenized_query = query.lower().split()
        top_n = self.bm25.get_top_n(tokenized_query, self.docs, n=k)
        return top_n

def answer_from_docs(
    query: str,
    docs_path: str,
    k: int = 3,
    chunk_size: int = 800,
    chunk_overlap: int = 80,
    # Kept parameters for compatibility, though not used for BM25
    temperature: float = 0.2,
    chain_type: str = "stuff",
) -> Tuple[str, List[Document]]:
    """
    Document-based retrieval answering using BM25 (Extractive).
    Returns: (best_chunk_content, source_documents)
    """
    docs: List[Document] = load_documents_from_folder(docs_path)
    
    if not docs:
        return "No documents found in the specified path.", []

    # Initialize BM25 Retriever
    retriever = BM25Retriever(docs)
    relevant_docs = retriever.get_top_k(query, k=k)

    if not relevant_docs:
        return "No relevant information found in local documents.", []

    # For extractive answering, we return the top chunk as the "answer"
    answer = relevant_docs[0].page_content
    
    return answer, relevant_docs

class RAGAgent:
    """Wrapper class for compatibility with ExperimentRunner."""
    def __init__(self, config: dict = None):
        self.config = config or {}

    def query(self, query_str: str) -> dict:
        k = self.config.get("k", 3)
        docs_path = "data_docs" # Default path
        
        answer, sources = answer_from_docs(
            query=query_str,
            docs_path=docs_path,
            k=k
        )
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": 0.7 if sources else 0.0,
            "mode": "docs"
        }
