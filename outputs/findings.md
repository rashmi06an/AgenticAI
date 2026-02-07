# Academic Findings: Hybrid RAG + Web Search Agent

## 1. Introduction to RAG
Retrieval-Augmented Generation (RAG) is a framework that enhances Large Language Models (LLMs) by integrating external, authoritative knowledge bases. Instead of relying solely on the LLM's static training data, RAG retrieves relevant document chunks to provide context-aware and verifiable answers.

## 2. Architecture & Decision Logic
The system implements a **Web-First hybrid approach**:
1. **Web Search**: Queries are first sent to the Serper API.
2. **Fallback Logic**: If the web context is sparse (<120 words), the LLM confidence is low (<0.6), or the context is irrelevant, the system falls back to the Document RAG.
3. **Document RAG**: Local documents are chunked and indexed in a FAISS vector database.

## 3. Vector Database & Semantic Search
- **Vector DB**: FAISS (Facebook AI Similarity Search) is utilized for efficient similarity search in high-dimensional embedding spaces.
- **Search Type**: Similarity search (Cosine similarity/L2) is used to retrieve the top-k most relevant document chunks based on query embeddings.
- **Persistence**: The FAISS index is persisted to disk and only rebuilt if the source document folder's hash changes, optimizing performance.

## 4. Hyperparameter Tuning Observations
Through experimentation with variables like `chunk_size`, `k`, and `chain_type`, we observed:
- **Chunk Size**: Smaller chunks (c. 500) provide better granularity but may lack context. Larger chunks (c. 1000) offer more context but risk noise.
- **Top-k (k)**: Increasing `k` improves recall but can hit LLM context limits or introduce irrelevant data.
- **Chain Type**: `stuff` is fast and effective for short contexts. `map_reduce` or `refine` are superior for complex, multi-document reasoning but incur higher latency.

## 5. Conclusion
The hybrid approach significantly increases robustness. Web search provides real-time information, while the RAG fallback ensures domain-specific accuracy when web snippets are insufficient. The best configuration generally involves a balanced `chunk_size` of 1000 and `k=3` for optimal relevance and noise reduction.
