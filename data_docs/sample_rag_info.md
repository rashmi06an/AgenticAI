# RAG System Overview
Retrieval-Augmented Generation (RAG) is a technique for enhancing the output of a large language model with information from a specific, often private or dynamic, data source.

## How it works
1. **Load**: Information is loaded into the system.
2. **Chunk**: Documents are split into smaller pieces.
3. **Embed**: Text is converted into numerical vectors.
4. **Retrieve**: Relevant chunks are found using vector similarity.
5. **Generate**: The LLM uses the retrieved context to answer.

FAISS is a library for efficient similarity search. It is developed by Facebook AI Research.
