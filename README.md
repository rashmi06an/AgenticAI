# Hybrid RAG + Web Search Agent

This project implements a hybrid search agent that prefers real-time web search and falls back to a document-based RAG system when web context is insufficient.

## Project Structure
- `app.py`: Streamlit-based user interface.
- `rag_agent.py`: RAG logic using FAISS and LangChain.
- `web_agent.py`: Serper API search implementation.
- `loaders.py`: Generic multi-format document loader.
- `config.py`: Hyperparameter management.
- `evaluate.py`: Automated experiment runner.
- `data_docs/`: Directory for source documents.
- `outputs/`: Directory for experiments, findings, and FAISS index.

## Setup Instructions

1. **Environment Setup**:
   It is recommended to use Python 3.10, 3.11, or 3.12.
   ```bash
   pip install -r rag_web_agent/requirements.txt
   ```

2. **API Keys**:
   Create a `.env` file in the `rag_web_agent/` directory (or the root) based on `.env.template`:
   ```env
   OPENAI_API_KEY=your_openai_key
   SERPER_API_KEY=your_serper_key
   ```

3. **Add Documents**:
   Place your PDF, DOCX, TXT, or MD files in the `data_docs/` folder.

## Running the Application

### Streamlit UI
```bash
streamlit run rag_web_agent/app.py
```

### Automated Experiments
```bash
PYTHONPATH=. python3 rag_web_agent/evaluate.py
```

## Features
- **Web-First Search**: Uses Serper API for real-time data.
- **Fallback Logic**: Heuristic and LLM-based checks to decide when to use local RAG.
- **Persistent Index**: FAISS index is stored and only rebuilt when documents change.
- **Hyperparameter Tuning**: Tunable chunk size, overlap, k, and chain types via UI.
- **Academic Evaluation**: Detailed logging of confidence, quality, and mode used.
