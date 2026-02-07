import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Default Hyperparameters
DEFAULT_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 100, # Max 10%
    "k": 3,
    "temperature": 0.7,
    "chain_type": "stuff", # stuff, map_reduce, refine, map_rerank
    "search_type": "similarity"
}

# Paths
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "data_docs")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
FAISS_INDEX_PATH = os.path.join(OUTPUTS_DIR, "faiss_index")
EXPERIMENTS_FILE = os.path.join(OUTPUTS_DIR, "experiments.csv")
FINDINGS_FILE = os.path.join(OUTPUTS_DIR, "findings.md")

# Fallback Thresholds
WEB_CHUNKS_MIN_WORDS = 120
CONFIDENCE_THRESHOLD = 0.6
