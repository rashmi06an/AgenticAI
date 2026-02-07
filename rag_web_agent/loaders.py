import os
import hashlib
import logging
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredFileLoader
)
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenericDocumentLoader:
    def __init__(self, directory_path: str):
        self.directory_path = directory_path

    def get_data_hash(self) -> str:
        """Calculate MD5 hash of the data directory to detect changes."""
        hash_md5 = hashlib.md5()
        if not os.path.exists(self.directory_path):
            return ""
        
        # Sort files to ensure consistent hash
        files = sorted([f for f in os.listdir(self.directory_path) if os.path.isfile(os.path.join(self.directory_path, f))])
        
        for filename in files:
            filepath = os.path.join(self.directory_path, filename)
            hash_md5.update(filename.encode())
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        
        return hash_md5.hexdigest()

    def load_documents(self) -> List[Document]:
        """Load all documents from the directory using appropriate loaders."""
        documents = []
        if not os.path.exists(self.directory_path):
            logger.warning(f"Directory {self.directory_path} does not exist.")
            return documents

        for filename in os.listdir(self.directory_path):
            filepath = os.path.join(self.directory_path, filename)
            if not os.path.isfile(filepath):
                continue

            ext = os.path.splitext(filename)[1].lower()
            try:
                if ext == ".pdf":
                    loader = PyPDFLoader(filepath)
                elif ext == ".docx":
                    loader = Docx2txtLoader(filepath)
                elif ext in [".txt", ".md"]:
                    loader = TextLoader(filepath)
                else:
                    logger.info(f"Using UnstructuredFileLoader fallback for {filename}")
                    loader = UnstructuredFileLoader(filepath)
                
                documents.extend(loader.load())
            except Exception as e:
                logger.error(f"Failed to load {filename}: {str(e)}")
                # Skip file as per requirement
                continue

        logger.info(f"Loaded {len(documents)} document pages/chunks from {self.directory_path}")
        return documents
def load_documents_from_folder(folder_path: str):
    """
    Convenience wrapper: loads all supported documents from a folder.
    Returns a list of LangChain Document objects.
    """
    loader = GenericDocumentLoader(folder_path)
    return loader.load_documents()

