import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Explicitly load .env from the project root (parent directory of src/)
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
else:
    # Fallback to search in current working directory / environment
    load_dotenv(override=True)

def _get(name: str, default: str = "") -> str:
    """Read a setting from Streamlit secrets first, then from environment variables."""
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass  # streamlit not installed, or no secrets file (e.g. inside the API container)
    return os.getenv(name, default)

@dataclass
class Settings:
    # Base Paths
    BASE_DIR: Path = field(default_factory=lambda: BASE_DIR)
    DATA_RAW_DIR: Path = field(default_factory=lambda: BASE_DIR / "data" / "raw")
    DATA_PROCESSED_DIR: Path = field(default_factory=lambda: BASE_DIR / "data" / "processed")
    CHROMA_PERSIST_DIR: Path = field(default_factory=lambda: BASE_DIR / "chroma_db")
    PARENT_STORE_PATH: Path = field(default_factory=lambda: BASE_DIR / "data" / "parent_docstore.json")
    BM25_INDEX_PATH: Path = field(default_factory=lambda: BASE_DIR / "data" / "bm25_index.pkl")

    # Chunking Configuration (Hierarchical)
    PARENT_CHUNK_SIZE: int = 1200
    PARENT_CHUNK_OVERLAP: int = 150
    CHILD_CHUNK_SIZE: int = 300
    CHILD_CHUNK_OVERLAP: int = 50

    # Model Configuration
    EMBEDDING_MODEL_NAME: str = _get("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    RERANKER_MODEL_NAME: str = _get("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")
    
    # LLM (Groq Cloud via ChatGroq)
    GROQ_MODEL: str = _get("GROQ_MODEL", "openai/gpt-oss-120b")
    GROQ_API_KEY: str = _get("GROQ_API_KEY", "")
    GROQ_TEMPERATURE: float = float(_get("GROQ_TEMPERATURE", "0.0"))
    GROQ_MAX_TOKENS: int = int(_get("GROQ_MAX_TOKENS", "1024"))

    # Elasticsearch Configuration (BM25 sparse indexer)
    ELASTICSEARCH_URL: str = _get("ELASTICSEARCH_URL", "http://localhost:9200")
    ELASTICSEARCH_API_KEY: str = _get("ELASTICSEARCH_API_KEY", "")
    ELASTICSEARCH_INDEX: str = _get("ELASTICSEARCH_INDEX", "fincon_child_chunks")
    USE_ELASTICSEARCH: bool = _get("USE_ELASTICSEARCH", "false").lower() == "true"

    # Query transformation: LLM-based expansion on top of the rule-based one
    USE_LLM_EXPANSION: bool = _get("USE_LLM_EXPANSION", "false").lower() == "true"

    # Cloud bootstrap: number of dataset rows indexed on first start (small on purpose)
    BOOTSTRAP_SAMPLES: int = int(_get("BOOTSTRAP_SAMPLES", "1000"))

    # Retrieval Configuration
    RETRIEVAL_TOP_K_DENSE: int = 15
    RETRIEVAL_TOP_K_SPARSE: int = 15
    RERANK_TOP_K: int = 3
    RRF_K: int = 60

settings = Settings()
