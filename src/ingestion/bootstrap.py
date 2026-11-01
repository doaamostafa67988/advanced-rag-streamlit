"""Cloud bootstrap: build a small hybrid index on first start.

Streamlit Community Cloud has no persistent disk and the full 75k corpus is far
too large to index there, so we index a small streamed subset of the dataset
plus any small files committed under data/raw/.
"""
from typing import Any, Dict, List

from src.config import settings
from src.ingestion.hierarchical import HierarchicalChunker
from src.ingestion.indexer import Indexer
from src.ingestion.parser import Document, DocumentParser

DATASET_NAME = "BAAI/IndustryInstruction_Finance-Economics"
MAX_LOCAL_FILE_BYTES = 5 * 1024 * 1024  # skip huge raw files (e.g. the 75k CSV)


def index_ready(indexer: Indexer) -> bool:
    """True when the vector index, parent store and BM25 index all exist."""
    try:
        count = indexer.collection.count() if indexer.collection else 0
    except Exception:
        count = 0
    return count > 0 and settings.PARENT_STORE_PATH.exists() and settings.BM25_INDEX_PATH.exists()


def _sanitize(meta: Dict[str, Any]) -> Dict[str, Any]:
    """ChromaDB only accepts str/int/float/bool metadata values."""
    clean = {}
    for k, v in meta.items():
        if v is None:
            continue
        clean[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
    return clean


def _local_documents() -> List[Document]:
    """Parse small files committed under data/raw/ (e.g. regulatory standards JSON)."""
    docs: List[Document] = []
    raw_dir = settings.DATA_RAW_DIR
    if not raw_dir.exists():
        return docs
    for f in raw_dir.glob("**/*"):
        if f.is_file() and f.stat().st_size <= MAX_LOCAL_FILE_BYTES:
            try:
                docs.extend(DocumentParser.parse_file(f))
            except Exception as e:
                print(f"[Bootstrap] Skipping {f.name}: {e}")
    return docs


def _dataset_documents(num_samples: int) -> List[Document]:
    """Stream the first N rows of the finance dataset (no full download)."""
    from datasets import load_dataset
    from src.ingestion.load_baai_dataset import extract_instruction_response

    stream = load_dataset(DATASET_NAME, split="train", streaming=True).take(num_samples)
    docs: List[Document] = []
    for idx, item in enumerate(stream):
        instruction, response = extract_instruction_response(
            item.get("conversations"), item.get("instruction", "")
        )
        if not instruction and not response:
            continue
        content = (
            f"Topic / Instruction: {instruction}\n\nFinancial Analysis & Detail:\n{response}"
            if instruction else response
        )
        docs.append(Document(
            page_content=content,
            metadata={
                "source": "baai_finance_economics",
                "record_id": f"baai_fin_{idx:06d}",
                "title": instruction[:120] if instruction else f"Finance Topic {idx + 1}",
                "lang": str(item.get("lang", "en")),
            },
        ))
    return docs


def reset_indexes(indexer: Indexer) -> None:
    """Drop the vector collection and stale files so a rebuild starts clean."""
    if indexer.collection is not None:
        indexer.client.delete_collection("fincon_child_chunks")
        indexer.collection = indexer.client.get_or_create_collection(
            name="fincon_child_chunks", embedding_function=indexer.embedding_fn
        )
    for path in (settings.PARENT_STORE_PATH, settings.BM25_INDEX_PATH):
        path.unlink(missing_ok=True)
    indexer._bm25_data = None


def build_index(indexer: Indexer, num_samples: int = settings.BOOTSTRAP_SAMPLES) -> Dict[str, int]:
    """Build parent store + vector + BM25 indexes, reusing the given Indexer."""
    docs = _local_documents()
    n_local = len(docs)
    if num_samples > 0:
        docs.extend(_dataset_documents(num_samples))
    if not docs:
        raise RuntimeError("No documents found: add files to data/raw/ or set BOOTSTRAP_SAMPLES > 0.")

    chunker = HierarchicalChunker()
    child_chunks, parent_store = chunker.process_documents(docs)
    for chunk in child_chunks:
        chunk["metadata"] = _sanitize(chunk["metadata"])
    for parent in parent_store.values():
        parent["metadata"] = _sanitize(parent["metadata"])

    chunker.save_parent_store(parent_store)
    indexer.build_indexes(child_chunks)
    return {
        "local_docs": n_local,
        "total_docs": len(docs),
        "parents": len(parent_store),
        "children": len(child_chunks),
    }


def ensure_index(indexer: Indexer) -> bool:
    """Build the index only if missing. Returns True if a build happened."""
    if index_ready(indexer):
        return False
    build_index(indexer)
    return True
