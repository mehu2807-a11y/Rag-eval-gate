"""
Ingestion pipeline: walk data/, chunk documents, embed them, and store in a
persistent Chroma collection.

Run directly to (re)build the index from scratch:
    python -m app.ingest
"""
import hashlib
from pathlib import Path
from typing import List, Tuple

import chromadb
from chromadb.utils import embedding_functions

from app import config


def load_documents(data_dir: Path) -> List[Tuple[str, str]]:
    """Return list of (source_path, raw_text) for every .txt/.md file under data_dir."""
    docs = []
    if not data_dir.exists():
        return docs
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() in (".txt", ".md") and path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                docs.append((str(path.relative_to(data_dir)), text))
    return docs


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Naive word-count chunker with overlap. Swap for a tokenizer-aware
    splitter (e.g. tiktoken) if you want exact token counts."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def _chunk_id(source: str, idx: int, text: str) -> str:
    h = hashlib.sha1(f"{source}:{idx}:{text[:50]}".encode()).hexdigest()[:12]
    return f"{source}::{idx}::{h}"


def get_client() -> chromadb.PersistentClient:
    config.CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_PATH))


def get_embedding_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )


def build_index(reset: bool = True) -> int:
    """Chunk + embed + store everything in data/. Returns number of chunks indexed."""
    client = get_client()
    ef = get_embedding_fn()

    if reset:
        try:
            client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME, embedding_function=ef
    )

    docs = load_documents(config.DATA_DIR)
    if not docs:
        print(f"No .txt/.md documents found in {config.DATA_DIR}. "
              f"Add some and re-run.")
        return 0

    ids, texts, metadatas = [], [], []
    for source, text in docs:
        for idx, chunk in enumerate(chunk_text(text)):
            ids.append(_chunk_id(source, idx, chunk))
            texts.append(chunk)
            metadatas.append({"source": source, "chunk_index": idx})

    if not texts:
        print("Documents found but produced zero chunks (empty files?).")
        return 0

    # Chroma add() has a batch size ceiling on some backends; chunk it defensively.
    batch_size = 512
    for i in range(0, len(texts), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=texts[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )

    print(f"Indexed {len(texts)} chunks from {len(docs)} documents "
          f"into collection '{config.COLLECTION_NAME}' at {config.CHROMA_PATH}")
    return len(texts)


if __name__ == "__main__":
    build_index(reset=True)
