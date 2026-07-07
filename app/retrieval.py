"""Top-k retrieval against the Chroma collection built by app/ingest.py."""
from typing import List, Dict

from app import config
from app.ingest import get_client, get_embedding_fn


def retrieve(query: str, top_k: int = None) -> List[Dict]:
    """Return list of {text, source, chunk_index, distance} for the top_k
    most relevant chunks to `query`."""
    top_k = top_k or config.TOP_K
    client = get_client()
    ef = get_embedding_fn()
    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME, embedding_function=ef
    )

    if collection.count() == 0:
        return []

    result = collection.query(query_texts=[query], n_results=top_k)

    hits = []
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0] if result.get("distances") else [None] * len(docs)

    for text, meta, dist in zip(docs, metas, dists):
        hits.append({
            "text": text,
            "source": meta.get("source"),
            "chunk_index": meta.get("chunk_index"),
            "distance": dist,
        })
    return hits
