"""ChromaDB wrapper: persistent collection + retrieval."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import settings
from .embeddings import LocalEmbeddingFunction


@lru_cache(maxsize=1)
def _client() -> chromadb.api.ClientAPI:
    return chromadb.PersistentClient(
        path=settings.chroma_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_collection():
    return _client().get_or_create_collection(
        name=settings.collection_name,
        embedding_function=LocalEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(ids: list[str], texts: list[str], metadatas: list[dict[str, Any]]) -> None:
    if not ids:
        return
    col = get_collection()
    col.upsert(ids=ids, documents=texts, metadatas=metadatas)


def query(text: str, top_k: int) -> list[dict[str, Any]]:
    col = get_collection()
    res = col.query(query_texts=[text], n_results=top_k)
    out: list[dict[str, Any]] = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for i, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists)):
        # cosine distance -> similarity score in [0,1]
        score = max(0.0, 1.0 - float(dist))
        out.append(
            {
                "id": cid,
                "chunk": doc,
                "metadata": meta or {},
                "score": score,
                "rank": i,
            }
        )
    return out


def collection_count() -> int:
    return get_collection().count()


def reset_collection() -> None:
    client = _client()
    try:
        client.delete_collection(settings.collection_name)
    except Exception:
        pass
    # Recreate via get_or_create
    get_collection()
