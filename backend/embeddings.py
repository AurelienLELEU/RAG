"""Local embeddings via sentence-transformers, wrapped for Chroma."""
from __future__ import annotations

from functools import lru_cache
from typing import List

from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer

from .config import settings


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed(texts: List[str]) -> List[List[float]]:
    return _model().encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


class LocalEmbeddingFunction(EmbeddingFunction):
    """Chroma-compatible embedding function backed by sentence-transformers."""

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002 (chroma API)
        return embed(list(input))

    # Required by newer chromadb versions
    def name(self) -> str:  # type: ignore[override]
        return f"local-st::{settings.embedding_model}"
