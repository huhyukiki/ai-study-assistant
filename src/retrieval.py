from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np
from numpy.typing import NDArray

from src.document import DocumentChunk


class EmbeddingsClient(Protocol):
    class Embeddings(Protocol):
        def create(self, *, model: str, input: list[str]): ...

    embeddings: Embeddings


@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    score: float


def create_embeddings(
    client: EmbeddingsClient,
    texts: Iterable[str],
    model: str,
    batch_size: int = 64,
) -> NDArray[np.float64]:
    """Create embeddings in batches and preserve input order."""
    items = list(texts)
    vectors: list[list[float]] = []

    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in response.data)

    if not vectors:
        return np.empty((0, 0), dtype=np.float64)
    return np.asarray(vectors, dtype=np.float64)


class VectorIndex:
    def __init__(
        self,
        chunks: list[DocumentChunk],
        embeddings: NDArray[np.float64],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            raise ValueError("at least one chunk is required")

        self.chunks = chunks
        self.embeddings = _normalize_rows(embeddings)

    def search(
        self,
        query_embedding: NDArray[np.float64],
        top_k: int = 4,
    ) -> list[SearchResult]:
        query = np.asarray(query_embedding, dtype=np.float64).reshape(1, -1)
        normalized_query = _normalize_rows(query)[0]
        scores = self.embeddings @ normalized_query
        count = min(max(top_k, 1), len(self.chunks))
        indices = np.argsort(scores)[::-1][:count]
        return [
            SearchResult(chunk=self.chunks[index], score=float(scores[index]))
            for index in indices
        ]


def _normalize_rows(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)

