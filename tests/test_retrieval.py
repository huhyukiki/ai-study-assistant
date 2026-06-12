import numpy as np
import pytest

from src.document import DocumentChunk
from src.retrieval import VectorIndex


def test_vector_index_returns_most_similar_chunk_first() -> None:
    chunks = [
        DocumentChunk("alpha", page=1, chunk_id=0),
        DocumentChunk("beta", page=2, chunk_id=1),
        DocumentChunk("gamma", page=3, chunk_id=2),
    ]
    index = VectorIndex(
        chunks,
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.2],
            ]
        ),
    )

    results = index.search(np.array([1.0, 0.0]), top_k=2)

    assert [result.chunk.text for result in results] == ["alpha", "gamma"]
    assert results[0].score == pytest.approx(1.0)


def test_vector_index_validates_input_lengths() -> None:
    with pytest.raises(ValueError):
        VectorIndex(
            [DocumentChunk("alpha", page=1, chunk_id=0)],
            np.empty((0, 2)),
        )

