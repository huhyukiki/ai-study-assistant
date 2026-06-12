import pytest

from src.document import chunk_pages


def test_chunk_pages_preserves_page_numbers_and_overlap() -> None:
    chunks = chunk_pages(["abcdefghij", "klmnop"], chunk_size=6, overlap=2)

    assert [(chunk.text, chunk.page) for chunk in chunks] == [
        ("abcdef", 1),
        ("efghij", 1),
        ("klmnop", 2),
    ]
    assert [chunk.chunk_id for chunk in chunks] == [0, 1, 2]


def test_chunk_pages_ignores_blank_pages() -> None:
    assert chunk_pages(["", "   "]) == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_chunk_pages_rejects_invalid_settings(
    chunk_size: int,
    overlap: int,
) -> None:
    with pytest.raises(ValueError):
        chunk_pages(["text"], chunk_size=chunk_size, overlap=overlap)

