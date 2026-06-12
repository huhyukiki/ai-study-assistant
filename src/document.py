from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import BinaryIO

from pypdf import PdfReader


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    page: int
    chunk_id: int


def extract_pdf_pages(file: bytes | BinaryIO) -> list[str]:
    """Extract text from a PDF while preserving page boundaries."""
    source = BytesIO(file) if isinstance(file, bytes) else file
    reader = PdfReader(source)
    return [(page.extract_text() or "").strip() for page in reader.pages]


def chunk_pages(
    pages: list[str],
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[DocumentChunk]:
    """Split page text into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    chunks: list[DocumentChunk] = []
    chunk_id = 0
    step = chunk_size - overlap

    for page_number, raw_text in enumerate(pages, start=1):
        text = " ".join(raw_text.split())
        for start in range(0, len(text), step):
            content = text[start : start + chunk_size].strip()
            if content:
                chunks.append(
                    DocumentChunk(
                        text=content,
                        page=page_number,
                        chunk_id=chunk_id,
                    )
                )
                chunk_id += 1
            if start + chunk_size >= len(text):
                break

    return chunks

