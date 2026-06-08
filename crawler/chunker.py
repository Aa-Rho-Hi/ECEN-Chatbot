"""
chunker.py — Splits PageDocs into semantically coherent chunks.

Strategy:
  - Faculty/people pages → one chunk per person block
  - News articles        → one chunk per article (already scoped pages)
  - Long pages           → RecursiveCharacterTextSplitter with overlap
  - Short pages (<800 chars) → kept as a single chunk
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crawler import PageDoc

CHUNK_SIZE = 600       # target tokens (approx chars / 4)
CHUNK_OVERLAP = 80


@dataclass
class Chunk:
    chunk_id: str          # "{url}::{index}"
    url: str
    title: str
    section: str
    text: str
    content_hash: str


def _split_by_size(text: str, size: int = CHUNK_SIZE * 4, overlap: int = CHUNK_OVERLAP * 4) -> list[str]:
    """Simple recursive splitter: tries paragraph → sentence → character boundaries."""
    if len(text) <= size:
        return [text]

    chunks = []
    # Try splitting on double newlines (paragraphs)
    paragraphs = re.split(r"\n{2,}", text)
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # If single paragraph is still too big, split on sentences
            if len(para) > size:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                buf = ""
                for sent in sentences:
                    if len(buf) + len(sent) + 1 <= size:
                        buf = (buf + " " + sent).strip()
                    else:
                        if buf:
                            chunks.append(buf)
                        buf = sent
                if buf:
                    current = buf
                else:
                    current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    # Add overlap: prepend last `overlap` chars of previous chunk
    overlapped = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            overlapped.append(chunk)
        else:
            prefix = chunks[i - 1][-overlap:].strip()
            overlapped.append((prefix + "\n" + chunk).strip())

    return overlapped


def chunk_docs(docs: list[PageDoc]) -> list[Chunk]:  # type: ignore[name-defined]
    all_chunks: list[Chunk] = []

    for doc in docs:
        text = doc.text.strip()
        if not text:
            continue

        # People pages: try to split per person entry
        if doc.section == "people":
            # Faculty profiles are often separated by headings or repeated patterns
            blocks = re.split(r"\n(?=[A-Z][a-z]+ [A-Z][a-z]+\n)", text)
            if len(blocks) > 1:
                for i, block in enumerate(blocks):
                    if len(block.strip()) > 50:
                        all_chunks.append(_make_chunk(doc, block.strip(), i))
                continue

        # Short pages → single chunk
        if len(text) <= CHUNK_SIZE * 4:
            all_chunks.append(_make_chunk(doc, text, 0))
            continue

        # Everything else → size-based split
        splits = _split_by_size(text)
        for i, split in enumerate(splits):
            all_chunks.append(_make_chunk(doc, split, i))

    return all_chunks


def _make_chunk(doc: PageDoc, text: str, index: int) -> Chunk:  # type: ignore[name-defined]
    import hashlib
    # Prefix title into the embedded text so the vector carries page context.
    # Strip noisy suffixes from TAMU page titles first.
    clean_title = (
        doc.title
        .replace(" | Texas A&M University Engineering", "")
        .replace(" - Engineering News", "")
        .strip()
    )
    embedded_text = f"{clean_title}\n\n{text}" if clean_title and index == 0 else text
    return Chunk(
        chunk_id=f"{doc.url}::{index}",
        url=doc.url,
        title=doc.title,
        section=doc.section,
        text=embedded_text,
        content_hash=hashlib.md5(embedded_text.encode()).hexdigest(),
    )
