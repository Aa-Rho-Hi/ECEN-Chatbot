"""
generate_training_data.py — Generates (query, passage) training pairs for
fine-tuning the embedding model WITHOUT any LLM API calls.

Strategy: self-supervised from the corpus itself.
  - query   = page title  (what a user might search for)
  - positive = chunk text  (the passage that answers it)

For chunks with rich content, also adds a "first-sentence query" variant.
This is standard practice for domain embedding fine-tuning (no negatives needed
— MultipleNegativesRankingLoss treats other in-batch examples as negatives).

Run:
    python generate_training_data.py
    python generate_training_data.py --limit 200   # quick test
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
import os

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

QDRANT_URL  = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION  = os.getenv("QDRANT_COLLECTION", "ecen_docs")
OUTPUT_FILE = "training_pairs.jsonl"

# Strip TAMU boilerplate suffixes from page titles
TITLE_STRIP = [
    " | Texas A&M University Engineering",
    " | Texas A&M University",
    " - Engineering News",
    " | TAMU ECE",
    " | Electrical & Computer Engineering",
]


def clean_title(raw: str) -> str:
    for suffix in TITLE_STRIP:
        raw = raw.replace(suffix, "")
    return raw.strip()


def first_sentence(text: str) -> str | None:
    """Extract the first meaningful sentence (>30 chars) from a chunk."""
    # Split on sentence-ending punctuation
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    for p in parts:
        p = p.strip()
        if len(p) >= 30 and not p.startswith(("©", "Skip", "Menu", "Toggle")):
            return p
    return None


def make_pairs(chunk: dict) -> list[dict]:
    text    = chunk.get("text", "").strip()
    title   = clean_title(chunk.get("title", ""))
    url     = chunk.get("url", "")
    section = chunk.get("section", "")
    cid     = chunk.get("chunk_id", "")

    if not text or len(text) < 100:
        return []

    # Skip nav/boilerplate
    low = text.lower()
    if any(low.startswith(p) for p in ("skip to", "©", "menu", "toggle", "contact us")):
        return []

    pairs = []

    # Pair 1: page title → chunk
    if title and len(title) > 5:
        pairs.append({
            "chunk_id": cid,
            "query":    title,
            "positive": text,
            "url":      url,
            "section":  section,
            "strategy": "title",
        })

    # Pair 2: section header → chunk (if section is informative)
    if section and len(section) > 5 and section.lower() not in ("home", "main", "content"):
        q = f"{title} — {section}" if title else section
        pairs.append({
            "chunk_id": f"{cid}::sec",
            "query":    q,
            "positive": text,
            "url":      url,
            "section":  section,
            "strategy": "section",
        })

    # Pair 3: first sentence as implicit query (works well for fact-dense chunks)
    sent = first_sentence(text)
    if sent and sent != text[:len(sent)] or (sent and len(text) > 300):
        pairs.append({
            "chunk_id": f"{cid}::sent",
            "query":    sent,
            "positive": text,
            "url":      url,
            "section":  section,
            "strategy": "first_sentence",
        })

    return pairs


def load_all_chunks(limit: int | None = None) -> list[dict]:
    client = QdrantClient(url=QDRANT_URL)
    chunks = []
    offset = None
    while True:
        result, offset = client.scroll(
            collection_name=COLLECTION,
            with_payload=True,
            with_vectors=False,
            limit=500,
            offset=offset,
        )
        for point in result:
            chunks.append(point.payload)
        if offset is None:
            break
        if limit and len(chunks) >= limit:
            break
    return chunks[:limit] if limit else chunks


def main(limit: int | None = None) -> None:
    chunks = load_all_chunks(limit)
    log.info("Loaded %d chunks from Qdrant", len(chunks))

    # Overwrite output file (no resume needed — this runs in seconds)
    output_path = Path(OUTPUT_FILE)
    written = 0

    with output_path.open("w") as out:
        for chunk in chunks:
            for pair in make_pairs(chunk):
                out.write(json.dumps(pair) + "\n")
                written += 1

    log.info("Done. Wrote %d training pairs to %s", written, OUTPUT_FILE)
    log.info(
        "Next: upload %s to Colab and run the fine-tuning notebook, "
        "or: sbatch finetune/hprc_job.sh",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N chunks (for testing)")
    args = parser.parse_args()
    main(limit=args.limit)
