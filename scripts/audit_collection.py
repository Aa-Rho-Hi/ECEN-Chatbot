"""
audit_collection.py — Audits the Qdrant collection for coverage issues.

Checks:
  1. Total chunk count and unique URL count
  2. Chunks per section
  3. Pages with only 1 chunk that are likely too short (potential crawl miss)
  4. Duplicate chunk_ids (should be 0 after the dedup fix)
  5. Pages with missing content_hash (old ingests before the fix)

Run: python3 scripts/audit_collection.py
"""

from collections import Counter, defaultdict
from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6333"
COLLECTION = "ecen_docs"

# Pages we know should have multiple chunks — flag if they only have 1
EXPECTED_MULTI_CHUNK = [
    "https://engineering.tamu.edu/electrical/academics/degrees/index.html",
    "https://engineering.tamu.edu/electrical/research/information-science-and-systems.html",
    "https://engineering.tamu.edu/electrical/research/research-areas.html",
    "https://engineering.tamu.edu/electrical/academics/degrees/graduate/index.html",
    "https://engineering.tamu.edu/electrical/academics/degrees/undergraduate/index.html",
]

client = QdrantClient(url=QDRANT_URL)

# ── Scroll entire collection ──────────────────────────────────────────────────
print("Scanning collection...")
all_points = []
offset = None
while True:
    results, offset = client.scroll(
        collection_name=COLLECTION,
        with_payload=True,
        with_vectors=False,
        limit=500,
        offset=offset,
    )
    all_points.extend(results)
    if offset is None:
        break

print(f"\n{'='*60}")
print(f"TOTAL POINTS:      {len(all_points)}")

# ── Chunk IDs ─────────────────────────────────────────────────────────────────
chunk_ids = [p.payload.get("chunk_id", "") for p in all_points]
unique_chunk_ids = set(chunk_ids)
duplicates = len(chunk_ids) - len(unique_chunk_ids)
print(f"UNIQUE CHUNK IDs:  {len(unique_chunk_ids)}")
print(f"DUPLICATES:        {duplicates}  {'✓' if duplicates == 0 else '✗ RE-RUN DEDUP SCRIPT'}")

# ── URLs ──────────────────────────────────────────────────────────────────────
url_chunks = defaultdict(list)
for p in all_points:
    url = p.payload.get("url", "unknown")
    url_chunks[url].append(p)

print(f"UNIQUE URLS:       {len(url_chunks)}")

# ── Sections ─────────────────────────────────────────────────────────────────
section_counts = Counter(p.payload.get("section", "unknown") for p in all_points)
print(f"\n{'─'*60}")
print("CHUNKS PER SECTION:")
for section, count in sorted(section_counts.items()):
    print(f"  {section:25s} {count:4d} chunks")

# ── Missing content_hash ──────────────────────────────────────────────────────
missing_hash = [p for p in all_points if not p.payload.get("content_hash")]
print(f"\n{'─'*60}")
print(f"MISSING content_hash: {len(missing_hash)}")
if missing_hash:
    print("  (These were ingested before the hash fix — re-ingest to fix)")
    for p in missing_hash[:5]:
        print(f"  {p.payload.get('chunk_id', '?')}")

# ── Single-chunk pages ────────────────────────────────────────────────────────
single_chunk_urls = {url: chunks for url, chunks in url_chunks.items() if len(chunks) == 1}
print(f"\n{'─'*60}")
print(f"SINGLE-CHUNK PAGES: {len(single_chunk_urls)} (may be fine for short pages)")

# ── Expected multi-chunk pages ────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("KEY PAGE COVERAGE CHECK:")
for url in EXPECTED_MULTI_CHUNK:
    chunks = url_chunks.get(url, [])
    total_chars = sum(len(c.payload.get("text", "")) for c in chunks)
    status = "✓" if len(chunks) >= 2 else ("⚠ ONLY 1 CHUNK" if chunks else "✗ MISSING")
    print(f"  {status}  {len(chunks)} chunks  {total_chars:5d} chars  {url.split('electrical/')[-1]}")

# ── Top pages by chunk count ──────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("TOP 15 PAGES BY CHUNK COUNT:")
top_pages = sorted(url_chunks.items(), key=lambda x: len(x[1]), reverse=True)[:15]
for url, chunks in top_pages:
    total_chars = sum(len(c.payload.get("text", "")) for c in chunks)
    print(f"  {len(chunks):2d} chunks  {total_chars:5d} chars  {url.split('electrical/')[-1]}")

# ── Zero-text chunks ─────────────────────────────────────────────────────────
empty = [p for p in all_points if len(p.payload.get("text", "")) < 50]
print(f"\n{'─'*60}")
print(f"NEAR-EMPTY CHUNKS (<50 chars): {len(empty)}")
for p in empty[:5]:
    print(f"  {p.payload.get('chunk_id')} → '{p.payload.get('text', '')[:60]}'")

print(f"\n{'='*60}")
print("Done.")
