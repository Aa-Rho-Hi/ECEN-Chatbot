#!/usr/bin/env bash
#
# rebuild.sh — one command to rebuild the whole TAMU ECE chatbot index.
#
#   crawl + chunk + embed + upsert   (crawler/ingest.py)
#   → knowledge graph                (backend/graph_builder.py)
#
# Usage:
#   ./scripts/rebuild.sh              # full re-crawl + re-embed everything
#   ./scripts/rebuild.sh --diff       # only re-embed pages whose content changed
#   ./scripts/rebuild.sh --skip-check # skip the DB health pre-flight
#
# ⚠ THIS WRITES TO WHATEVER PG_DSN IN .env POINTS AT.
#   That is currently the SHARED SUPABASE INSTANCE the deployed site serves from
#   — not a local database. There is no separate staging copy. A bad run here is
#   a production incident.
#
#   These instructions used to say "requires the Dockerized pgvector container",
#   and the `unset PG_DSN` below was written so .env's local port-5433 value
#   would win. .env now holds the Supabase DSN, so that unset does the opposite
#   of what its comment claims: it removes your override and sends the write to
#   production. Check before you run:
#
#     grep -E '^(PG_DSN|EMBEDDING_MODEL)=' .env
#
#   EMBEDDING_MODEL must match the model the index was built with, or every
#   vector written here lands in a different coordinate system from the rest.
#   crawler/ingest.py now refuses to run on a mismatch — do not force past it
#   without re-embedding the whole corpus.
#
set -euo pipefail

# ── Resolve project root (this script lives in <root>/scripts) ────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

# ── Parse args ────────────────────────────────────────────────────────────────
DIFF=""
SKIP_CHECK=0
for arg in "$@"; do
  case "$arg" in
    --diff)       DIFF="--diff" ;;
    --skip-check) SKIP_CHECK=1 ;;
    -h|--help)    sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

# ── Deliberately NOT unsetting PG_DSN any more. ──────────────────────────────
#    The old code did `unset PG_DSN` so that .env's local Docker value would win
#    over a stray shell export. Now that .env points at production Supabase,
#    that would discard the safer override and target the live database instead.
#    An explicitly exported PG_DSN is a considered choice; respect it.
#
#    Show what is about to be written to, so a production target is impossible
#    to miss.
banner_target() {
  # The .env path is passed explicitly: python-dotenv's find_dotenv() walks the
  # CALLER's stack frame, which does not exist for a script fed on stdin, so it
  # raises and a tolerant `except` would leave this printing "(unset)" — a
  # safety banner that silently omits the production warning is worse than none.
  "$PYTHON" - "$ROOT/.env" <<'PY'
import os
import sys

env_path = sys.argv[1]
loaded = False
try:
    from dotenv import dotenv_values
    values = dotenv_values(env_path)
    loaded = True
except Exception as e:
    print(f"  !! could not read {env_path}: {e}")
    print("  !! cannot confirm the write target — inspect .env yourself first.")
    values = {}

# An exported PG_DSN wins over .env for this script (we no longer unset it),
# so report the value that will actually be used.
dsn = os.environ.get("PG_DSN") or values.get("PG_DSN")
model = os.environ.get("EMBEDDING_MODEL") or values.get("EMBEDDING_MODEL") \
    or "all-MiniLM-L6-v2"

if not dsn:
    print("  target database : (unset — crawler/ingest.py default)")
elif loaded:
    host = dsn.split("@")[-1] if "@" in dsn else dsn
    print(f"  target database : {host}")
    if "supabase" in host or "run.app" in host:
        print("  ** THIS IS THE SHARED PRODUCTION DATABASE. **")
print(f"  embedding model : {os.path.basename(model.rstrip('/')) or model}")
PY
}

# ── Activate a local virtualenv if one exists (optional) ──────────────────────
for v in .venv venv env; do
  if [ -f "$ROOT/$v/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "$ROOT/$v/bin/activate"
    echo "• Activated virtualenv: $v"
    break
  fi
done

banner() { printf '\n\033[1;35m== %s ==\033[0m\n' "$1"; }

banner "Write target"
banner_target

# ── Stage 0: DB health pre-flight ─────────────────────────────────────────────
if [ "$SKIP_CHECK" -eq 0 ]; then
  banner "Checking database (scripts/check_db.py)"
  if ! "$PYTHON" scripts/check_db.py; then
    echo "
✗ Database check failed. Is the pgvector container running?
    docker compose up -d postgres
  Then re-run this script (or pass --skip-check to bypass)." >&2
    exit 1
  fi
fi

# ── Stage 1: crawl + ingest ───────────────────────────────────────────────────
banner "Crawling + ingesting ${DIFF:+(diff mode) }(crawler/ingest.py)"
"$PYTHON" crawler/ingest.py $DIFF

# ── Stage 2: rebuild the knowledge graph ──────────────────────────────────────
banner "Rebuilding knowledge graph (backend/graph_builder.py)"
"$PYTHON" backend/graph_builder.py

banner "Done"
echo "Index + graph rebuilt. Restart the backend to load the new graph.json."
