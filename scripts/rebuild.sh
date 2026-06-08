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
# Requires the Dockerized pgvector container to be up:
#   docker compose up -d postgres
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

# ── A stray PG_DSN exported in the shell points at the wrong (native 5432) DB
#    and shadows .env. Unset it here so .env's Dockerized 5433 value wins.
#    This only affects this script's process, not your shell. ────────────────────
unset PG_DSN || true

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
