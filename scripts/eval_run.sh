#!/usr/bin/env bash
# eval_run.sh — run the DeepEval suite against the CURRENTLY CHECKED-OUT code.
#
# Handles the backend lifecycle so the footguns in §8 of the RUNBOOK can't bite:
# kills stale listeners on the port, sets EVAL_MODE + a lifted rate limit, waits
# for the models to finish loading, runs the suite, then shuts the backend down.
#
# Deliberately does NOT touch git — checking out branches under a dirty tree is
# how you lose work. Run it once per ref yourself:
#
#     git checkout main              && ./scripts/eval_run.sh
#     git checkout robustness/hardening && ./scripts/eval_run.sh
#     python scripts/deepeval_eval.py --compare <first-archive> <second-archive>
#
# It prints the archive path at the end, and the exact compare command to run.
#
# Usage: ./scripts/eval_run.sh [extra args passed to deepeval_eval.py]
#        ./scripts/eval_run.sh --fast
#        ./scripts/eval_run.sh --tag multiturn

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8000}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-180}"   # cold start loads two ML models (~60-90s)
LOG="$REPO/eval_reports/backend-eval.log"

cd "$REPO"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "warning: OPENAI_API_KEY is not set — the judge will be unavailable." >&2
  echo "         Pass --no-llm for deterministic checks only, or export the key." >&2
fi

REF="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "=== Evaluating $REF ($SHA) ==="

if ! git diff --quiet 2>/dev/null; then
  echo "note: working tree has uncommitted changes — you are evaluating those, not $REF." >&2
fi

# ── Clear the port ───────────────────────────────────────────────────────────
# A zombie uvicorn keeps serving STALE CODE while the new one dies with
# "address already in use" — you would be evaluating the old build and never
# know it. This is the single most expensive mistake in this workflow.
STALE="$(lsof -ti :"$PORT" 2>/dev/null || true)"
if [ -n "$STALE" ]; then
  echo "Killing stale listener(s) on :$PORT — $STALE"
  # shellcheck disable=SC2086
  kill -9 $STALE 2>/dev/null || true
  sleep 1
fi

mkdir -p "$REPO/eval_reports"

# ── Start the backend ────────────────────────────────────────────────────────
# EVAL_MODE=1 makes /chat/sync echo retrieval context (needed for Faithfulness).
# The default 10/minute rate limit applies locally too and would fail most cases.
echo "Starting backend (log: $LOG)…"
(
  cd "$REPO/backend" && \
  CHAT_RATE_LIMIT=1000/minute EVAL_MODE=1 python main.py
) > "$LOG" 2>&1 &
BACKEND_PID=$!

cleanup() {
  if kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "Stopping backend (pid $BACKEND_PID)…"
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# ── Wait for readiness ───────────────────────────────────────────────────────
echo -n "Waiting for models to load"
READY=0
for _ in $(seq 1 "$BOOT_TIMEOUT"); do
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo
    echo "Backend exited during startup. Last 30 lines of $LOG:" >&2
    tail -30 "$LOG" >&2
    exit 1
  fi
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    READY=1; echo " ready."; break
  fi
  echo -n "."
  sleep 1
done

if [ "$READY" -ne 1 ]; then
  echo
  echo "Backend did not become healthy within ${BOOT_TIMEOUT}s. Last 30 lines:" >&2
  tail -30 "$LOG" >&2
  exit 1
fi

# Confirm it can actually reach the vector store before spending judge tokens on
# 60 cases that would all return the no-info fallback.
if curl -sf "http://127.0.0.1:$PORT/health/deep" >/dev/null 2>&1; then
  echo "Vector store reachable."
else
  echo "WARNING: /health/deep is not OK — the database is unreachable or empty." >&2
  echo "         Every case would fail on retrieval, not on answer quality." >&2
  echo "         Response: $(curl -s "http://127.0.0.1:$PORT/health/deep" | head -c 300)" >&2
  read -r -p "Continue anyway? [y/N] " reply
  case "$reply" in [yY]*) ;; *) echo "Aborted."; exit 1 ;; esac
fi

# ── Run the suite ────────────────────────────────────────────────────────────
echo
python scripts/deepeval_eval.py --delay 0 "$@"
SUITE_STATUS=$?

# ── Point at the archive this run produced ───────────────────────────────────
ARCHIVE="$(ls -t "$REPO"/eval_reports/history/*.json 2>/dev/null | head -1)"
echo
echo "=== Done: $REF ($SHA), suite exit $SUITE_STATUS ==="
if [ -n "$ARCHIVE" ]; then
  echo "Archived run: $ARCHIVE"
  echo
  echo "After evaluating the other ref, compare with:"
  echo "  python scripts/deepeval_eval.py --compare \\"
  echo "      $ARCHIVE \\"
  echo "      <the-other-archive>"
  echo "(list them with: ls -t eval_reports/history/*.json | head -5)"
fi

exit "$SUITE_STATUS"
