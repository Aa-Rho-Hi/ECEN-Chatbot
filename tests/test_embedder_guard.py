"""
Tests for the embedder-identity guard (crawler/ingest.py, backend/retriever.py).

Background: the database lives on Supabase and is shared between the deployed
service and local development. Locally, .env names the fine-tuned embedder while
PG_DSN points at that production instance, and rebuild.sh's comments still
describe a local Docker Postgres that PG_DSN no longer resolves to. One
`./scripts/rebuild.sh` would crawl, embed with the fine-tuned model, and upsert
into the live index.

That failure is silent rather than loud: both models are 384-dimensional, so
pgvector accepts the write and the HNSW index stores two mutually meaningless
coordinate systems. Retrieval then returns near-random chunks and the LLM
correctly reports it cannot find the answer — which is exactly the
"I don't have specific details" symptom, with no error anywhere.

So the database records which embedder wrote it, an ingest that would change
that identity refuses to run, and the backend reports a mismatch at startup and
on /health/deep.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_db_pool_resilience import FakeConn, FakePool, retriever  # noqa: E402,F401

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


# ── Identity normalization ───────────────────────────────────────────────────

@pytest.mark.parametrize("ref,expected", [
    ("all-MiniLM-L6-v2", "all-MiniLM-L6-v2"),
    # The same model referred to by its Hub path must not look like a change.
    ("sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2"),
    # An absolute local path and a relative one are the same model.
    ("/Users/x/proj/finetune/tamu-ece-embedder", "tamu-ece-embedder"),
    ("finetune/tamu-ece-embedder", "tamu-ece-embedder"),
    ("finetune/tamu-ece-embedder/", "tamu-ece-embedder"),
])
def test_identity_normalization(ref, expected):
    assert retriever.embedder_identity(ref) == expected


def test_the_two_real_models_are_distinguishable():
    """The whole guard rests on these not colliding."""
    prod = retriever.embedder_identity("all-MiniLM-L6-v2")
    local = retriever.embedder_identity(
        "/Users/roheeeee/Documents/Claude/Projects/chatbot/finetune/tamu-ece-embedder")
    assert prod != local


def test_ingest_and_retriever_normalize_identically():
    """Two implementations of the same rule will drift unless pinned. If they
    disagree, the writer records one name and the reader compares another, and
    the guard silently stops guarding."""
    import ast
    src = open(os.path.join(REPO, "crawler", "ingest.py")).read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "embedder_identity")
    ns = {"os": os, "EMBED_MODEL": "all-MiniLM-L6-v2"}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<ingest>", "exec"), ns)
    ingest_identity = ns["embedder_identity"]

    for ref in ["all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2",
                "/Users/x/finetune/tamu-ece-embedder", "finetune/tamu-ece-embedder/"]:
        assert ingest_identity(ref) == retriever.embedder_identity(ref), ref


# ── Read side: does the backend notice? ──────────────────────────────────────

def _conn_returning(meta_table_exists, stored_value, monkeypatch):
    conn = FakeConn(alive=True)

    class Cur:
        def __enter__(self): return self
        def __exit__(self, *e): return False
        def execute(self, sql, params=None):
            self._sql = sql
        def fetchone(self):
            if "to_regclass" in self._sql:
                return ("ecen_meta" if meta_table_exists else None,)
            if "ecen_meta" in self._sql:
                return (stored_value,) if stored_value else None
            return (0,)

    conn.cursor = lambda: Cur()
    monkeypatch.setattr(retriever, "_get_pool", lambda: FakePool([conn]))
    retriever._breaker_record_success()


def test_matching_embedder_reports_ok(monkeypatch):
    monkeypatch.setattr(retriever, "EMBED_MODEL", "all-MiniLM-L6-v2")
    _conn_returning(True, "all-MiniLM-L6-v2", monkeypatch)
    ok, detail = retriever.embedder_matches_index()
    assert ok is True
    assert "all-MiniLM-L6-v2" in detail


def test_mismatched_embedder_is_reported(monkeypatch):
    """The exact local configuration: fine-tuned queries against a base-model
    index."""
    monkeypatch.setattr(retriever, "EMBED_MODEL",
                        "/Users/roheeeee/proj/finetune/tamu-ece-embedder")
    _conn_returning(True, "all-MiniLM-L6-v2", monkeypatch)
    ok, detail = retriever.embedder_matches_index()
    assert ok is False
    assert "MISMATCH" in detail
    assert "all-MiniLM-L6-v2" in detail and "tamu-ece-embedder" in detail


def test_hub_path_is_not_a_false_alarm(monkeypatch):
    """A cosmetic spelling change must not page anyone."""
    monkeypatch.setattr(retriever, "EMBED_MODEL",
                        "sentence-transformers/all-MiniLM-L6-v2")
    _conn_returning(True, "all-MiniLM-L6-v2", monkeypatch)
    assert retriever.embedder_matches_index()[0] is True


def test_index_predating_the_guard_is_not_flagged(monkeypatch):
    """Existing deployments have no ecen_meta table yet; that is not an error."""
    monkeypatch.setattr(retriever, "EMBED_MODEL", "all-MiniLM-L6-v2")
    _conn_returning(False, None, monkeypatch)
    ok, detail = retriever.embedder_matches_index()
    assert ok is True
    assert "predates" in detail


def test_unreachable_database_does_not_report_a_mismatch(monkeypatch):
    """During an outage we know nothing about the embedder — claiming a
    mismatch would send you chasing the wrong bug."""
    import psycopg2
    monkeypatch.setattr(retriever, "_get_pool",
                        lambda: (_ for _ in ()).throw(psycopg2.OperationalError("down")))
    ok, _ = retriever.embedder_matches_index()
    assert ok is True


# ── Write side: does ingest refuse? ──────────────────────────────────────────

def _ingest_source():
    return open(os.path.join(REPO, "crawler", "ingest.py")).read()


def test_guard_runs_before_the_crawl_and_before_upsert():
    """Ordering is the point: an hour of crawling shouldn't precede a check
    that was always going to fail, and nothing may touch the index first."""
    src = _ingest_source()
    body = src[src.index("def ingest("):]
    guard = body.index("check_embedder_compatible")
    crawl = body.index("crawl()")
    upsert = body.index("INSERT INTO ecen_docs")
    assert guard < crawl < upsert


def test_guard_aborts_rather_than_continuing():
    src = _ingest_source()
    body = src[src.index("def ingest("):]
    seg = body[body.index("check_embedder_compatible"):][:220]
    assert "SystemExit" in seg or "return" in seg


def test_mismatch_message_names_both_models_and_the_target_db():
    """An error you can act on without opening the source."""
    src = _ingest_source()
    seg = src[src.index("EMBEDDER MISMATCH"):][:1200]
    assert "index was built with" in seg
    assert "this run would use" in seg
    assert "target database" in seg
    assert "FORCE_EMBEDDER_CHANGE" in seg


def test_forced_change_warns_that_diff_mode_is_unsafe():
    """Forcing with --diff re-embeds only changed rows, interleaving two vector
    spaces — strictly worse than a clean swap and much harder to spot."""
    src = _ingest_source()
    seg = src[src.index("EMBEDDER CHANGE FORCED"):][:400]
    assert "--diff" in seg or "FULL ingest" in seg


def test_identity_is_recorded_after_a_successful_ingest():
    src = _ingest_source()
    assert "record_embedder(conn)" in src
    body = src[src.index("def ingest("):]
    assert body.index("record_embedder(conn)") > body.index("INSERT INTO ecen_docs")


def test_empty_or_untracked_index_adopts_the_current_model():
    """First-ever ingest, and existing deployments, must not be blocked."""
    src = _ingest_source()
    seg = src[src.index("def check_embedder_compatible"):][:1400]
    assert "if stored is None" in seg
    assert "return True" in seg
