"""
Regression tests for connection-pool resilience (backend/retriever.py).

Background: the pool handed back whatever connection it held, with no liveness
check and no close-on-error. Two consequences in production:

  1. Managed Postgres front-ends (the Supabase pooler, PgBouncer, cloud idle
     timeouts) close server-side connections after a few minutes idle. psycopg2
     cannot detect this locally — `conn.closed` still reads 0 — so after a quiet
     period every request drew a dead connection and failed until the process
     was restarted.
  2. A connection that errored mid-query was returned to the pool in an unknown
     state, spreading one failure across every later request that drew it.

retriever.py imports heavy ML/DB packages that aren't installed in CI, so the
module is loaded against lightweight stubs — these tests exercise the pool
bookkeeping only, which is pure Python.
"""
import os
import sys
import types

import pytest


# ── Stub the heavy imports so retriever can be loaded in CI ──────────────────

def _install_stubs():
    if "psycopg2" in sys.modules and hasattr(sys.modules["psycopg2"], "_is_stub"):
        return

    psycopg2 = types.ModuleType("psycopg2")
    psycopg2._is_stub = True

    class OperationalError(Exception):
        pass

    class InterfaceError(Exception):
        pass

    class PoolError(Exception):
        pass

    psycopg2.OperationalError = OperationalError
    psycopg2.InterfaceError = InterfaceError

    pool_mod = types.ModuleType("psycopg2.pool")
    pool_mod.PoolError = PoolError

    class ThreadedConnectionPool:  # pragma: no cover - not exercised here
        def __init__(self, *a, **kw):
            pass

    pool_mod.ThreadedConnectionPool = ThreadedConnectionPool
    psycopg2.pool = pool_mod

    extras = types.ModuleType("psycopg2.extras")
    extras.execute_values = lambda *a, **kw: None
    psycopg2.extras = extras

    sys.modules["psycopg2"] = psycopg2
    sys.modules["psycopg2.pool"] = pool_mod
    sys.modules["psycopg2.extras"] = extras

    pgvector = types.ModuleType("pgvector")
    pgv_psycopg2 = types.ModuleType("pgvector.psycopg2")
    pgv_psycopg2.register_vector = lambda conn: None
    pgvector.psycopg2 = pgv_psycopg2
    sys.modules["pgvector"] = pgvector
    sys.modules["pgvector.psycopg2"] = pgv_psycopg2

    rank_bm25 = types.ModuleType("rank_bm25")
    rank_bm25.BM25Okapi = object
    sys.modules["rank_bm25"] = rank_bm25

    st = types.ModuleType("sentence_transformers")
    st.CrossEncoder = object
    st.SentenceTransformer = object
    sys.modules["sentence_transformers"] = st


_install_stubs()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import retriever  # noqa: E402


# ── Test doubles ─────────────────────────────────────────────────────────────

class FakeConn:
    """Minimal psycopg2-connection stand-in. `alive=False` simulates a socket
    the upstream pooler closed without telling us: `closed` still reads 0, and
    the failure only surfaces when a query is attempted."""

    def __init__(self, alive=True, name="c"):
        self.alive = alive
        self.closed = 0
        self.name = name
        self.rollbacks = 0

    def cursor(self):
        conn = self

        class _Cur:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

            def execute(self_inner, sql, *a):
                if not conn.alive:
                    raise sys.modules["psycopg2"].OperationalError("server closed the connection")

            def fetchone(self_inner):
                return (1,)

        return _Cur()

    def rollback(self):
        self.rollbacks += 1
        if not self.alive:
            raise sys.modules["psycopg2"].OperationalError("connection already closed")


class FakePool:
    def __init__(self, conns):
        self._queue = list(conns)
        self.returned = []
        self.closed = []

    def getconn(self):
        if not self._queue:
            raise sys.modules["psycopg2"].pool.PoolError("connection pool exhausted")
        return self._queue.pop(0)

    def putconn(self, conn, close=False):
        if close:
            self.closed.append(conn)
        else:
            self.returned.append(conn)
            self._queue.append(conn)


@pytest.fixture(autouse=True)
def _clean_pool_state():
    retriever._registered_conn_ids.clear()
    retriever._conn_last_used.clear()
    yield
    retriever._registered_conn_ids.clear()
    retriever._conn_last_used.clear()


# ── Liveness detection ───────────────────────────────────────────────────────

def test_is_usable_detects_silently_dropped_connection():
    dead = FakeConn(alive=False)
    assert dead.closed == 0, "precondition: psycopg2 does not know it's dead"
    assert retriever._is_usable(dead) is False


def test_is_usable_accepts_live_connection():
    assert retriever._is_usable(FakeConn(alive=True)) is True


def test_is_usable_short_circuits_on_locally_closed_connection():
    conn = FakeConn(alive=True)
    conn.closed = 1
    assert retriever._is_usable(conn) is False


# ── Checkout path ────────────────────────────────────────────────────────────

def test_checkout_discards_stale_connections_and_returns_a_live_one(monkeypatch):
    """The scenario that used to wedge the service: connections aged out
    overnight, so the first morning request drew a corpse."""
    dead1, dead2 = FakeConn(alive=False, name="dead1"), FakeConn(alive=False, name="dead2")
    live = FakeConn(alive=True, name="live")
    pool = FakePool([dead1, dead2, live])

    # No last-used timestamps recorded → treated as idle beyond the threshold,
    # so every candidate gets validated.
    got = retriever._checkout(pool)

    assert got is live
    assert pool.closed == [dead1, dead2], "dead connections must be closed, not recycled"


def test_checkout_skips_validation_for_recently_used_connection(monkeypatch):
    """Hot-path connections must not pay an extra SELECT 1 round trip."""
    import time
    conn = FakeConn(alive=True)
    retriever._conn_last_used[id(conn)] = time.monotonic()
    pool = FakePool([conn])

    calls = {"n": 0}
    monkeypatch.setattr(retriever, "_is_usable",
                        lambda c: calls.__setitem__("n", calls["n"] + 1) or True)

    assert retriever._checkout(pool) is conn
    assert calls["n"] == 0, "recently-used connection should not be validated"


def test_checkout_registers_pgvector_once_per_connection():
    conn = FakeConn(alive=True)
    pool = FakePool([conn])
    retriever._checkout(pool)
    assert id(conn) in retriever._registered_conn_ids


# ── Error handling in _conn() ────────────────────────────────────────────────

def test_conn_closes_connection_after_operational_error():
    """A connection that died mid-query must not go back into rotation."""
    conn = FakeConn(alive=True)
    pool = FakePool([conn])

    with pytest.raises(sys.modules["psycopg2"].OperationalError):
        with retriever._conn_from_pool(pool):
            raise sys.modules["psycopg2"].OperationalError("server closed the connection")

    assert pool.closed == [conn]
    assert pool.returned == []


def test_conn_returns_healthy_connection_to_pool():
    conn = FakeConn(alive=True)
    pool = FakePool([conn])

    with retriever._conn_from_pool(pool) as c:
        assert c is conn

    assert pool.returned == [conn]
    assert pool.closed == []
    assert id(conn) in retriever._conn_last_used, "successful use should refresh idle clock"


def test_conn_keeps_query_error_connection_when_still_usable():
    """A query-level error (bad SQL) leaves the socket fine — rollback and reuse
    it rather than needlessly churning the pool."""
    conn = FakeConn(alive=True)
    pool = FakePool([conn])

    with pytest.raises(ValueError):
        with retriever._conn_from_pool(pool):
            raise ValueError("bad SQL")

    assert pool.returned == [conn]
    assert pool.closed == []


# ── Pool exhaustion ──────────────────────────────────────────────────────────

def test_acquire_waits_before_failing_on_exhausted_pool(monkeypatch):
    monkeypatch.setattr(retriever, "POOL_ACQUIRE_TIMEOUT", 0.05)
    pool = FakePool([])
    with pytest.raises(sys.modules["psycopg2"].pool.PoolError):
        retriever._acquire(pool)


def test_acquire_succeeds_once_a_connection_is_freed(monkeypatch):
    monkeypatch.setattr(retriever, "POOL_ACQUIRE_TIMEOUT", 1.0)
    conn = FakeConn(alive=True)
    pool = FakePool([])

    calls = {"n": 0}
    original = pool.getconn

    def flaky_getconn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise sys.modules["psycopg2"].pool.PoolError("exhausted")
        return conn

    pool.getconn = flaky_getconn
    assert retriever._acquire(pool) is conn
    assert calls["n"] == 3
