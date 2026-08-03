"""
Tests for the Postgres circuit breaker (backend/retriever.py).

Background — the 2026-07-26 Supabase pause: with the database unreachable,
psycopg2 spends connect_timeout (10s) on every attempt, and
ThreadedConnectionPool opens POOL_MIN connections eagerly, so a failed pool
build costs that again. Every request therefore held a worker thread for ~10s
to produce an error it could have produced instantly, and a handful of
concurrent users was enough to stall the service. The breaker trips after a few
consecutive failures and fails in microseconds until a cooldown elapses.

The properties that matter most here are about RECOVERY: a breaker that never
closes is worse than no breaker, because it turns a transient outage into a
permanent one that needs a restart.
"""
import os
import sys

import pytest

# tests/ is a package (it has __init__.py), so a bare sibling import doesn't
# resolve — put this directory on the path first.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse the module stubs from the pool tests so retriever imports without
# psycopg2/torch installed.
from test_db_pool_resilience import FakeConn, FakePool, retriever  # noqa: E402,F401

psycopg2 = sys.modules["psycopg2"]


@pytest.fixture(autouse=True)
def _reset_breaker():
    retriever._breaker_record_success()
    retriever._registered_conn_ids.clear()
    retriever._conn_last_used.clear()
    yield
    retriever._breaker_record_success()
    retriever._registered_conn_ids.clear()
    retriever._conn_last_used.clear()


@pytest.fixture
def breaker(monkeypatch):
    monkeypatch.setattr(retriever, "BREAKER_THRESHOLD", 3)
    monkeypatch.setattr(retriever, "BREAKER_COOLDOWN", 30.0)
    return retriever


# ── Tripping ─────────────────────────────────────────────────────────────────

def test_closed_breaker_allows_calls(breaker):
    breaker._breaker_check()   # must not raise


def test_stays_closed_below_threshold(breaker):
    breaker._breaker_record_failure()
    breaker._breaker_record_failure()
    breaker._breaker_check()   # 2 < 3, still closed
    assert breaker._breaker_state()["open"] is False


def test_trips_at_threshold_and_fails_fast(breaker):
    for _ in range(3):
        breaker._breaker_record_failure()
    with pytest.raises(retriever.DatabaseUnavailable):
        breaker._breaker_check()
    assert breaker._breaker_state()["open"] is True


def test_failing_fast_is_actually_fast(breaker):
    """The whole point: no connect timeout is paid while the breaker is open."""
    import time
    for _ in range(3):
        breaker._breaker_record_failure()
    t0 = time.perf_counter()
    for _ in range(1000):
        try:
            breaker._breaker_check()
        except retriever.DatabaseUnavailable:
            pass
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"1000 open-breaker checks took {elapsed:.3f}s"


# ── Recovery (the properties that matter) ────────────────────────────────────

def test_half_opens_after_cooldown(breaker, monkeypatch):
    """A breaker that never closes turns a transient outage into a permanent
    one requiring a restart."""
    for _ in range(3):
        breaker._breaker_record_failure()
    with pytest.raises(retriever.DatabaseUnavailable):
        breaker._breaker_check()

    # Pretend the cooldown elapsed.
    monkeypatch.setattr(retriever, "_breaker_opened_at",
                        retriever._breaker_opened_at - 31.0)
    breaker._breaker_check()   # half-open probe is allowed through


def test_success_closes_the_breaker(breaker, monkeypatch):
    for _ in range(3):
        breaker._breaker_record_failure()
    monkeypatch.setattr(retriever, "_breaker_opened_at",
                        retriever._breaker_opened_at - 31.0)
    breaker._breaker_check()
    breaker._breaker_record_success()

    assert breaker._breaker_state()["open"] is False
    assert breaker._breaker_state()["consecutive_failures"] == 0
    breaker._breaker_check()   # fully closed again


def test_failed_probe_re_arms_the_cooldown(breaker, monkeypatch):
    """If the half-open probe fails, the next caller must NOT be let straight
    through — otherwise every request becomes a probe and we are back to paying
    a connect timeout each time, which is the bug the breaker exists to fix."""
    for _ in range(3):
        breaker._breaker_record_failure()
    monkeypatch.setattr(retriever, "_breaker_opened_at",
                        retriever._breaker_opened_at - 31.0)
    breaker._breaker_check()          # probe allowed
    breaker._breaker_record_failure()  # probe failed

    with pytest.raises(retriever.DatabaseUnavailable):
        breaker._breaker_check()
    assert breaker._breaker_state()["cooldown_remaining_s"] > 25


def test_breaker_can_be_disabled(breaker, monkeypatch):
    monkeypatch.setattr(retriever, "BREAKER_THRESHOLD", 0)
    for _ in range(50):
        breaker._breaker_record_failure()
    breaker._breaker_check()   # disabled → never raises


# ── Integration with _conn() ─────────────────────────────────────────────────

def test_conn_records_failure_on_operational_error(breaker, monkeypatch):
    conn = FakeConn(alive=True)
    pool = FakePool([conn])
    monkeypatch.setattr(retriever, "_get_pool", lambda: pool)

    with pytest.raises(psycopg2.OperationalError):
        with retriever._conn():
            raise psycopg2.OperationalError("server closed the connection")

    assert breaker._breaker_state()["consecutive_failures"] == 1


def test_conn_resets_breaker_on_success(breaker, monkeypatch):
    breaker._breaker_record_failure()
    breaker._breaker_record_failure()
    conn = FakeConn(alive=True)
    pool = FakePool([conn])
    monkeypatch.setattr(retriever, "_get_pool", lambda: pool)

    with retriever._conn() as c:
        assert c is conn

    assert breaker._breaker_state()["consecutive_failures"] == 0


def test_conn_fails_fast_once_breaker_is_open(breaker, monkeypatch):
    """The key integration property: an open breaker must not even reach the
    pool, so no connection attempt (and no 10s timeout) happens."""
    calls = {"n": 0}

    def counting_pool():
        calls["n"] += 1
        return FakePool([FakeConn(alive=True)])

    monkeypatch.setattr(retriever, "_get_pool", counting_pool)
    for _ in range(3):
        breaker._breaker_record_failure()

    with pytest.raises(retriever.DatabaseUnavailable):
        with retriever._conn():
            pass
    assert calls["n"] == 0, "open breaker must not attempt a connection"


def test_pool_build_failure_trips_the_breaker(breaker, monkeypatch):
    """The Supabase-pause shape exactly: ThreadedConnectionPool construction
    raises OperationalError because the tenant does not exist."""
    def boom():
        raise psycopg2.OperationalError(
            'FATAL:  (ENOTFOUND) tenant/user postgres.xxx not found')

    monkeypatch.setattr(retriever, "_get_pool", boom)

    for i in range(3):
        with pytest.raises(psycopg2.OperationalError):
            with retriever._conn():
                pass
        assert breaker._breaker_state()["consecutive_failures"] == i + 1

    # Fourth caller is short-circuited instead of paying another timeout.
    with pytest.raises(retriever.DatabaseUnavailable):
        with retriever._conn():
            pass


# ── Health reporting ─────────────────────────────────────────────────────────

def test_db_healthy_reports_breaker_message(breaker, monkeypatch):
    monkeypatch.setattr(retriever, "_get_pool",
                        lambda: (_ for _ in ()).throw(
                            psycopg2.OperationalError("down")))
    for _ in range(3):
        breaker._breaker_record_failure()

    ok, detail = retriever.db_healthy()
    assert ok is False
    assert "breaker open" in detail


def test_db_healthy_force_bypasses_the_breaker(breaker, monkeypatch):
    """An operator asking 'is it back yet?' must get a real answer, not the
    breaker's cached pessimism."""
    conn = FakeConn(alive=True)

    class CountingCur:
        def __enter__(self): return self
        def __exit__(self, *e): return False
        def execute(self, sql, *a): pass
        def fetchone(self): return (1581,)

    conn.cursor = lambda: CountingCur()
    monkeypatch.setattr(retriever, "_get_pool", lambda: FakePool([conn]))
    for _ in range(3):
        breaker._breaker_record_failure()

    assert retriever.db_healthy()[0] is False          # breaker's view
    ok, detail = retriever.db_healthy(bypass_breaker=True)
    assert ok is True and "1581" in detail


def test_breaker_state_snapshot_shape(breaker):
    s = breaker._breaker_state()
    assert set(s) == {"open", "consecutive_failures", "cooldown_remaining_s"}
