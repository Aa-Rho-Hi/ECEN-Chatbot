"""
Regression tests for admin-endpoint auth and re-index concurrency.

Background:
  * /admin/reindex, /admin/stats and /admin/test-llm were unauthenticated. A
    loop over /admin/reindex was a free way to stack crawlers against the
    department's web server, and /admin/test-llm spent LLM credits per call.
  * run_reindex had no mutual exclusion, so the 2AM cron firing while a manual
    re-index was still running started overlapping crawls that raced each
    other's upserts into ecen_docs.

main.py can't be imported in CI (fastapi/slowapi/torch aren't installed), so the
auth wiring is asserted against the source. That is the regression that actually
matters here: a future /admin route added without the dependency.
"""
import ast
import asyncio
import os
import re
import sys
import types

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..")
MAIN_PY = os.path.join(REPO, "backend", "main.py")


def _main_source():
    with open(MAIN_PY) as f:
        return f.read()


# ── Every admin route is authenticated ───────────────────────────────────────

def _admin_routes():
    """(path, decorator_source) for every route whose path starts with /admin."""
    tree = ast.parse(_main_source())
    src = _main_source().splitlines()
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            func = dec.func
            if not (isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "app"
                    and func.attr in {"get", "post", "put", "delete", "patch"}):
                continue
            if not dec.args or not isinstance(dec.args[0], ast.Constant):
                continue
            path = dec.args[0].value
            if isinstance(path, str) and path.startswith("/admin"):
                text = "\n".join(src[dec.lineno - 1:dec.end_lineno])
                found.append((path, text))
    return found


def test_admin_routes_exist():
    paths = [p for p, _ in _admin_routes()]
    assert paths, "expected at least one /admin route in main.py"


@pytest.mark.parametrize("path,decorator", _admin_routes())
def test_every_admin_route_requires_admin(path, decorator):
    """Any /admin endpoint must carry Depends(require_admin). Adding one without
    it silently reopens the hole this test was written for."""
    assert "require_admin" in decorator, (
        f"{path} is not protected — add "
        f"dependencies=[Depends(require_admin)] to its decorator")


def test_admin_auth_fails_closed_when_token_unset():
    """With no ADMIN_TOKEN configured the endpoints must be disabled, not open —
    a missing env var in a fresh deployment must not expose them."""
    src = _main_source()
    body = src[src.index("async def require_admin"):]
    body = body[:body.index("\n# ── Endpoints")] if "\n# ── Endpoints" in body else body
    assert "if not ADMIN_TOKEN" in body
    assert "503" in body, "unconfigured admin auth should refuse, not allow"


def test_admin_auth_uses_constant_time_comparison():
    """A plain `==` on a secret leaks it a character at a time under timing
    analysis."""
    src = _main_source()
    assert "hmac.compare_digest" in src
    assert not re.search(r"supplied\s*==\s*ADMIN_TOKEN", src)


def test_health_endpoint_is_not_behind_admin_auth():
    """Cloud Run's probe can't send a token — /health must stay open."""
    src = _main_source()
    health = src[src.index('@app.get("/health")'):]
    health = health[:health.index("async def health")]
    assert "require_admin" not in health


# ── Re-index mutual exclusion ────────────────────────────────────────────────

@pytest.fixture
def scheduler_module(monkeypatch):
    """Import backend/scheduler.py against an apscheduler stub."""
    if "apscheduler" not in sys.modules:
        aps = types.ModuleType("apscheduler")
        sched_pkg = types.ModuleType("apscheduler.schedulers")
        asyncio_mod = types.ModuleType("apscheduler.schedulers.asyncio")

        class AsyncIOScheduler:  # pragma: no cover
            def add_job(self, *a, **kw):
                pass

        asyncio_mod.AsyncIOScheduler = AsyncIOScheduler
        triggers = types.ModuleType("apscheduler.triggers")
        cron = types.ModuleType("apscheduler.triggers.cron")

        class CronTrigger:  # pragma: no cover
            def __init__(self, *a, **kw):
                pass

        cron.CronTrigger = CronTrigger
        sys.modules.update({
            "apscheduler": aps,
            "apscheduler.schedulers": sched_pkg,
            "apscheduler.schedulers.asyncio": asyncio_mod,
            "apscheduler.triggers": triggers,
            "apscheduler.triggers.cron": cron,
        })

    sys.path.insert(0, os.path.join(REPO, "backend"))
    import scheduler
    return scheduler


@pytest.mark.asyncio
async def test_second_reindex_is_skipped_while_one_runs(scheduler_module, monkeypatch):
    """The exact production hazard: cron fires while a manual re-index is still
    crawling. The second call must no-op, not start a competing crawler."""
    runs = {"n": 0}
    started = asyncio.Event()
    release = asyncio.Event()

    async def fake_locked_run():
        runs["n"] += 1
        started.set()
        await release.wait()

    monkeypatch.setattr(scheduler_module, "_run_reindex_locked", fake_locked_run)

    first = asyncio.create_task(scheduler_module.run_reindex())
    await started.wait()

    assert scheduler_module.reindex_running() is True
    await scheduler_module.run_reindex()   # should return immediately
    assert runs["n"] == 1, "overlapping re-index must be skipped"

    release.set()
    await first
    assert scheduler_module.reindex_running() is False


@pytest.mark.asyncio
async def test_lock_is_released_after_a_failed_reindex(scheduler_module, monkeypatch):
    """A crash inside the crawl must not leave the lock held — that would block
    every later re-index until the process restarted."""
    async def boom():
        raise RuntimeError("crawler died")

    monkeypatch.setattr(scheduler_module, "_run_reindex_locked", boom)

    with pytest.raises(RuntimeError):
        await scheduler_module.run_reindex()

    assert scheduler_module.reindex_running() is False


def test_reindex_endpoint_short_circuits_when_already_running():
    src = _main_source()
    body = src[src.index("async def manual_reindex"):]
    body = body[:body.index("\n\n\n")] if "\n\n\n" in body else body
    assert "reindex_running()" in body, (
        "/admin/reindex should check the lock before spawning a task")
