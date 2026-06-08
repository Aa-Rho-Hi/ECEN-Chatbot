"""
scheduler.py — APScheduler job that re-indexes the TAMU ECE site daily.
Runs inside the FastAPI process; can also be triggered via POST /admin/reindex.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

load_dotenv(override=True)

log = logging.getLogger(__name__)

REINDEX_CRON = os.getenv("REINDEX_CRON", "0 2 * * *")   # default: 2 AM daily
CRAWLER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "crawler"))

# Tracks the outcome of the last run — readable via GET /health
last_reindex: dict = {"status": "never_run", "finished_at": None, "message": ""}


async def run_reindex() -> None:
    """Runs the crawler + ingest pipeline in diff mode (only new/changed pages).

    Uses asyncio.create_subprocess_exec so the FastAPI event loop is NOT blocked
    during the crawl (which can take up to an hour).
    """
    global last_reindex
    started_at = datetime.now(timezone.utc).isoformat()
    log.info("Scheduled re-index starting at %s (cwd: %s)", started_at, CRAWLER_DIR)

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "ingest.py", "--diff",
            cwd=CRAWLER_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=3600)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            msg = "Re-index timed out after 1 hour."
            log.error(msg)
            last_reindex = {"status": "timeout", "finished_at": datetime.now(timezone.utc).isoformat(), "message": msg}
            return

        stdout_text = stdout.decode(errors="replace").strip()
        stderr_text = stderr.decode(errors="replace").strip()

        if stdout_text:
            log.info("Re-index stdout:\n%s", stdout_text)
        if stderr_text:
            log.info("Re-index stderr:\n%s", stderr_text)

        if proc.returncode == 0:
            msg = f"Re-index completed successfully at {datetime.now(timezone.utc).isoformat()}"
            log.info(msg)
            last_reindex = {"status": "ok", "finished_at": datetime.now(timezone.utc).isoformat(), "message": stdout_text[-500:] if stdout_text else "done"}
        else:
            msg = f"Re-index failed with exit code {proc.returncode}"
            log.error("%s\n%s", msg, stderr_text)
            last_reindex = {"status": "error", "finished_at": datetime.now(timezone.utc).isoformat(), "message": stderr_text[-500:]}

    except Exception as e:
        msg = f"Re-index error: {e}"
        log.exception(msg)
        last_reindex = {"status": "error", "finished_at": datetime.now(timezone.utc).isoformat(), "message": str(e)}


def create_scheduler() -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler()

    parts = REINDEX_CRON.split()
    if len(parts) == 5:
        minute, hour, day, month, day_of_week = parts
        trigger = CronTrigger(
            minute=minute, hour=hour, day=day, month=month, day_of_week=day_of_week,
            timezone="America/Chicago",   # TAMU is in Central Time
        )
    else:
        log.warning("Invalid REINDEX_CRON '%s', defaulting to 2 AM daily.", REINDEX_CRON)
        trigger = CronTrigger(hour=2, minute=0, timezone="America/Chicago")

    scheduler.add_job(run_reindex, trigger=trigger, id="reindex", replace_existing=True)

    log.info("Re-index scheduled: %s (next run logged after scheduler starts)", REINDEX_CRON)
    return scheduler
