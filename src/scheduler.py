"""
Scheduler.

Runs pipeline jobs on a timer using APScheduler.

Usage:
    # Run the scheduler (blocks, runs until Ctrl+C)
    python -m src.scheduler

    # Or import and start in a thread from app.py
    from src.scheduler import start_scheduler_thread
    start_scheduler_thread()
"""

import logging
import threading
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from src.config import get_config
from src.pipeline import (
    sync_wc2026_fixtures,
    sync_recent_results,
    sync_odds,
    compute_team_form_stats,
    promote_wc_results,
)
from src.scraper import FootballDataAPI
from src.db import init_db

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _create_scheduler() -> BackgroundScheduler:
    """Build and configure the scheduler with all jobs."""
    cfg = get_config()
    init_db()

    api = FootballDataAPI(api_key=cfg.get("football_data_api_key"))
    scheduler = BackgroundScheduler(timezone="UTC")

    # --- Job 1: Sync fixtures & results every 2 hours ---
    scheduler.add_job(
        sync_wc2026_fixtures,
        trigger=IntervalTrigger(minutes=120),
        args=[api],
        id="sync_fixtures",
        name="Sync WC2026 fixtures",
        replace_existing=True,
        max_instances=1,
    )

    # --- Job 2: Sync recent results every 30 minutes (during match days) ---
    scheduler.add_job(
        sync_recent_results,
        trigger=IntervalTrigger(minutes=30),
        args=[api, 1],  # last 1 day only
        id="sync_results",
        name="Sync recent results",
        replace_existing=True,
        max_instances=1,
    )

    # --- Job 3: Promote finished WC matches to historical (after results sync) ---
    scheduler.add_job(
        promote_wc_results,
        trigger=IntervalTrigger(minutes=30),
        id="promote_results",
        name="Promote WC results to historical",
        replace_existing=True,
        max_instances=1,
    )

    # --- Job 4: Fetch odds every 30 minutes ---
    scheduler.add_job(
        sync_odds,
        trigger=IntervalTrigger(minutes=30),
        kwargs={"odds_api_key": cfg.get("odds_api_key")},
        id="sync_odds",
        name="Sync betting odds",
        replace_existing=True,
        max_instances=1,
    )

    # --- Job 5: Recompute team form stats every 3 hours ---
    scheduler.add_job(
        compute_team_form_stats,
        trigger=IntervalTrigger(hours=3),
        id="compute_form",
        name="Compute team form stats",
        replace_existing=True,
        max_instances=1,
    )

    # --- Job 6: Model retraining (daily at 4am UTC) ---
    # This calls the retrain script if it exists.
    # Uncomment once the model.py is ready:
    #
    # scheduler.add_job(
    #     _retrain_model,
    #     trigger=CronTrigger(hour=4, minute=0),
    #     id="retrain_model",
    #     name="Daily model retrain",
    #     replace_existing=True,
    #     max_instances=1,
    # )

    return scheduler


def _retrain_model():
    """
    Wrapper for model retraining
    this just calls the retrain function.
    """
    try:
        from src.model import retrain
        retrain()
        logger.info("Model retrained successfully")
    except ImportError:
        logger.warning("src.model.retrain not found — DS hasn't built it yet")
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_scheduler_instance = None


def start_scheduler_thread() -> BackgroundScheduler:
    """
    Start the scheduler in a background thread.
    Safe to call from Streamlit's app.py — won't block.
    Returns the scheduler instance (for status checks).
    """
    global _scheduler_instance
    if _scheduler_instance and _scheduler_instance.running:
        logger.info("Scheduler already running")
        return _scheduler_instance

    _scheduler_instance = _create_scheduler()
    _scheduler_instance.start()
    logger.info("Scheduler started in background thread")
    return _scheduler_instance


def get_scheduler_status() -> list[dict]:
    """Get status of all scheduled jobs (for dashboard display)."""
    if not _scheduler_instance:
        return []

    jobs = []
    for job in _scheduler_instance.get_jobs():
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": str(job.next_run_time) if job.next_run_time else "paused",
            "trigger": str(job.trigger),
        })
    return jobs


# ---------------------------------------------------------------------------
# Standalone mode
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    logger.info("Starting scheduler in standalone mode (Ctrl+C to stop)...")
    scheduler = _create_scheduler()
    scheduler.start()

    # Run initial sync immediately
    logger.info("Running initial sync...")
    cfg = get_config()
    api = FootballDataAPI(api_key=cfg.get("football_data_api_key"))
    sync_wc2026_fixtures(api)
    sync_recent_results(api)
    promote_wc_results()
    sync_odds(odds_api_key=cfg.get("odds_api_key"))
    compute_team_form_stats()

    logger.info("Initial sync complete. Scheduler running.")
    logger.info("Scheduled jobs:")
    for job in scheduler.get_jobs():
        logger.info(f"  {job.name}: next run at {job.next_run_time}")

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Scheduler stopped.")