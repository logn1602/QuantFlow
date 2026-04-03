"""
scheduler/job_runner.py
------------------------
APScheduler job runner. Runs the ingestion pipeline on a fixed interval.
Runs both yfinance and Alpha Vantage fetchers.

Usage:
    python scheduler/job_runner.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config import FETCH_INTERVAL_MINUTES
from ingestion.yfinance_fetcher import fetch_intraday as yf_intraday
from ingestion.alpha_vantage_fetcher import fetch_intraday as av_intraday
from db.connection import test_connection
from utils.logger import get_logger

logger = get_logger(__name__)


def run_yfinance_job():
    logger.info("--- yFinance intraday job started ---")
    results = yf_intraday()
    total = sum(results.values())
    logger.info(f"--- yFinance job done: {total} total rows inserted ---")


def run_alpha_vantage_job():
    logger.info("--- Alpha Vantage intraday job started ---")
    results = av_intraday()
    total = sum(results.values())
    logger.info(f"--- Alpha Vantage job done: {total} total rows inserted ---")


def start():
    # Verify DB is reachable before starting scheduler
    if not test_connection():
        logger.error("Cannot reach database. Check your .env DB settings. Exiting.")
        sys.exit(1)

    scheduler = BlockingScheduler(timezone="America/New_York")

    # yFinance: every FETCH_INTERVAL_MINUTES
    scheduler.add_job(
        run_yfinance_job,
        trigger=IntervalTrigger(minutes=FETCH_INTERVAL_MINUTES),
        id="yfinance_intraday",
        next_run_time=None,   # don't run immediately on start
        replace_existing=True,
    )

    # Alpha Vantage: every FETCH_INTERVAL_MINUTES, offset by 5 min
    # to avoid hitting rate limits at the same time as yfinance
    from datetime import datetime, timedelta
    av_start = datetime.now() + timedelta(minutes=5)
    scheduler.add_job(
        run_alpha_vantage_job,
        trigger=IntervalTrigger(minutes=FETCH_INTERVAL_MINUTES, start_date=av_start),
        id="alpha_vantage_intraday",
        replace_existing=True,
    )

    logger.info(
        f"Scheduler started. Jobs run every {FETCH_INTERVAL_MINUTES} min. "
        f"Press Ctrl+C to stop."
    )

    # Run once immediately so you see output right away
    run_yfinance_job()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    start()
