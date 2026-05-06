"""
scheduler/job_runner.py
------------------------
APScheduler job runner. Runs the full QuantFlow pipeline on schedule.

Jobs:
  - yFinance intraday      : every 15 min (market hours)
  - Alpha Vantage intraday : every 15 min, offset by 5 min
  - Technical indicators   : every 15 min (after price fetch)
  - Anomaly detection      : every hour
  - Sentiment analysis     : every 6 hours (4x per day)
  - ARIMA / Prophet        : daily at market close (4:30 PM ET)
  - XGBoost / LightGBM     : daily at market close (4:45 PM ET)
  - Stacking Ensemble      : daily after base models (5:00 PM ET)
  - Backtesting            : daily after ensemble (5:15 PM ET)

Usage:
    python scheduler/job_runner.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datetime import datetime, timedelta

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from config import FETCH_INTERVAL_MINUTES
from db.connection import test_connection
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Job functions ─────────────────────────────────────────────────────────────

def run_yfinance_job():
    logger.info("--- yFinance intraday job started ---")
    try:
        from ingestion.yfinance_fetcher import fetch_intraday
        results = fetch_intraday()
        total = sum(results.values())
        logger.info(f"--- yFinance job done: {total} rows inserted ---")
    except Exception as e:
        logger.error(f"yFinance job failed: {e}")


def run_alpha_vantage_job():
    logger.info("--- Alpha Vantage intraday job started ---")
    try:
        from ingestion.alpha_vantage_fetcher import fetch_intraday
        results = fetch_intraday()
        total = sum(results.values())
        logger.info(f"--- Alpha Vantage job done: {total} rows inserted ---")
    except Exception as e:
        logger.error(f"Alpha Vantage job failed: {e}")


def run_indicators_job():
    logger.info("--- Indicators job started ---")
    try:
        from indicators import run as indicators_run
        results = indicators_run()
        total = sum(results.values())
        logger.info(f"--- Indicators job done: {total} rows saved ---")
    except Exception as e:
        logger.error(f"Indicators job failed: {e}")


def run_anomaly_job():
    logger.info("--- Anomaly detection job started ---")
    try:
        from anomaly_detection import run as anomaly_run
        results = anomaly_run()
        total = sum(results.values())
        logger.info(f"--- Anomaly job done: {total} anomalies flagged ---")
    except Exception as e:
        logger.error(f"Anomaly job failed: {e}")


def run_sentiment_job():
    logger.info("--- Sentiment job started ---")
    try:
        from sentiment import run as sentiment_run
        results = sentiment_run()
        total = sum(results.values())
        logger.info(f"--- Sentiment job done: {total} rows saved ---")
    except Exception as e:
        logger.error(f"Sentiment job failed: {e}")


def run_forecasting_job():
    logger.info("--- ARIMA/Prophet forecasting job started ---")
    try:
        from forecasting import run as forecasting_run
        results = forecasting_run()
        logger.info(f"--- Forecasting job done: {results} ---")
    except Exception as e:
        logger.error(f"Forecasting job failed: {e}")


def run_xgboost_job():
    logger.info("--- XGBoost/LightGBM job started ---")
    try:
        from xgboost_model import run as xgb_run
        results = xgb_run()
        logger.info(f"--- XGBoost job done: {results} ---")
    except Exception as e:
        logger.error(f"XGBoost job failed: {e}")


def run_ensemble_job():
    """Stacking ensemble — runs after ARIMA/Prophet + XGBoost/LightGBM are done."""
    logger.info("--- Stacking Ensemble job started ---")
    try:
        from ensemble import run as ensemble_run
        results = ensemble_run()
        for ticker, n in results.items():
            logger.info(f"  {ticker} [ensemble_stack]: {n} rows saved")
        logger.info("--- Ensemble job done ---")
    except Exception as e:
        logger.error(f"Ensemble job failed: {e}")


def run_backtest_job():
    """Backtesting — runs after ensemble forecasts are ready."""
    logger.info("--- Backtest job started ---")
    try:
        from backtest import run as backtest_run
        results = backtest_run()
        for ticker, m in results.items():
            if m:
                logger.info(
                    f"  {ticker}: return={m['total_return']:+.2f}% | "
                    f"alpha={m['alpha']:+.2f}% | sharpe={m['sharpe_ratio']:.3f}"
                )
        logger.info("--- Backtest job done ---")
    except Exception as e:
        logger.error(f"Backtest job failed: {e}")


# ── Scheduler setup ───────────────────────────────────────────────────────────

def start():
    if not test_connection():
        logger.error("Cannot reach database. Check your .env DB settings. Exiting.")
        sys.exit(1)

    scheduler = BlockingScheduler(timezone="America/New_York")

    # ── Every 15 min: price ingestion ────────────────────────────────────────
    scheduler.add_job(
        run_yfinance_job,
        trigger=IntervalTrigger(minutes=FETCH_INTERVAL_MINUTES),
        id="yfinance_intraday",
        replace_existing=True,
    )

    # Alpha Vantage offset by 5 min to avoid rate limit collision
    av_start = datetime.now() + timedelta(minutes=5)
    scheduler.add_job(
        run_alpha_vantage_job,
        trigger=IntervalTrigger(
            minutes=FETCH_INTERVAL_MINUTES,
            start_date=av_start,
        ),
        id="alpha_vantage_intraday",
        replace_existing=True,
    )

    # ── Every 15 min: recompute indicators (offset by 8 min) ─────────────────
    ind_start = datetime.now() + timedelta(minutes=8)
    scheduler.add_job(
        run_indicators_job,
        trigger=IntervalTrigger(
            minutes=FETCH_INTERVAL_MINUTES,
            start_date=ind_start,
        ),
        id="indicators",
        replace_existing=True,
    )

    # ── Every hour: anomaly detection ────────────────────────────────────────
    scheduler.add_job(
        run_anomaly_job,
        trigger=IntervalTrigger(hours=1),
        id="anomaly_detection",
        replace_existing=True,
    )

    # ── Every 6 hours: sentiment analysis ────────────────────────────────────
    scheduler.add_job(
        run_sentiment_job,
        trigger=IntervalTrigger(hours=6),
        id="sentiment",
        replace_existing=True,
    )

    # ── Daily at 4:30 PM ET: ARIMA + Prophet forecasting ─────────────────────
    scheduler.add_job(
        run_forecasting_job,
        trigger=CronTrigger(hour=16, minute=30, timezone="America/New_York"),
        id="forecasting",
        replace_existing=True,
    )

    # ── Daily at 4:45 PM ET: XGBoost + LightGBM ──────────────────────────────
    # Runs after forecasting so all features are fresh
    scheduler.add_job(
        run_xgboost_job,
        trigger=CronTrigger(hour=16, minute=45, timezone="America/New_York"),
        id="xgboost",
        replace_existing=True,
    )

    # ── Daily at 5:00 PM ET: Stacking Ensemble ────────────────────────────────
    # Runs after all 4 base models have completed
    scheduler.add_job(
        run_ensemble_job,
        trigger=CronTrigger(hour=17, minute=0, timezone="America/New_York"),
        id="ensemble",
        replace_existing=True,
    )

    # ── Daily at 5:15 PM ET: Backtesting ─────────────────────────────────────
    # Runs after ensemble forecasts are saved
    scheduler.add_job(
        run_backtest_job,
        trigger=CronTrigger(hour=17, minute=15, timezone="America/New_York"),
        id="backtest",
        replace_existing=True,
    )

    logger.info("QuantFlow scheduler started. Jobs:")
    logger.info(f"  Price ingestion  : every {FETCH_INTERVAL_MINUTES} min")
    logger.info(f"  Indicators       : every {FETCH_INTERVAL_MINUTES} min")
    logger.info(f"  Anomaly detection: every 60 min")
    logger.info(f"  Sentiment        : every 6 hours")
    logger.info(f"  ARIMA/Prophet    : daily at 4:30 PM ET")
    logger.info(f"  XGBoost/LightGBM : daily at 4:45 PM ET")
    logger.info(f"  Stacking Ensemble: daily at 5:00 PM ET")
    logger.info(f"  Backtesting      : daily at 5:15 PM ET")
    logger.info("Press Ctrl+C to stop.")

    # Run immediately on start
    run_yfinance_job()
    run_indicators_job()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    start()