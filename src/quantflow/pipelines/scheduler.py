"""
scheduler/job_runner.py
------------------------
APScheduler job runner. Runs the full QuantFlow pipeline on schedule.

Jobs:
  - yFinance intraday      : every 15 min (market hours)
  - Split guard            : at startup, 4:20 PM ET, and after the daily bars
  - yFinance daily bars    : daily after the close (5:05 PM ET)
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
from datetime import datetime, time, timedelta

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from quantflow import config
from quantflow.config import FETCH_INTERVAL_MINUTES, TICKERS
from quantflow.db.connection import test_connection
from quantflow.db.forecasts import clear_forecasts as db_clear_forecasts
from quantflow.ingestion import split_guard
from quantflow.ingestion.yfinance_fetcher import DAILY_REFRESH_PERIOD
from quantflow.utils.logger import get_logger
from quantflow.utils.trading_calendar import DAILY_BAR_FINAL_AFTER, EXCHANGE_TZ

logger = get_logger(__name__)

# Before the 4:30 PM model chain, so a split revealed by the day's bars stops
# it rather than being modelled.
SPLIT_GUARD_AT = time(16, 20)
# Five minutes after today's daily bar is final.
DAILY_BARS_AT = (
    datetime.combine(datetime.min, DAILY_BAR_FINAL_AFTER) + timedelta(minutes=5)
).time()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _clear_model_forecasts(models: list[str], tickers: list[str] | None = None):
    """Delete prior forecast rows for the given models so the scheduler
    doesn't accumulate one row per run per forecast_date.

    Only the models this job is about to rewrite are cleared, so a failing
    job never blanks another model's forecasts. If the job then crashes
    mid-run the dashboard shows nothing for these models until the next
    run — missing is better than silently averaged with stale predictions.
    """
    tickers = tickers or TICKERS
    deleted = db_clear_forecasts(tickers, models)
    logger.info(
        f"  Cleared {deleted} prior forecast rows for "
        f"{', '.join(models)} across {len(tickers)} tickers"
    )
    return deleted


# ── Price-basis gate ──────────────────────────────────────────────────────────
#
# After a stock split, every row stored before it sits on the old share basis
# (see ingestion/split_guard.py). Indicators, anomaly flags, forecasts and the
# backtest computed from that mix would be wrong, and they are stored. So the
# jobs that read prices skip while the split guard's last answer was a break.
# The guard keeps no state of its own; this is the scheduler remembering its
# last answer. It starts empty, and start() runs the guard before any job
# that reads prices.

_price_basis_error: str | None = None


def _paused_for_prices(job: str) -> bool:
    """True, after logging why, while stored prices are on more than one basis."""
    if _price_basis_error is None:
        return False
    logger.error(
        f"{job} skipped: {_price_basis_error}. It resumes once the split guard "
        f"passes; the guard re-checks at {SPLIT_GUARD_AT:%H:%M} and after the "
        f"daily bars ({DAILY_BARS_AT:%H:%M} New York time)."
    )
    return True


# ── Job functions ─────────────────────────────────────────────────────────────
#
# Failure policy — deliberately different from the CLI entry points.
#
# run_models.py and backtest.py exit non-zero on failure so CI goes red. The
# scheduler must NOT: it is a long-running process, and a transient failure
# (network blip, rate limit, market holiday) has to leave the remaining jobs
# scheduled rather than killing the daemon. So every job swallows its
# exception, but logs it with logger.exception so the full traceback lands in
# logs/pipeline.log instead of a one-line message with no context.


def run_yfinance_job():
    logger.info("--- yFinance intraday job started ---")
    try:
        from quantflow.ingestion.yfinance_fetcher import fetch_intraday

        results = fetch_intraday()
        total = sum(results.values())
        logger.info(f"--- yFinance job done: {total} rows inserted ---")
    except Exception as e:
        logger.exception(f"yFinance job failed: {type(e).__name__}: {e}")


def run_split_guard_job():
    """Check stored prices against Yahoo's, and pause or resume the price jobs.

    Only a basis break pauses them. Any other failure, such as the database
    being unreachable, is logged and leaves the last answer standing, and so
    does a pass that could not check every ticker while a break is outstanding.
    """
    global _price_basis_error
    logger.info("--- Split guard job started ---")
    try:
        unchecked = split_guard.run()
    except split_guard.PriceBasisError as e:
        _price_basis_error = f"stored prices are on more than one basis ({e})"
        logger.error(f"--- Split guard failed: {e}. Price jobs paused ---")
        return
    except Exception as e:
        logger.exception(f"Split guard job failed: {type(e).__name__}: {e}")
        return

    if _price_basis_error is not None:
        if unchecked:
            logger.warning(
                f"Split guard could not re-check {', '.join(unchecked)}; "
                f"price jobs stay paused"
            )
            return
        logger.info("Split guard passes again; price jobs resume")
    _price_basis_error = None
    logger.info("--- Split guard job done ---")


def run_daily_bars_job():
    """Store each date's official daily bar once the session is final, then
    re-run the split guard on the result.

    Nothing in the model chain waits for this job. The 4:30 PM model jobs
    build today's bar from its 15-min bars, which closes within a few
    hundredths of a percent of the official close, and the next day's runs
    read the official bar stored here. Making the models wait on it would add
    one more wall-clock dependency to a scheduler that has no completion
    signals (Known problem #4 in docs/architecture.md).
    """
    logger.info("--- yFinance daily bars job started ---")
    try:
        from quantflow.ingestion.yfinance_fetcher import fetch_historical

        results = fetch_historical(period=DAILY_REFRESH_PERIOD)
        total = sum(results.values())
        logger.info(f"--- Daily bars job done: {total} rows inserted ---")
    except Exception as e:
        logger.exception(f"Daily bars job failed: {type(e).__name__}: {e}")
    run_split_guard_job()


def run_alpha_vantage_job():
    logger.info("--- Alpha Vantage intraday job started ---")
    try:
        from quantflow.ingestion.alpha_vantage_fetcher import fetch_intraday

        results = fetch_intraday()
        total = sum(results.values())
        logger.info(f"--- Alpha Vantage job done: {total} rows inserted ---")
    except Exception as e:
        logger.exception(f"Alpha Vantage job failed: {type(e).__name__}: {e}")


def run_indicators_job():
    logger.info("--- Indicators job started ---")
    if _paused_for_prices("Indicators job"):
        return
    try:
        from quantflow.features.indicators import run as indicators_run

        results = indicators_run()
        total = sum(results.values())
        logger.info(f"--- Indicators job done: {total} rows saved ---")
    except Exception as e:
        logger.exception(f"Indicators job failed: {type(e).__name__}: {e}")


def run_anomaly_job():
    logger.info("--- Anomaly detection job started ---")
    if _paused_for_prices("Anomaly job"):
        return
    try:
        from quantflow.features.anomalies import run as anomaly_run

        results = anomaly_run()
        total = sum(results.values())
        logger.info(f"--- Anomaly job done: {total} anomalies flagged ---")
    except Exception as e:
        logger.exception(f"Anomaly job failed: {type(e).__name__}: {e}")


def run_sentiment_job():
    logger.info("--- Sentiment job started ---")
    try:
        from quantflow.features.sentiment import run as sentiment_run

        results = sentiment_run()
        total = sum(results.values())
        logger.info(f"--- Sentiment job done: {total} rows saved ---")
    except Exception as e:
        logger.exception(f"Sentiment job failed: {type(e).__name__}: {e}")


def run_forecasting_job():
    logger.info("--- ARIMA/Prophet forecasting job started ---")
    if _paused_for_prices("Forecasting job"):
        return
    try:
        from quantflow.models.statistical import run as forecasting_run

        _clear_model_forecasts(["arima", "prophet"])
        results = forecasting_run()
        logger.info(f"--- Forecasting job done: {results} ---")
    except Exception as e:
        logger.exception(f"Forecasting job failed: {type(e).__name__}: {e}")


def run_xgboost_job():
    logger.info("--- XGBoost/LightGBM job started ---")
    if _paused_for_prices("XGBoost job"):
        return
    try:
        from quantflow.models.boosting import run as xgb_run

        _clear_model_forecasts(["xgboost", "lightgbm"])
        results = xgb_run()
        logger.info(f"--- XGBoost job done: {results} ---")
    except Exception as e:
        logger.exception(f"XGBoost job failed: {type(e).__name__}: {e}")


def run_ensemble_job():
    """Stacking ensemble — runs after ARIMA/Prophet + XGBoost/LightGBM are done."""
    logger.info("--- Stacking Ensemble job started ---")
    if _paused_for_prices("Ensemble job"):
        return
    try:
        from quantflow.models.ensemble import run as ensemble_run

        _clear_model_forecasts(["ensemble_stack"])
        results = ensemble_run()
        for ticker, n in results.items():
            logger.info(f"  {ticker} [ensemble_stack]: {n} rows saved")
        logger.info("--- Ensemble job done ---")
    except Exception as e:
        logger.exception(f"Ensemble job failed: {type(e).__name__}: {e}")


def run_backtest_job():
    """Backtesting — runs after ensemble forecasts are ready."""
    logger.info("--- Backtest job started ---")
    if _paused_for_prices("Backtest job"):
        return
    try:
        from quantflow.evaluation.backtest import run as backtest_run

        results = backtest_run()
        for ticker, m in results.items():
            if m:
                logger.info(
                    f"  {ticker}: return={m['total_return']:+.2f}% | "
                    f"alpha={m['alpha']:+.2f}% | sharpe={m['sharpe_ratio']:.3f}"
                )
        logger.info("--- Backtest job done ---")
    except Exception as e:
        logger.exception(f"Backtest job failed: {type(e).__name__}: {e}")


# ── Scheduler setup ───────────────────────────────────────────────────────────


def start():
    # Config first, then connectivity. A long-running daemon started with a
    # broken .env should die immediately rather than log a failure every
    # 15 minutes forever.
    config.validate()
    config.require_or_exit()

    try:
        split_guard.configured_threshold()
    except ValueError as e:
        logger.error(f"{e}. Exiting.")
        sys.exit(1)

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

    # ── Daily at 4:20 PM ET: split guard, before the model chain ─────────────
    # Both this job and the next run however late they start (one run for any
    # number of missed slots). APScheduler otherwise skips a run that starts
    # more than a second late, so a laptop waking at 16:25 skipped the guard
    # and the model chain ran on the previous day's answer.
    scheduler.add_job(
        run_split_guard_job,
        trigger=CronTrigger(
            hour=SPLIT_GUARD_AT.hour, minute=SPLIT_GUARD_AT.minute, timezone=EXCHANGE_TZ
        ),
        id="split_guard",
        replace_existing=True,
        misfire_grace_time=None,
        coalesce=True,
    )

    # ── Daily at 5:05 PM ET: official daily bars, then the split guard ───────
    # After DAILY_BAR_FINAL_AFTER (5:00 PM ET), so today's bar is stored.
    scheduler.add_job(
        run_daily_bars_job,
        trigger=CronTrigger(
            hour=DAILY_BARS_AT.hour, minute=DAILY_BARS_AT.minute, timezone=EXCHANGE_TZ
        ),
        id="yfinance_daily",
        replace_existing=True,
        misfire_grace_time=None,
        coalesce=True,
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
    logger.info(
        f"  Split guard      : at startup, {SPLIT_GUARD_AT:%H:%M} ET and after the "
        f"daily bars; while it fails, the jobs that read prices skip"
    )
    logger.info(f"  Daily bars       : daily at {DAILY_BARS_AT:%H:%M} ET")
    logger.info(f"  Indicators       : every {FETCH_INTERVAL_MINUTES} min")
    logger.info("  Anomaly detection: every 60 min")
    logger.info("  Sentiment        : every 6 hours")
    logger.info("  ARIMA/Prophet    : daily at 4:30 PM ET")
    logger.info("  XGBoost/LightGBM : daily at 4:45 PM ET")
    logger.info("  Stacking Ensemble: daily at 5:00 PM ET")
    logger.info("  Backtesting      : daily at 5:15 PM ET")
    logger.info("Press Ctrl+C to stop.")

    # Run immediately on start. The split guard runs before the first job that
    # reads prices, as it does in the daily workflow.
    run_yfinance_job()
    run_split_guard_job()
    run_indicators_job()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    start()
