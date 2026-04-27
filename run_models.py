"""
run_models.py
-------------
Runs the full forecasting pipeline in one command:
  1. ARIMA + Prophet  (statistical time series models)
  2. XGBoost + LightGBM (ML models with 33 engineered features)

Clears stale forecasts before retraining to keep data clean.

Usage:
    python run_models.py                  # all tickers, all models
    python run_models.py --ticker AAPL    # one ticker only
    python run_models.py --clear          # clear old forecasts first
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(__file__))

from config import TICKERS
from db.connection import get_engine
from utils.logger import get_logger
from sqlalchemy import text

logger = get_logger("run_models")


def clear_forecasts(tickers: list[str] = None):
    """Clear existing forecasts so old data doesn't pollute new runs."""
    tickers = tickers or TICKERS
    engine = get_engine()
    with engine.begin() as conn:
        for ticker in tickers:
            conn.execute(
                text("DELETE FROM forecasts WHERE ticker = :t"),
                {"t": ticker}
            )
    logger.info(f"Cleared old forecasts for: {', '.join(tickers)}")


def run(tickers: list[str] = None, clear: bool = True):
    """
    Run all 4 forecasting models in sequence.
    Args:
        tickers: list of tickers. Defaults to config.TICKERS
        clear:   whether to clear old forecasts before retraining
    """
    tickers = tickers or TICKERS

    if clear:
        logger.info("Clearing old forecasts...")
        clear_forecasts(tickers)

    # ── Step 1: ARIMA + Prophet ───────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Step 1 — ARIMA + Prophet")
    logger.info("=" * 50)
    try:
        from forecasting import run as forecast_run
        forecast_results = forecast_run(tickers=tickers)
        for ticker, model_results in forecast_results.items():
            for model, n in model_results.items():
                logger.info(f"  {ticker} [{model}]: {n} rows saved")
    except Exception as e:
        logger.error(f"ARIMA/Prophet failed: {e}")

    # ── Step 2: XGBoost + LightGBM ───────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Step 2 — XGBoost + LightGBM")
    logger.info("=" * 50)
    try:
        from xgboost_model import run as xgb_run
        xgb_results = xgb_run(tickers=tickers)
        for ticker, model_results in xgb_results.items():
            for model, n in model_results.items():
                logger.info(f"  {ticker} [{model}]: {n} rows saved")
    except Exception as e:
        logger.error(f"XGBoost/LightGBM failed: {e}")

    # ── Step 3: Stacking Ensemble ─────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Step 3 — Stacking Ensemble (Ridge meta-learner)")
    logger.info("=" * 50)
    try:
        from ensemble import run as ensemble_run
        ensemble_results = ensemble_run(tickers=tickers)
        for ticker, n in ensemble_results.items():
            logger.info(f"  {ticker} [ensemble_stack]: {n} rows saved")
    except Exception as e:
        logger.error(f"Stacking ensemble failed: {e}")

    logger.info("=" * 50)
    logger.info("All models complete. Launch dashboard: streamlit run dashboard.py")
    logger.info("=" * 50)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all forecasting models")
    parser.add_argument("--ticker", type=str, help="Single ticker e.g. AAPL")
    parser.add_argument(
        "--clear", action="store_true", default=True,
        help="Clear old forecasts before retraining (default: True)"
    )
    parser.add_argument(
        "--no-clear", action="store_false", dest="clear",
        help="Keep existing forecasts and only add new ones"
    )
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else None
    run(tickers=tickers, clear=args.clear)