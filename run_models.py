"""
run_models.py
-------------
Runs the full forecasting pipeline in one command:
  1. ARIMA + Prophet  (statistical time series models)
  2. XGBoost + LightGBM (ML models with 33 engineered features)
  3. Stacking Ensemble  (NNLS meta-learner over all 4 base models)

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


def run(tickers: list[str] = None, clear: bool = True) -> tuple[dict, list[str]]:
    """
    Run all 5 forecasting models in sequence.

    Args:
        tickers: list of tickers. Defaults to config.TICKERS
        clear:   whether to clear old forecasts before retraining

    Returns:
        (results, failed_steps) — each step stays individually wrapped so one
        model family failing does not stop the others, but the failures are
        reported back so the caller can set a real exit code instead of
        exiting 0 on a total failure.
    """
    tickers = tickers or TICKERS
    results = {}
    failed  = []

    if clear:
        logger.info("Clearing old forecasts...")
        try:
            clear_forecasts(tickers)
        except Exception as e:
            logger.exception(f"Clearing forecasts failed: {type(e).__name__}: {e}")
            failed.append("clear_forecasts")

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
        results["forecasting"] = forecast_results
    except Exception as e:
        logger.exception(f"ARIMA/Prophet failed: {type(e).__name__}: {e}")
        failed.append("arima_prophet")

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
        results["xgboost_lightgbm"] = xgb_results
    except Exception as e:
        logger.exception(f"XGBoost/LightGBM failed: {type(e).__name__}: {e}")
        failed.append("xgboost_lightgbm")

    # ── Step 3: Stacking Ensemble ─────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Step 3 — Stacking Ensemble (NNLS meta-learner)")
    logger.info("=" * 50)
    try:
        from ensemble import run as ensemble_run
        ensemble_results = ensemble_run(tickers=tickers)
        for ticker, n in ensemble_results.items():
            logger.info(f"  {ticker} [ensemble_stack]: {n} rows saved")
        results["ensemble"] = ensemble_results
    except Exception as e:
        logger.exception(f"Stacking ensemble failed: {type(e).__name__}: {e}")
        failed.append("ensemble")

    logger.info("=" * 50)
    if failed:
        logger.error(f"Pipeline finished with {len(failed)} failed step(s): "
                     f"{', '.join(failed)}")
    else:
        logger.info("All models complete. Launch dashboard: streamlit run dashboard.py")
    logger.info("=" * 50)

    return results, failed


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
    results, failed = run(tickers=tickers, clear=args.clear)

    # Any failed step is a failed run. Previously every exception was swallowed
    # and the process still exited 0, so a total failure showed up as a green
    # CI run.
    if failed:
        print("\n--- PIPELINE FAILED ---", file=sys.stderr)
        for step in failed:
            print(f"  FAILED: {step}", file=sys.stderr)
        print(f"{len(failed)} step(s) failed — see the log above for tracebacks.",
              file=sys.stderr)
        sys.exit(1)

    print("\nAll pipeline steps completed successfully.")