"""
run_models.py
-------------
Runs the full forecasting pipeline in one command:
  1. ARIMA + Prophet  (statistical time series models)
  2. XGBoost + LightGBM (ML models with 33 engineered features)
  3. Stacking Ensemble  (NNLS meta-learner over all 4 base models)

Clears stale forecasts before retraining to keep data clean.

Clearing stale forecasts is the DEFAULT — every run recomputes the full
forecast set, so keeping the old rows would leave the dashboard averaging
across runs.

Usage:
    python run_models.py                  # all tickers, all models
    python run_models.py --ticker AAPL    # one ticker only
    python run_models.py --no-clear       # keep existing forecasts (rarely wanted)
"""

import argparse
import sys

from quantflow import config
from quantflow.config import TICKERS
from quantflow.db.forecasts import clear_forecasts as db_clear_forecasts
from quantflow.utils.logger import get_logger

logger = get_logger("run_models")


def clear_forecasts(tickers: list[str] | None = None):
    """Clear existing forecasts so old data doesn't pollute new runs."""
    tickers = tickers or TICKERS
    db_clear_forecasts(tickers)
    logger.info(f"Cleared old forecasts for: {', '.join(tickers)}")


def run(tickers: list[str] | None = None, clear: bool = True) -> tuple[dict, list[str]]:
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
    failed = []

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
        from quantflow.models.statistical import run as forecast_run

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
        from quantflow.models.boosting import run as xgb_run

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
        from quantflow.models.ensemble import run as ensemble_run

        ensemble_results = ensemble_run(tickers=tickers)
        for ticker, n in ensemble_results.items():
            logger.info(f"  {ticker} [ensemble_stack]: {n} rows saved")
        results["ensemble"] = ensemble_results
    except Exception as e:
        logger.exception(f"Stacking ensemble failed: {type(e).__name__}: {e}")
        failed.append("ensemble")

    logger.info("=" * 50)
    if failed:
        logger.error(
            f"Pipeline finished with {len(failed)} failed step(s): {', '.join(failed)}"
        )
    else:
        logger.info("All models complete. Launch dashboard: streamlit run dashboard.py")
    logger.info("=" * 50)

    return results, failed


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all forecasting models")
    parser.add_argument("--ticker", type=str, help="Single ticker e.g. AAPL")
    # Clearing is the default, so there is no --clear flag: the old one was a
    # no-op (store_true with default=True can only ever be True) and implied
    # clearing was opt-in when it never was.
    parser.add_argument(
        "--no-clear",
        action="store_false",
        dest="clear",
        default=True,
        help="Keep existing forecasts instead of replacing them (rarely wanted)",
    )
    args = parser.parse_args()

    # Fail fast on missing config rather than surfacing it as an auth error
    # deep inside a model job.
    config.validate()
    config.require_or_exit()

    tickers = [args.ticker.upper()] if args.ticker else None
    results, failed = run(tickers=tickers, clear=args.clear)

    # Any failed step is a failed run. Previously every exception was swallowed
    # and the process still exited 0, so a total failure showed up as a green
    # CI run.
    if failed:
        print("\n--- PIPELINE FAILED ---", file=sys.stderr)
        for step in failed:
            print(f"  FAILED: {step}", file=sys.stderr)
        print(
            f"{len(failed)} step(s) failed — see the log above for tracebacks.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\nAll pipeline steps completed successfully.")
