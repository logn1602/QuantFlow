"""
db/metrics.py
-------------
Persistence for per-run model evaluation metrics.

MLflow stores metrics in a local sqlite file, which Streamlit Cloud
cannot read. The dashboard therefore reads its MAPE figures from the
model_metrics table in Postgres, written here.

Metric persistence is best-effort by design: a failure to save a metric
must never take down a model run.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sqlalchemy import text

from db.connection import get_engine
from utils.logger import get_logger

logger = get_logger(__name__)

MODELS = ["arima", "prophet", "xgboost", "lightgbm", "ensemble_stack"]


def _as_float(value):
    """Coerce a metric to float, or None when missing/non-numeric."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def save_model_metrics(
    ticker: str, model: str, metrics: dict, holdout_days: int
) -> None:
    """
    Write one metrics row for (ticker, model). Missing rmse/mae/mape keys
    are stored as NULL. Never raises — logs a warning and returns instead.
    """
    if not metrics:
        logger.warning(f"No metrics to save for {ticker} [{model}]")
        return

    try:
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO model_metrics
                        (ticker, model, holdout_days, rmse, mae, mape)
                    VALUES
                        (:ticker, :model, :holdout_days, :rmse, :mae, :mape)
                    ON CONFLICT (ticker, model, run_at) DO NOTHING
                """),
                {
                    "ticker": ticker,
                    "model": model,
                    "holdout_days": int(holdout_days),
                    "rmse": _as_float(metrics.get("rmse")),
                    "mae": _as_float(metrics.get("mae")),
                    "mape": _as_float(metrics.get("mape")),
                },
            )
        logger.info(
            f"  Metrics saved — {ticker} [{model}] "
            f"MAPE: {metrics.get('mape')} over {holdout_days} days"
        )
    except Exception as e:
        logger.warning(
            f"Metric persistence failed for {ticker} [{model}] (non-critical): {e}"
        )


def load_latest_metrics(ticker: str) -> dict:
    """
    Return the most recent metrics row per model for a ticker:
        {model: {"rmse", "mae", "mape", "holdout_days", "run_at"}}
    Returns {} if nothing is stored or the query fails.
    """
    try:
        engine = get_engine()
        query = text("""
            SELECT DISTINCT ON (model)
                   model, holdout_days, rmse, mae, mape, run_at
            FROM model_metrics
            WHERE ticker = :ticker
            ORDER BY model, run_at DESC
        """)
        with engine.connect() as conn:
            rows = conn.execute(query, {"ticker": ticker}).mappings().all()
    except Exception as e:
        logger.warning(f"Could not load metrics for {ticker}: {e}")
        return {}

    return {
        row["model"]: {
            "rmse": _as_float(row["rmse"]),
            "mae": _as_float(row["mae"]),
            "mape": _as_float(row["mape"]),
            "holdout_days": row["holdout_days"],
            "run_at": row["run_at"],
        }
        for row in rows
    }
