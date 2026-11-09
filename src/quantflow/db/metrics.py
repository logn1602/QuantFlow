"""
db/metrics.py
-------------
Persistence for per-run model evaluation metrics.

Each daily run records its 30-day holdout figures in the model_metrics
table, a record kept in Postgres because MLflow's local sqlite file never
leaves the machine that ran it. Nothing displays them now: the dashboard
shows the 7-day error the rolling-origin evaluation measured
(dashboard/accuracy.py; docs/architecture.md, Known problems 1 and 2).

Metric persistence is best-effort by design: a failure to save a metric
must never take down a model run.
"""

from sqlalchemy import text

from quantflow.db.connection import get_engine
from quantflow.utils.logger import get_logger

logger = get_logger(__name__)


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
