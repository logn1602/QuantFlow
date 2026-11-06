"""Writer and readers for the forecasts table.

Consolidates three near-duplicate implementations from forecasting.py,
xgboost_model.py and ensemble.py. They differed only in how defensively they
coerced the forecast date and in whether run_at came from pd.Timestamp.now()
or datetime.now(); the SQL was identical. The coercion here is the union of all
three, so every previous call site gets the same result.
"""

import datetime as _dt

import pandas as pd
from sqlalchemy import text

from quantflow.db.batch import insert_rows
from quantflow.db.connection import get_engine
from quantflow.utils.logger import get_logger

logger = get_logger(__name__)


def _as_date(value):
    """Coerce a forecast date to datetime.date.

    Callers pass pd.Timestamp (ARIMA/Prophet via bdate_range), numpy
    datetime64, or an already-converted date. Strings are passed through for
    Postgres to parse, matching the previous ensemble behaviour.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (pd.Timestamp, _dt.datetime)):
        return value.date()
    if hasattr(value, "date"):
        return value.date()
    return value


def save_forecasts(df: pd.DataFrame) -> int:
    """Insert forecast rows, skipping conflicts. Returns rows written.

    One batched INSERT inside one transaction. Any error other than a skipped
    conflict raises and rolls the whole forecast back, so a model run never
    reports forecasts it did not store. It used to catch errors row by row
    instead, which cannot work on Postgres: the first failure aborts the
    transaction, every later row fails with it, and the commit rolls all of
    them back.
    """
    if df.empty:
        return 0

    run_at = _dt.datetime.now()
    rows = [
        {
            "ticker": row["ticker"],
            "model": row["model"],
            "forecast_date": _as_date(row["ds"]),
            "predicted_close": round(float(row["predicted_close"]), 4),
            "lower_bound": round(float(row["lower_bound"]), 4),
            "upper_bound": round(float(row["upper_bound"]), 4),
            "run_at": run_at,
        }
        for row in df.to_dict("records")
    ]
    with get_engine().begin() as conn:
        return insert_rows(conn, "forecasts", rows, skip_conflicts=True)


def load_base_forecasts(ticker: str, models: list[str]) -> pd.DataFrame:
    """Stored forecasts for the given models: model, forecast_date and the
    predicted close with its bounds. The ensemble's input."""
    query = text("""
        SELECT model, forecast_date, predicted_close, lower_bound, upper_bound
        FROM forecasts
        WHERE ticker = :ticker
          AND model = ANY(:models)
        ORDER BY forecast_date, model
    """)
    with get_engine().connect() as conn:
        return pd.read_sql(query, conn, params={"ticker": ticker, "models": models})


def clear_forecasts(tickers: list[str], models: list[str] | None = None) -> int:
    """Delete prior forecast rows so a rerun does not accumulate one row per run
    per forecast_date. Returns rows deleted.

    models=None clears every model for those tickers — the full reset that
    run_models performs before regenerating the whole forecast set. Passing a
    list clears only those models, which is what the scheduler needs: a job
    must never blank forecasts belonging to a model it is not about to rewrite,
    or a mid-run crash leaves the dashboard showing another model's stale rows.
    """
    engine = get_engine()
    if models is None:
        sql = "DELETE FROM forecasts WHERE ticker = ANY(:tickers)"
        params = {"tickers": tickers}
    else:
        sql = (
            "DELETE FROM forecasts "
            "WHERE ticker = ANY(:tickers) AND model = ANY(:models)"
        )
        params = {"tickers": tickers, "models": models}

    with engine.begin() as conn:
        return conn.execute(text(sql), params).rowcount
