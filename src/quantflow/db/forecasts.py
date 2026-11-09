"""Writer and readers for the forecasts table.

Consolidates three near-duplicate implementations from forecasting.py,
xgboost_model.py and ensemble.py. They differed only in how defensively they
coerced the forecast date and in whether run_at came from pd.Timestamp.now()
or datetime.now(); the SQL was identical. The coercion here is the union of all
three, so every previous call site gets the same result.

The daily run (pipelines/run_models.py) holds its forecasts in a ForecastSet
until every model is done, then replaces each ticker's stored forecasts in one
transaction (replace_forecasts): a reader, the dashboard among them, finds
the previous set or the new one, never a ticker with some models or none.
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

    ARIMA and Prophet pass datetime.date already; the boosters pass the
    pd.Timestamp values next_trading_days returns; the ensemble passes the
    forecast_date values it read back. numpy datetime64 is accepted too.
    Strings are passed through for Postgres to parse, matching the previous
    ensemble behaviour.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (pd.Timestamp, _dt.datetime)):
        return value.date()
    if hasattr(value, "date"):
        return value.date()
    return value


def _rows(df: pd.DataFrame) -> list[dict]:
    """A forecast frame (ticker, model, ds and the three prices) as the rows
    stored for it, the prices rounded as the table keeps them, all with one
    run_at: now."""
    run_at = _dt.datetime.now()
    return [
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
    with get_engine().begin() as conn:
        return insert_rows(conn, "forecasts", _rows(df), skip_conflicts=True)


# What load_base_forecasts returns, in its order.
BASE_COLUMNS = [
    "model",
    "forecast_date",
    "predicted_close",
    "lower_bound",
    "upper_bound",
]


class ForecastSet:
    """One run's forecasts, held until the run is done, then stored a ticker
    at a time (replace_forecasts, with rows).

    save stands in for save_forecasts and base for load_base_forecasts, so
    a model step and the ensemble work the same on a set as on the table: a
    model saved twice for a ticker keeps its later forecast, as readers of the
    table take each model's latest run.
    """

    def __init__(self) -> None:
        self._held: dict[tuple[str, str], list[dict]] = {}

    def save(self, df: pd.DataFrame) -> int:
        """Holds the forecast's rows; returns how many."""
        rows = _rows(df) if not df.empty else []
        made: dict[tuple[str, str], list[dict]] = {}
        for row in rows:
            made.setdefault((row["ticker"], row["model"]), []).append(row)
        self._held.update(made)
        return len(rows)

    def rows(self, ticker: str) -> list[dict]:
        """Every held row of the ticker, as replace_forecasts stores them."""
        return [
            row
            for (held_ticker, _), held in self._held.items()
            if held_ticker == ticker
            for row in held
        ]

    def base(self, ticker: str, models: list[str]) -> pd.DataFrame:
        """The held forecasts of those models for the ticker, shaped as
        load_base_forecasts reads them from the table."""
        rows = [row for row in self.rows(ticker) if row["model"] in models]
        frame = pd.DataFrame(rows, columns=["ticker", *BASE_COLUMNS, "run_at"])
        frame = frame[BASE_COLUMNS].sort_values(["forecast_date", "model"])
        return frame.reset_index(drop=True)


def replace_forecasts(ticker: str, rows: list[dict]) -> int:
    """Delete the ticker's stored forecasts and insert rows (ForecastSet.rows)
    in one transaction; returns rows written. If the insert fails, the delete
    rolls back with it, so the previous forecasts stay. No rows is refused: it
    would leave the ticker with none."""
    if not rows:
        raise ValueError(f"{ticker}: no forecast rows to replace the stored ones")
    with get_engine().begin() as conn:
        _delete(conn, [ticker], None)
        return insert_rows(conn, "forecasts", rows, skip_conflicts=True)


def load_base_forecasts(ticker: str, models: list[str]) -> pd.DataFrame:
    """Each model's most recent stored forecast run: model, forecast_date and
    the predicted close with its bounds. The ensemble's input.

    Only the latest run_at per model, the rule the dashboard uses. The table
    keeps every run (run_at is part of its key), so after a run with
    --no-clear, reading every row gave the ensemble two predictions per date.
    """
    query = text("""
        SELECT f.model, f.forecast_date, f.predicted_close,
               f.lower_bound, f.upper_bound
        FROM forecasts f
        JOIN (
            SELECT model, MAX(run_at) AS run_at
            FROM forecasts
            WHERE ticker = :ticker
              AND model = ANY(:models)
            GROUP BY model
        ) latest
          ON f.model = latest.model AND f.run_at = latest.run_at
        WHERE f.ticker = :ticker
        ORDER BY f.forecast_date, f.model
    """)
    with get_engine().connect() as conn:
        return pd.read_sql(query, conn, params={"ticker": ticker, "models": models})


def clear_forecasts(tickers: list[str], models: list[str] | None = None) -> int:
    """Delete prior forecast rows so a rerun does not accumulate one row per run
    per forecast_date. Returns rows deleted.

    models=None clears every model for those tickers. Passing a list clears
    only those models, which is what the scheduler needs: a job must never
    blank forecasts belonging to a model it is not about to rewrite, or a
    mid-run crash leaves the dashboard showing another model's stale rows. The
    daily run does not clear first; it replaces each ticker's forecasts at the
    end (ForecastSet, replace_forecasts).
    """
    with get_engine().begin() as conn:
        return _delete(conn, tickers, models)


def _delete(conn, tickers: list[str], models: list[str] | None) -> int:
    if models is None:
        sql = "DELETE FROM forecasts WHERE ticker = ANY(:tickers)"
        params: dict = {"tickers": tickers}
    else:
        sql = (
            "DELETE FROM forecasts "
            "WHERE ticker = ANY(:tickers) AND model = ANY(:models)"
        )
        params = {"tickers": tickers, "models": models}
    return conn.execute(text(sql), params).rowcount
