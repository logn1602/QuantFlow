"""Readers for the raw_prices table.

Consolidates four near-duplicate loaders that previously lived in
indicators.py, anomaly_detection.py, forecasting.py and dashboard.py. They are
kept as four distinct functions rather than merged into one: their projections,
casts and indexes genuinely differ, and callers depend on those exact shapes.
What is shared -- engine handling, the empty-result warning, ordering -- is
factored into _read.
"""

import pandas as pd
from sqlalchemy import text

from quantflow.db.connection import get_engine
from quantflow.utils.logger import get_logger

logger = get_logger(__name__)


def _read(sql: str, params: dict, ticker: str) -> pd.DataFrame:
    """Run a raw_prices query and warn once if it returns nothing."""
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)
    if df.empty:
        logger.warning(f"No price data found for {ticker}")
    return df


def load_ohlcv(ticker: str, source: str = "yfinance") -> pd.DataFrame:
    """Full OHLCV indexed by UTC timestamp, all columns float.

    Used by the indicator engine, which needs every column to compute
    Bollinger Bands and MACD.
    """
    df = _read(
        """
        SELECT ts, open, high, low, close, volume
        FROM raw_prices
        WHERE ticker = :ticker AND source = :source
        ORDER BY ts ASC
        """,
        {"ticker": ticker, "source": source},
        ticker,
    )
    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts")
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def load_close_volume(ticker: str, source: str = "yfinance") -> pd.DataFrame:
    """Close and volume only, indexed by UTC timestamp.

    Used by anomaly detection, which scores the close series and needs no
    other column.
    """
    df = _read(
        """
        SELECT ts, close, volume
        FROM raw_prices
        WHERE ticker = :ticker AND source = :source
        ORDER BY ts ASC
        """,
        {"ticker": ticker, "source": source},
        ticker,
    )
    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts")
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df


def load_daily_close(ticker: str, source: str = "yfinance") -> pd.DataFrame:
    """One close price per calendar date, as columns ds and y.

    The ds/y naming is Prophet's required input contract, and every forecasting
    model consumes this shape. Intraday bars are collapsed by casting ts to a
    date and dropping duplicates, which is what turns ~2,500 intraday rows per
    ticker into the daily series the models actually train on.
    """
    df = _read(
        """
        SELECT ts::date AS ds, close AS y
        FROM raw_prices
        WHERE ticker = :ticker AND source = :source
        ORDER BY ts ASC
        """,
        {"ticker": ticker, "source": source},
        ticker,
    )
    if df.empty:
        return df

    df["ds"] = pd.to_datetime(df["ds"]).dt.normalize()
    df["y"] = df["y"].astype(float)
    return df.drop_duplicates(subset="ds").sort_values("ds").reset_index(drop=True)


def load_recent_ohlcv(ticker: str, days: int = 180) -> pd.DataFrame:
    """Daily OHLCV over a trailing window, as a date column rather than an index.

    Dashboard-facing. The window is bound as a parameter, not interpolated.
    """
    df = _read(
        """
        SELECT ts::date AS date, open, high, low, close, volume
        FROM raw_prices
        WHERE ticker = :ticker AND source = 'yfinance'
          AND ts >= NOW() - INTERVAL '1 day' * :days
        ORDER BY ts ASC
        """,
        {"ticker": ticker, "days": days},
        ticker,
    )
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    return df
