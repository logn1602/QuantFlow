"""
ingestion/yfinance_fetcher.py
-----------------------------
Fetches OHLCV data from Yahoo Finance via yfinance.
- fetch_historical(): pull 2 years of daily candles (run once to seed DB)
- fetch_intraday():   pull today's 15-min candles (run on schedule)
"""

import pandas as pd
import yfinance as yf

from quantflow.config import TICKERS
from quantflow.db.batch import frame_to_rows
from quantflow.db.prices import RAW_PRICE_COLUMNS, insert_bars
from quantflow.utils.logger import get_logger

logger = get_logger(__name__)

SOURCE = "yfinance"


def _bars_to_rows(df: pd.DataFrame, ticker: str) -> list[dict]:
    """raw_prices rows from a yf.download frame.

    Rows without a close are dropped, NaN becomes NULL, and every timestamp is
    pinned to UTC. yfinance returns daily bars as naive exchange-date
    midnights; leaving those for Postgres to interpret would make the stored
    instant depend on the session time zone, which is UTC on Supabase but is
    not guaranteed anywhere else.
    """
    if df.empty:
        return []

    frame = df.copy()
    frame.columns = [str(c).lower() for c in frame.columns]
    frame.index.name = "ts"
    frame = frame.reset_index()

    ts = pd.to_datetime(frame["ts"])
    frame["ts"] = (
        ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
    )
    frame["ticker"] = ticker
    frame["source"] = SOURCE
    frame = frame.dropna(subset=["close"]).drop_duplicates(subset="ts")
    return frame_to_rows(frame, RAW_PRICE_COLUMNS)


def _upsert_prices(df: pd.DataFrame, ticker: str) -> int:
    """Insert bars into raw_prices. Returns rows actually written.

    Bars already stored are skipped by ON CONFLICT and not counted. Any other
    write error raises, so a CLI run exits non-zero instead of reporting rows
    that were rolled back.
    """
    return insert_bars(_bars_to_rows(df, ticker))


def _download(ticker: str, **kwargs) -> pd.DataFrame:
    """yf.download for one ticker; a failed download is logged and returned
    as an empty frame.

    Only the download is guarded. A write error after it must still raise.
    """
    try:
        data = yf.download(ticker, progress=False, **kwargs)
    except Exception as e:
        logger.error(f"Download failed for {ticker}: {e}")
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def fetch_historical(tickers: list[str] | None = None, period: str = "2y") -> dict:
    """
    Pull daily OHLCV for the given tickers going back `period`.
    Good for seeding the DB on first run.

    Args:
        tickers: list of ticker symbols. Defaults to config.TICKERS
        period:  yfinance period string: '1y', '2y', '5y', etc.

    Returns:
        dict of {ticker: rows_inserted}
    """
    tickers = tickers or TICKERS
    results = {}

    for ticker in tickers:
        logger.info(f"Fetching historical data for {ticker} ({period})")
        data = _download(ticker, period=period, interval="1d", auto_adjust=True)
        n = _upsert_prices(data, ticker)
        results[ticker] = n
        logger.info(f"  {ticker}: {n} rows inserted")

    return results


def fetch_intraday(tickers: list[str] | None = None) -> dict:
    """
    Pull the last 5 days of 15-min candles.
    Designed to run on a schedule every 15 minutes.

    Args:
        tickers: list of ticker symbols. Defaults to config.TICKERS

    Returns:
        dict of {ticker: rows_inserted}
    """
    tickers = tickers or TICKERS
    results = {}

    for ticker in tickers:
        logger.info(f"Fetching intraday data for {ticker}")
        data = _download(ticker, period="5d", interval="15m", auto_adjust=True)
        n = _upsert_prices(data, ticker)
        results[ticker] = n
        logger.info(f"  {ticker}: {n} new rows")

    return results


if __name__ == "__main__":
    # Quick test: python -m quantflow.ingestion.yfinance_fetcher
    print("Testing yfinance fetcher with AAPL...")
    r = fetch_historical(["AAPL"], period="1mo")
    print(f"Inserted: {r}")
