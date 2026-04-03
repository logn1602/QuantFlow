"""
ingestion/yfinance_fetcher.py
-----------------------------
Fetches OHLCV data from Yahoo Finance via yfinance.
- fetch_historical(): pull 2 years of daily candles (run once to seed DB)
- fetch_intraday():   pull today's 15-min candles (run on schedule)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import yfinance as yf
from sqlalchemy import text

from config import TICKERS
from db.connection import get_engine
from utils.logger import get_logger

logger = get_logger(__name__)

SOURCE = "yfinance"


def _upsert_prices(df: pd.DataFrame, ticker: str) -> int:
    """Insert rows into raw_prices, skip duplicates. Returns rows inserted."""
    if df.empty:
        return 0

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df["ticker"] = ticker
    df["source"] = SOURCE

    # Rename 'datetime' index to 'ts'
    df.index.name = "ts"
    df = df.reset_index()

    # Keep only the columns we care about
    df = df[["ticker", "source", "ts", "open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["close"])

    engine = get_engine()
    inserted = 0

    with engine.begin() as conn:
        for _, row in df.iterrows():
            try:
                conn.execute(
                    text("""
                        INSERT INTO raw_prices
                            (ticker, source, ts, open, high, low, close, volume)
                        VALUES
                            (:ticker, :source, :ts, :open, :high, :low, :close, :volume)
                        ON CONFLICT (ticker, source, ts) DO NOTHING
                    """),
                    row.to_dict(),
                )
                inserted += 1
            except Exception as e:
                logger.warning(f"Row skipped for {ticker} @ {row['ts']}: {e}")

    return inserted


def fetch_historical(tickers: list[str] = None, period: str = "2y") -> dict:
    """
    Pull daily OHLCV for the given tickers going back `period`.
    Good for seeding the DB on first run.

    Args:
        tickers: list of ticker symbols. Defaults to config.TICKERS
        period:  yfinance period string — '1y', '2y', '5y', etc.

    Returns:
        dict of {ticker: rows_inserted}
    """
    tickers = tickers or TICKERS
    results = {}

    for ticker in tickers:
        try:
            logger.info(f"Fetching historical data for {ticker} ({period})")
            data = yf.download(
                ticker,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            n = _upsert_prices(data, ticker)
            results[ticker] = n
            logger.info(f"  {ticker}: {n} rows inserted")

        except Exception as e:
            logger.error(f"Failed to fetch historical data for {ticker}: {e}")
            results[ticker] = 0

    return results


def fetch_intraday(tickers: list[str] = None) -> dict:
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
        try:
            logger.info(f"Fetching intraday data for {ticker}")
            data = yf.download(
                ticker,
                period="5d",
                interval="15m",
                auto_adjust=True,
                progress=False,
            )
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            n = _upsert_prices(data, ticker)
            results[ticker] = n
            logger.info(f"  {ticker}: {n} new rows")

        except Exception as e:
            logger.error(f"Failed to fetch intraday for {ticker}: {e}")
            results[ticker] = 0

    return results


if __name__ == "__main__":
    # Quick test — run directly: python ingestion/yfinance_fetcher.py
    print("Testing yfinance fetcher with AAPL...")
    r = fetch_historical(["AAPL"], period="1mo")
    print(f"Inserted: {r}")
