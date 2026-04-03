"""
ingestion/alpha_vantage_fetcher.py
-----------------------------------
Fetches intraday and daily OHLCV from Alpha Vantage REST API.
Free tier: 25 requests/day, 15-min interval data available.

Docs: https://www.alphavantage.co/documentation/
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import requests
import pandas as pd
from sqlalchemy import text

from config import ALPHA_VANTAGE_API_KEY, TICKERS
from db.connection import get_engine
from utils.logger import get_logger

logger = get_logger(__name__)

BASE_URL = "https://www.alphavantage.co/query"
SOURCE   = "alpha_vantage"

# Free tier allows 25 req/day — add a small delay between calls to be safe
REQUEST_DELAY_SECONDS = 12


def _upsert_prices(df: pd.DataFrame, ticker: str) -> int:
    """Insert rows into raw_prices, skip duplicates. Returns rows inserted."""
    if df.empty:
        return 0

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
                    {
                        "ticker": ticker,
                        "source": SOURCE,
                        "ts":     row["ts"],
                        "open":   row["open"],
                        "high":   row["high"],
                        "low":    row["low"],
                        "close":  row["close"],
                        "volume": row["volume"],
                    },
                )
                inserted += 1
            except Exception as e:
                logger.warning(f"Row skipped for {ticker}: {e}")

    return inserted


def fetch_intraday(tickers: list[str] = None, interval: str = "15min") -> dict:
    """
    Fetch intraday OHLCV from Alpha Vantage.

    Args:
        tickers:  list of symbols. Defaults to config.TICKERS
        interval: '1min' | '5min' | '15min' | '30min' | '60min'

    Returns:
        dict of {ticker: rows_inserted}
    """
    if not ALPHA_VANTAGE_API_KEY:
        logger.error("ALPHA_VANTAGE_API_KEY is not set. Skipping Alpha Vantage fetch.")
        return {}

    tickers = tickers or TICKERS
    results = {}

    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(REQUEST_DELAY_SECONDS)

        logger.info(f"Alpha Vantage intraday fetch: {ticker} ({interval})")
        try:
            resp = requests.get(
                BASE_URL,
                params={
                    "function":   "TIME_SERIES_INTRADAY",
                    "symbol":     ticker,
                    "interval":   interval,
                    "outputsize": "compact",   # last 100 data points
                    "apikey":     ALPHA_VANTAGE_API_KEY,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            key = f"Time Series ({interval})"
            if key not in data:
                # API returns error messages as plain keys
                note = data.get("Note") or data.get("Information") or data
                logger.warning(f"No data for {ticker}: {note}")
                results[ticker] = 0
                continue

            rows = []
            for ts_str, ohlcv in data[key].items():
                rows.append({
                    "ts":     pd.Timestamp(ts_str, tz="US/Eastern"),
                    "open":   float(ohlcv["1. open"]),
                    "high":   float(ohlcv["2. high"]),
                    "low":    float(ohlcv["3. low"]),
                    "close":  float(ohlcv["4. close"]),
                    "volume": int(ohlcv["5. volume"]),
                })

            df = pd.DataFrame(rows)
            n = _upsert_prices(df, ticker)
            results[ticker] = n
            logger.info(f"  {ticker}: {n} rows inserted")

        except requests.RequestException as e:
            logger.error(f"HTTP error for {ticker}: {e}")
            results[ticker] = 0

    return results


def fetch_daily(tickers: list[str] = None) -> dict:
    """
    Fetch daily adjusted OHLCV from Alpha Vantage.
    Useful for longer historical seeding.

    Args:
        tickers: list of symbols. Defaults to config.TICKERS

    Returns:
        dict of {ticker: rows_inserted}
    """
    if not ALPHA_VANTAGE_API_KEY:
        logger.error("ALPHA_VANTAGE_API_KEY not set.")
        return {}

    tickers = tickers or TICKERS
    results = {}

    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(REQUEST_DELAY_SECONDS)

        logger.info(f"Alpha Vantage daily fetch: {ticker}")
        try:
            resp = requests.get(
                BASE_URL,
                params={
                    "function":   "TIME_SERIES_DAILY",
                    "symbol":     ticker,
                    "outputsize": "compact",   # last 100 trading days
                    "apikey":     ALPHA_VANTAGE_API_KEY,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            key = "Time Series (Daily)"
            if key not in data:
                note = data.get("Note") or data.get("Information") or data
                logger.warning(f"No data for {ticker}: {note}")
                results[ticker] = 0
                continue

            rows = []
            for date_str, ohlcv in data[key].items():
                rows.append({
                    "ts":     pd.Timestamp(date_str, tz="UTC"),
                    "open":   float(ohlcv["1. open"]),
                    "high":   float(ohlcv["2. high"]),
                    "low":    float(ohlcv["3. low"]),
                    "close":  float(ohlcv["4. close"]),
                    "volume": int(ohlcv["5. volume"]),
                })

            df = pd.DataFrame(rows)
            n = _upsert_prices(df, ticker)
            results[ticker] = n
            logger.info(f"  {ticker}: {n} rows inserted")

        except requests.RequestException as e:
            logger.error(f"HTTP error for {ticker}: {e}")
            results[ticker] = 0

    return results


if __name__ == "__main__":
    print("Testing Alpha Vantage fetcher with AAPL (daily)...")
    r = fetch_daily(["AAPL"])
    print(f"Inserted: {r}")
