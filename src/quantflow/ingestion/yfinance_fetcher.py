"""
ingestion/yfinance_fetcher.py
-----------------------------
Fetches OHLCV data from Yahoo Finance via yfinance.
- fetch_historical(): daily candles: 2 years to seed the DB, or the last few
                      days after each close (--daily)
- fetch_intraday():   the last 5 days of 15-min candles (--intraday)

Prices are stored raw (auto_adjust=False): split-adjusted, as Yahoo serves
every bar, but not dividend-adjusted. Yahoo never dividend-adjusts 15-min
bars, so adjusted daily bars would sit on a different basis from the 15-min
bars of the same date, and a stored value would depend on when it was fetched.
The consequences are written up under "Price data" in docs/architecture.md.

Usage:
    python -m quantflow.ingestion.yfinance_fetcher --daily
    python -m quantflow.ingestion.yfinance_fetcher --intraday
    python -m quantflow.ingestion.yfinance_fetcher --daily --period 1mo --ticker AAPL
"""

import argparse
import sys

import pandas as pd
import yfinance as yf

from quantflow.config import TICKERS
from quantflow.db.batch import frame_to_rows
from quantflow.db.prices import RAW_PRICE_COLUMNS, insert_bars
from quantflow.utils.logger import get_logger
from quantflow.utils.trading_calendar import DAILY_BAR_FINAL_AFTER, EXCHANGE_TZ

logger = get_logger(__name__)

SOURCE = "yfinance"


def _now() -> pd.Timestamp:
    """The current time; a seam so tests can fix it."""
    return pd.Timestamp.now(tz="UTC")


def drop_unfinished_session(data: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """data without today's daily bar while that bar may not be final.

    Yahoo returns today's daily bar during the session too, holding the latest
    price as its close. Stored then, ON CONFLICT DO NOTHING would keep that
    snapshot forever. Until DAILY_BAR_FINAL_AFTER, the loaders build today's
    bar from its 15-min bars instead, and the next run stores the final one.
    """
    if data.empty:
        return data
    now_local = now.tz_convert(EXCHANGE_TZ)
    if now_local.time() >= DAILY_BAR_FINAL_AFTER:
        return data

    index = pd.DatetimeIndex(data.index)
    if index.tz is not None:
        index = index.tz_convert(EXCHANGE_TZ).tz_localize(None)
    unfinished = index.normalize() == now_local.normalize().tz_localize(None)
    if unfinished.any():
        logger.info("  Skipping today's daily bar: the session is not final yet")
    return data[~unfinished]


def bars_to_rows(df: pd.DataFrame, ticker: str) -> list[dict]:
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
    return insert_bars(bars_to_rows(df, ticker))


def download(ticker: str, **kwargs) -> pd.DataFrame:
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
    Pull raw daily OHLCV for the given tickers going back `period`.
    "2y" seeds the DB; the daily refresh asks for "5d" after each close, so
    a missed run heals itself on the next one.

    Args:
        tickers: list of ticker symbols. Defaults to config.TICKERS
        period:  yfinance period string: '5d', '1y', '2y', etc.

    Returns:
        dict of {ticker: rows_inserted}
    """
    tickers = tickers or TICKERS
    results = {}

    for ticker in tickers:
        logger.info(f"Fetching daily bars for {ticker} ({period})")
        data = download(ticker, period=period, interval="1d", auto_adjust=False)
        data = drop_unfinished_session(data, _now())
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
        data = download(ticker, period="5d", interval="15m", auto_adjust=False)
        n = _upsert_prices(data, ticker)
        results[ticker] = n
        logger.info(f"  {ticker}: {n} new rows")

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch Yahoo Finance bars")
    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument("--daily", action="store_true", help="daily bars")
    kind.add_argument("--intraday", action="store_true", help="15-min bars")
    parser.add_argument(
        "--period", default="5d", help="period for --daily (default: 5d)"
    )
    parser.add_argument("--ticker", help="one ticker instead of config.TICKERS")
    args = parser.parse_args(argv)

    tickers = [args.ticker.upper()] if args.ticker else None
    if args.daily:
        results = fetch_historical(tickers, period=args.period)
    else:
        results = fetch_intraday(tickers)
    logger.info(f"Done: {sum(results.values())} rows written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
