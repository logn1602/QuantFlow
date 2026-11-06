"""
ingestion/split_guard.py
------------------------
Fail the refresh when stored prices no longer agree with Yahoo's.

Yahoo serves every bar split-adjusted as of the moment it is fetched, and
rows here are never rewritten. So after a split, every row stored before it
sits on the old share basis (a 10-for-1 split leaves them ten times too high)
and the models would read the split as a 90% crash. This compares each
ticker's stored daily closes with Yahoo's current raw closes for the same
dates and fails loudly if any date differs by more than SPLIT_GUARD_THRESHOLD.

Why a threshold rather than equality: stored closes legitimately differ from
Yahoo's raw closes a little. The seeded history was stored dividend-adjusted
as of the seed date (about 2% a year at most on these tickers), and dates
built from 15-min bars differ from the official close by a few hundredths of
a percent. A split moves prices by far more.

The check keeps no state: once the affected rows are rescaled, it passes again
by itself. Rescaling is a scoped UPDATE on the live database, run by hand.

Usage:
    python -m quantflow.ingestion.split_guard
"""

import argparse
import sys

import pandas as pd

from quantflow.config import SPLIT_GUARD_THRESHOLD, TICKERS
from quantflow.db.prices import load_daily_bars
from quantflow.ingestion.yfinance_fetcher import download
from quantflow.utils.logger import get_logger

logger = get_logger(__name__)


class PriceBasisError(RuntimeError):
    """Stored prices and Yahoo's disagree by more than the threshold."""


def find_basis_breaks(
    stored: pd.DataFrame, reference: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """Dates where stored and reference closes differ by more than threshold.

    Both frames need date and close columns. Returns date, stored, yahoo and
    ratio (stored / yahoo) for each offending date; empty when none do.
    """
    merged = stored[["date", "close"]].merge(
        reference[["date", "close"]], on="date", suffixes=("_stored", "_yahoo")
    )
    ratio = merged["close_stored"] / merged["close_yahoo"]
    breaks = merged.assign(ratio=ratio)[(ratio - 1).abs() > threshold]
    return breaks.rename(
        columns={"close_stored": "stored", "close_yahoo": "yahoo"}
    ).reset_index(drop=True)


def yahoo_daily_closes(ticker: str, start: str) -> pd.DataFrame:
    """Yahoo's raw daily closes from start, as date and close columns."""
    data = download(ticker, start=start, interval="1d", auto_adjust=False)
    if data.empty:
        return pd.DataFrame(columns=["date", "close"])
    index = pd.DatetimeIndex(data.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    return pd.DataFrame(
        {"date": index.normalize(), "close": data["Close"].astype(float).to_numpy()}
    )


def check(
    tickers: list[str] | None = None, threshold: float | None = None
) -> dict[str, pd.DataFrame]:
    """Basis breaks per ticker; tickers with none are left out."""
    threshold = SPLIT_GUARD_THRESHOLD if threshold is None else threshold
    if not 0 < threshold < 1:
        raise ValueError(
            f"SPLIT_GUARD_THRESHOLD must be a fraction between 0 and 1, "
            f"got {threshold}; use 0.15 for 15%"
        )

    breaks = {}
    for ticker in tickers or TICKERS:
        stored = load_daily_bars(ticker)
        if stored.empty:
            continue
        reference = yahoo_daily_closes(ticker, stored["date"].min().date().isoformat())
        if reference.empty:
            logger.warning(f"{ticker}: no reference prices from Yahoo; not checked")
            continue
        found = find_basis_breaks(stored, reference, threshold)
        if found.empty:
            logger.info(f"{ticker}: stored prices agree with Yahoo")
        else:
            breaks[ticker] = found
    return breaks


def run(tickers: list[str] | None = None, threshold: float | None = None) -> None:
    """Raise PriceBasisError, after logging each break, if any ticker has one."""
    breaks = check(tickers, threshold)
    if not breaks:
        return

    for ticker, found in breaks.items():
        logger.error(
            f"{ticker}: {len(found)} stored daily closes differ from Yahoo's by "
            f"more than the threshold, {found['date'].min().date()} to "
            f"{found['date'].max().date()}, median stored/Yahoo ratio "
            f"{found['ratio'].median():.4f}. A stock split after these rows were "
            f"stored is the usual cause: rescale them before the next run."
        )
    raise PriceBasisError(
        "stored prices disagree with Yahoo for " + ", ".join(sorted(breaks))
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail if stored prices disagree with Yahoo's (a stock split)"
    )
    parser.add_argument("--ticker", help="one ticker instead of config.TICKERS")
    args = parser.parse_args(argv)

    try:
        run([args.ticker.upper()] if args.ticker else None)
    except PriceBasisError as e:
        print(f"SPLIT GUARD FAILED: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
