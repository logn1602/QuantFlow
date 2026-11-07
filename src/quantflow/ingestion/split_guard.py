"""
ingestion/split_guard.py
------------------------
Fail the refresh when stored prices no longer agree with Yahoo's.

Yahoo serves every bar split-adjusted as of the moment it is fetched, and
rows here are never rewritten. So after a split, every row stored before it
sits on the old share basis (a 10-for-1 split leaves them ten times too high)
and the models would read the split as a 90% crash. This compares each
ticker's stored daily closes with Yahoo's current raw closes for the same
dates and fails loudly when they disagree by more than SPLIT_GUARD_THRESHOLD
on a run of consecutive dates.

Why a threshold rather than equality: stored closes legitimately differ from
Yahoo's raw closes a little. The seeded history was stored dividend-adjusted
as of the seed date (about 2% a year at most on these tickers), and dates
built from 15-min bars differ from the official close by a few hundredths of
a percent. A split moves prices by far more.

Why a run of dates: a split shifts every row stored before it, which is a
long unbroken run. One or two isolated mismatches are a bad print on one
side, and failing the refresh on them would stop every stage until someone
hand-edited a row. They are logged as warnings instead.

The check keeps no state: once the affected rows are rescaled, it passes again
by itself. Rescaling is a scoped UPDATE on the live database, run by hand.

Usage:
    python -m quantflow.ingestion.split_guard
"""

import argparse
import sys

import pandas as pd

from quantflow import config
from quantflow.config import TICKERS
from quantflow.db.prices import load_daily_bars
from quantflow.ingestion.yfinance_fetcher import download
from quantflow.utils.logger import get_logger
from quantflow.utils.trading_calendar import exchange_dates

logger = get_logger(__name__)

# A fraction, not a percentage. Any stock split moves prices by far more
# (3-for-2, the smallest common ratio, is 50%); dividend drift on these
# tickers stays under about 2% a year.
DEFAULT_THRESHOLD = 0.15

# Consecutive compared dates that must disagree before the guard fails.
MIN_BREAK_RUN = 3


class PriceBasisError(RuntimeError):
    """Stored prices and Yahoo's disagree by more than the threshold."""


def configured_threshold(raw: str | None = None) -> float:
    """SPLIT_GUARD_THRESHOLD as a fraction between 0 and 1.

    Parsed here rather than when config is imported, so a typo such as "15%"
    fails the guard with a message naming the variable instead of taking every
    entry point down at import. Unset or blank means DEFAULT_THRESHOLD.
    """
    raw = config.SPLIT_GUARD_THRESHOLD if raw is None else raw
    if not raw.strip():
        return DEFAULT_THRESHOLD
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(
            f"SPLIT_GUARD_THRESHOLD must be a fraction such as 0.15, got {raw!r}"
        ) from None
    return _checked(value)


def _checked(threshold: float) -> float:
    if not 0 < threshold < 1:
        raise ValueError(
            f"SPLIT_GUARD_THRESHOLD must be a fraction between 0 and 1, "
            f"got {threshold}; use 0.15 for 15%"
        )
    return threshold


def mismatched_closes(
    stored: pd.DataFrame, reference: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """Dates where stored and reference closes differ by more than threshold.

    Both frames need date and close columns. Returns date, stored, yahoo,
    ratio (stored / yahoo) and run: how many consecutive compared dates the
    mismatch belongs to, counted over the dates both sides have. Reference
    closes that are missing or not positive are left out; a zero from Yahoo
    is a bad print, not a price.
    """
    reference = reference[reference["close"] > 0]
    merged = (
        stored[["date", "close"]]
        .merge(reference[["date", "close"]], on="date", suffixes=("_stored", "_yahoo"))
        .sort_values("date")
        .reset_index(drop=True)
    )
    ratio = merged["close_stored"] / merged["close_yahoo"]
    off = (ratio - 1).abs() > threshold
    run_id = (off != off.shift()).cumsum()
    run = off.groupby(run_id).transform("size")
    found = merged.assign(ratio=ratio, run=run)[off]
    return found.rename(
        columns={"close_stored": "stored", "close_yahoo": "yahoo"}
    ).reset_index(drop=True)


def find_basis_breaks(
    stored: pd.DataFrame,
    reference: pd.DataFrame,
    threshold: float,
    min_run: int = MIN_BREAK_RUN,
) -> pd.DataFrame:
    """The mismatched closes that belong to a run of at least min_run dates."""
    found = mismatched_closes(stored, reference, threshold)
    return found[found["run"] >= min_run].reset_index(drop=True)


def yahoo_daily_closes(ticker: str, start: str) -> pd.DataFrame:
    """Yahoo's raw daily closes from start, as date and close columns."""
    data = download(ticker, start=start, interval="1d", auto_adjust=False)
    if data.empty:
        return pd.DataFrame(columns=["date", "close"])
    return pd.DataFrame(
        {
            "date": exchange_dates(data.index),
            "close": data["Close"].astype(float).to_numpy(),
        }
    )


def _dates(found: pd.DataFrame) -> str:
    days = [d.date().isoformat() for d in found["date"]]
    return ", ".join(days) if len(days) <= 5 else f"{days[0]} to {days[-1]}"


def check(
    tickers: list[str] | None = None, threshold: float | None = None
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """(breaks, unchecked) for the given tickers.

    breaks maps each ticker with a basis break to its mismatched closes;
    tickers without one are left out. unchecked lists the tickers Yahoo
    returned no reference prices for, which were not checked at all.
    """
    threshold = configured_threshold() if threshold is None else _checked(threshold)

    breaks = {}
    unchecked = []
    for ticker in tickers or TICKERS:
        stored = load_daily_bars(ticker)
        if stored.empty:
            continue
        reference = yahoo_daily_closes(ticker, stored["date"].min().date().isoformat())
        if reference.empty:
            logger.warning(f"{ticker}: no reference prices from Yahoo; not checked")
            unchecked.append(ticker)
            continue

        found = mismatched_closes(stored, reference, threshold)
        isolated = found[found["run"] < MIN_BREAK_RUN]
        if not isolated.empty:
            logger.warning(
                f"{ticker}: stored close differs from Yahoo's by more than the "
                f"threshold on {len(isolated)} isolated dates ({_dates(isolated)}); "
                f"a bad print on one side rather than a split, so not failing"
            )
        found = found[found["run"] >= MIN_BREAK_RUN]
        if found.empty:
            logger.info(f"{ticker}: stored prices agree with Yahoo")
        else:
            breaks[ticker] = found.reset_index(drop=True)
    return breaks, unchecked


def run(tickers: list[str] | None = None, threshold: float | None = None) -> list[str]:
    """Raise PriceBasisError, after logging each break, if any ticker has one.

    Returns the tickers that could not be checked because Yahoo returned no
    reference prices for them.
    """
    breaks, unchecked = check(tickers, threshold)
    if not breaks:
        return unchecked

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
        threshold = configured_threshold()
    except ValueError as e:
        print(f"SPLIT GUARD MISCONFIGURED: {e}", file=sys.stderr)
        return 1
    try:
        run([args.ticker.upper()] if args.ticker else None, threshold)
    except PriceBasisError as e:
        print(f"SPLIT GUARD FAILED: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
