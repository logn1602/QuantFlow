"""
ingestion/backfill.py
---------------------
One-off backfill of Yahoo's daily bars for dates that have only 15-min bars.

Every refresh between the seed and the daily-bar fix stored only 15-min bars,
so for those dates the loaders build the day from 15-min bars (see
db.prices.daily_bars), whose last close can differ from the official closing
auction. This inserts the official daily bar for each of them.

What it writes, and why it is safe to re-run:
  - only dates that have 15-min bars and no daily bar, worked out from the
    table at run time, so a second run finds nothing left to do;
  - raw prices (auto_adjust=False), the basis the 15-min bars are on;
  - inserts with ON CONFLICT DO NOTHING, so no existing row is changed;
  - one shared inserted_at, the batch timestamp printed at the end, which is
    what lets the printed undo query delete exactly the rows it wrote.

Dates with no bars at all (missed refreshes) are counted but not filled:
filling them adds trading days the models have never seen, which deserves its
own decision.

Usage:
    python -m quantflow.ingestion.backfill --dry-run
    python -m quantflow.ingestion.backfill
    python -m quantflow.ingestion.backfill --ticker AAPL --dry-run
"""

import argparse
import datetime as dt
import sys
from dataclasses import dataclass, field

import pandas as pd

from quantflow.config import TICKERS
from quantflow.db.prices import classify_bars, insert_bars, load_bar_timestamps
from quantflow.ingestion.yfinance_fetcher import (
    bars_to_rows,
    download,
    drop_unfinished_session,
)
from quantflow.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TickerPlan:
    """What a backfill would write for one ticker."""

    ticker: str
    rows: list[dict] = field(default_factory=list)
    # Dates that need a daily bar but Yahoo returned none for.
    no_yahoo_bar: list[pd.Timestamp] = field(default_factory=list)
    # Yahoo trading dates in the window with no stored bar of either kind.
    gaps: list[pd.Timestamp] = field(default_factory=list)

    def date_range(self) -> str:
        if not self.rows:
            return "nothing"
        days = sorted(pd.Timestamp(r["ts"]).date() for r in self.rows)
        return f"{days[0]} to {days[-1]}"


def dates_needing_daily_bars(stored_ts: pd.DataFrame) -> list[pd.Timestamp]:
    """Trading dates that have 15-min bars but no daily bar."""
    if stored_ts.empty:
        return []
    classified = classify_bars(stored_ts)
    per_date = classified.groupby("date")["is_intraday"].agg(["all"])
    return list(per_date.index[per_date["all"]])


def plan_ticker(
    ticker: str, stored_ts: pd.DataFrame, yahoo_daily: pd.DataFrame
) -> TickerPlan:
    """Rows to write: Yahoo's daily bar for each date that needs one."""
    plan = TickerPlan(ticker)
    wanted = set(dates_needing_daily_bars(stored_ts))
    if not wanted:
        return plan

    yahoo_dates = pd.DatetimeIndex(yahoo_daily.index).normalize()
    keep = yahoo_dates.isin(list(wanted))
    plan.rows = bars_to_rows(yahoo_daily[keep], ticker)
    plan.no_yahoo_bar = sorted(wanted - set(yahoo_dates))

    stored_dates = set(classify_bars(stored_ts)["date"])
    plan.gaps = sorted(
        d for d in yahoo_dates if d >= min(wanted) and d not in stored_dates
    )
    return plan


def undo_sql(batch: dt.datetime) -> str:
    """The query that deletes exactly the rows one backfill batch wrote.

    Every row a batch writes carries the batch's inserted_at, set by this
    module rather than left to DEFAULT NOW(), so matching that exact
    microsecond timestamp selects that batch and nothing else. The midnight
    condition is a second fence: the backfill only ever writes daily bars.
    """
    literal = f"TIMESTAMPTZ '{batch.isoformat(sep=' ')}'"
    where = (
        "WHERE source = 'yfinance'\n"
        f"  AND inserted_at = {literal}\n"
        "  AND (ts AT TIME ZONE 'UTC')::time = TIME '00:00'"
    )
    return (
        "-- 1. Preview: the rows this batch wrote, per ticker.\n"
        "SELECT ticker, count(*) AS rows, min(ts) AS first_ts, max(ts) AS last_ts\n"
        f"FROM raw_prices\n{where}\n"
        "GROUP BY ticker ORDER BY ticker;\n\n"
        "-- 2. Undo: delete exactly those rows.\n"
        f"DELETE FROM raw_prices\n{where};"
    )


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def run(
    tickers: list[str] | None = None,
    dry_run: bool = False,
    now: pd.Timestamp | None = None,
) -> dict[str, int]:
    """Plan, and unless dry_run, write the backfill. Returns rows per ticker:
    written, or for a dry run, the rows that would be written."""
    tickers = tickers or TICKERS
    now = now or pd.Timestamp.now(tz="UTC")
    batch = now.floor("us").to_pydatetime()

    print(
        "Backfill plan (dry run, nothing written)"
        if dry_run
        else f"Backfill batch {batch.isoformat(sep=' ')}"
    )
    counts: dict[str, int] = {}
    for ticker in tickers:
        stored = load_bar_timestamps(ticker)
        wanted = dates_needing_daily_bars(stored)
        if not wanted:
            counts[ticker] = 0
            print(f"  {ticker:6s} nothing to backfill")
            continue

        yahoo = download(
            ticker,
            start=min(wanted).date().isoformat(),
            interval="1d",
            auto_adjust=False,
        )
        plan = plan_ticker(ticker, stored, drop_unfinished_session(yahoo, now))

        if dry_run:
            counts[ticker] = len(plan.rows)
            verb = "to insert"
        else:
            rows = [{**row, "inserted_at": batch} for row in plan.rows]
            counts[ticker] = insert_bars(rows)
            verb = "written"
        print(
            f"  {ticker:6s} {_plural(counts[ticker], 'row')} {verb}, "
            f"{plan.date_range()}; "
            f"{_plural(len(plan.no_yahoo_bar), 'date')} Yahoo has no daily bar for; "
            f"{_plural(len(plan.gaps), 'trading date')} with no stored bars "
            "(not filled)"
        )

    total = sum(counts.values())
    print(f"Total: {_plural(total, 'row')} {'to insert' if dry_run else 'written'}.")
    if not dry_run and total:
        print("\nTo undo exactly these rows, run:\n")
        print(undo_sql(batch))
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill Yahoo daily bars for dates with only 15-min bars"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan, write nothing"
    )
    parser.add_argument("--ticker", help="one ticker instead of config.TICKERS")
    args = parser.parse_args(argv)

    run([args.ticker.upper()] if args.ticker else None, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
