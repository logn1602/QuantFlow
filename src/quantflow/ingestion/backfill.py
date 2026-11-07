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

A ticker Yahoo returns no daily bars for (a failed download, logged by
download()) is reported, the other tickers are still filled, and the run
exits 1, so a Yahoo outage cannot pass for a backfill with nothing to do.
Running again fills what is left.

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
    as_daily_bars,
    bars_to_rows,
    download,
    drop_unfinished_session,
)
from quantflow.utils.trading_calendar import exchange_dates


class BackfillError(RuntimeError):
    """Yahoo returned no daily bars for at least one ticker that needs them."""


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
    """Trading dates that have 15-min bars but no daily bar.

    Asked per date as "any 15-min bar, no daily bar" rather than "every row is
    a 15-min bar": a stray row off the session grid is neither, and must not
    make a date look already filled.
    """
    if stored_ts.empty:
        return []
    per_date = (
        classify_bars(stored_ts)
        .groupby("date")
        .agg(has_daily=("is_daily", "any"), has_intraday=("is_intraday", "any"))
    )
    needs = per_date["has_intraday"] & ~per_date["has_daily"]
    return list(per_date.index[needs])


def plan_ticker(
    ticker: str, stored_ts: pd.DataFrame, yahoo_daily: pd.DataFrame
) -> TickerPlan:
    """Rows to write: Yahoo's daily bar for each date that needs one.

    yahoo_daily is a yf.download frame. Its dates are read as exchange dates
    whatever time zone its index carries, and the rows are stamped 00:00 UTC,
    where daily bars are stored.
    """
    plan = TickerPlan(ticker)
    wanted = set(dates_needing_daily_bars(stored_ts))
    if not wanted:
        return plan
    if yahoo_daily.empty:
        plan.no_yahoo_bar = sorted(wanted)
        return plan

    yahoo_dates = exchange_dates(yahoo_daily.index)
    keep = yahoo_dates.isin(list(wanted))
    plan.rows = bars_to_rows(as_daily_bars(yahoo_daily)[keep], ticker)
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
    no_download: list[str] = []
    # Each ticker's insert commits on its own, so a failure part-way leaves
    # earlier tickers written. The undo query is printed in `finally` so it
    # reaches the operator whether or not the run completed.
    try:
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
            if yahoo.empty:
                # A failed download (download() logs why) or Yahoo serving
                # nothing for dates it has 15-min bars for. Either way this
                # ticker was not backfilled, which must not read as done.
                counts[ticker] = 0
                no_download.append(ticker)
                print(f"  {ticker:6s} no daily bars from Yahoo; nothing written")
                continue
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
                f"{_plural(len(plan.no_yahoo_bar), 'date')} Yahoo has no final "
                "daily bar for; "
                f"{_plural(len(plan.gaps), 'trading date')} with no stored bars "
                "(not filled)"
            )
        if no_download:
            raise BackfillError(
                "no daily bars from Yahoo for "
                + ", ".join(no_download)
                + "; run again for those tickers"
            )
    finally:
        total = sum(counts.values())
        print(
            f"Total: {_plural(total, 'row')} {'to insert' if dry_run else 'written'}."
        )
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

    try:
        run([args.ticker.upper()] if args.ticker else None, dry_run=args.dry_run)
    except BackfillError as e:
        print(f"BACKFILL INCOMPLETE: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
