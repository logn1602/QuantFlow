"""
ingestion/backfill.py
---------------------
One-off backfill of Yahoo's daily bars for every NYSE trading day that has
none, from each ticker's first stored date to the last completed session.

Two kinds of date lack one. Every refresh between the seed and the daily-bar
fix stored only 15-min bars, so for those dates the loaders build the day from
15-min bars (see db.prices.daily_bars), whose last close can differ from the
official closing auction. And some trading days have no bars at all: between
the seed's last daily bar and the first 15-min refresh, or while the daily
refresh was not running. The models read the days either side of such a gap
as consecutive. This inserts the official daily bar for both kinds.

What it writes, and why it is safe to re-run:
  - only trading days without a daily bar, from the first stored date to the
    last session whose daily bar is final (trading_calendar's
    last_completed_session), worked out from the table at run time, so a
    second run finds nothing left to do;
  - raw prices (auto_adjust=False), the basis the 15-min bars are on;
  - inserts with ON CONFLICT DO NOTHING, so no existing row is changed;
  - one shared inserted_at, the batch timestamp printed at the end, which is
    what lets the printed undo query delete exactly the rows it wrote.

The dry run lists every trading day with no bars that it would fill. A day
the calendar expects but Yahoo has no bar for, such as a one-off closure the
calendar cannot know about (the 2025-01-09 national day of mourning), is
listed and left empty.

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
from quantflow.utils.trading_calendar import (
    TRADING_DAY,
    exchange_dates,
    last_completed_session,
)


class BackfillError(RuntimeError):
    """Yahoo returned no daily bars for at least one ticker that needs them."""


@dataclass
class TickerPlan:
    """What a backfill would write for one ticker."""

    ticker: str
    rows: list[dict] = field(default_factory=list)
    # Dates filled that had only 15-min bars.
    only_15min: list[pd.Timestamp] = field(default_factory=list)
    # Trading days filled that had no bars at all.
    no_bars: list[pd.Timestamp] = field(default_factory=list)
    # Dates that need a daily bar but Yahoo returned none for.
    no_yahoo_bar: list[pd.Timestamp] = field(default_factory=list)

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


def trading_days_without_bars(
    stored_ts: pd.DataFrame, last_session: pd.Timestamp
) -> list[pd.Timestamp]:
    """NYSE trading days from the first stored date to last_session that have
    neither a daily bar nor a 15-min bar. A date holding only stray rows off
    the session grid counts as empty: the loaders leave those rows out."""
    if stored_ts.empty:
        return []
    classified = classify_bars(stored_ts)
    usable = set(
        classified.loc[classified["is_daily"] | classified["is_intraday"], "date"]
    )
    expected = pd.date_range(classified["date"].min(), last_session, freq=TRADING_DAY)
    return [d for d in expected if d not in usable]


def wanted_dates(
    stored_ts: pd.DataFrame, last_session: pd.Timestamp
) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    """(dates with only 15-min bars, trading days with no bars), both up to
    last_session: every date the backfill should give a daily bar."""
    only_15min = [d for d in dates_needing_daily_bars(stored_ts) if d <= last_session]
    return only_15min, trading_days_without_bars(stored_ts, last_session)


def plan_ticker(
    ticker: str,
    stored_ts: pd.DataFrame,
    yahoo_daily: pd.DataFrame,
    last_session: pd.Timestamp,
) -> TickerPlan:
    """Rows to write: Yahoo's daily bar for each date that needs one, up to
    last_session.

    yahoo_daily is a yf.download frame. Its dates are read as exchange dates
    whatever time zone its index carries, and the rows are stamped 00:00 UTC,
    where daily bars are stored.
    """
    plan = TickerPlan(ticker)
    only_15min, no_bars = wanted_dates(stored_ts, last_session)
    wanted = set(only_15min) | set(no_bars)
    if not wanted:
        return plan
    if yahoo_daily.empty:
        plan.no_yahoo_bar = sorted(wanted)
        return plan

    yahoo_dates = exchange_dates(yahoo_daily.index)
    keep = yahoo_dates.isin(list(wanted))
    plan.rows = bars_to_rows(as_daily_bars(yahoo_daily)[keep], ticker)
    available = set(yahoo_dates)
    plan.only_15min = [d for d in only_15min if d in available]
    plan.no_bars = [d for d in no_bars if d in available]
    plan.no_yahoo_bar = sorted(wanted - available)
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


def _date_runs(dates: list[pd.Timestamp]) -> str:
    """dates as a list of runs of consecutive trading days, e.g.
    "2026-04-09 to 2026-04-14, 2026-04-16"."""
    runs: list[list[pd.Timestamp]] = []
    for day in sorted(dates):
        if runs and day == runs[-1][-1] + TRADING_DAY:
            runs[-1].append(day)
        else:
            runs.append([day])
    return ", ".join(
        f"{run[0].date()}" if len(run) == 1 else f"{run[0].date()} to {run[-1].date()}"
        for run in runs
    )


def _report(ticker: str, plan: TickerPlan, n: int, verb: str) -> None:
    def counted(dates: list[pd.Timestamp], what: str, rest: str) -> str:
        text = f"{_plural(len(dates), what)} {rest}"
        return f"{text}: {_date_runs(dates)}" if dates else text

    indent = " " * 11
    print(f"  {ticker:6s} {_plural(n, 'row')} {verb}, {plan.date_range()}")
    print(indent + f"{_plural(len(plan.only_15min), 'date')} with only 15-min bars")
    print(indent + counted(plan.no_bars, "trading day", "with no bars"))
    print(
        indent + counted(plan.no_yahoo_bar, "date", "Yahoo has no final daily bar for")
    )


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
    last_session = last_completed_session(now)

    print(
        "Backfill plan (dry run, nothing written)"
        if dry_run
        else f"Backfill batch {batch.isoformat(sep=' ')}"
    )
    print(f"Trading days up to the last completed session, {last_session.date()}")
    counts: dict[str, int] = {}
    no_download: list[str] = []
    # Each ticker's insert commits on its own, so a failure part-way leaves
    # earlier tickers written. The undo query is printed in `finally` so it
    # reaches the operator whether or not the run completed.
    try:
        for ticker in tickers:
            stored = load_bar_timestamps(ticker)
            if stored.empty:
                counts[ticker] = 0
                print(f"  {ticker:6s} no stored bars; seed it first")
                continue
            only_15min, no_bars = wanted_dates(stored, last_session)
            wanted = only_15min + no_bars
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
                # nothing for trading days it should have. Either way this
                # ticker was not backfilled, which must not read as done.
                counts[ticker] = 0
                no_download.append(ticker)
                print(f"  {ticker:6s} no daily bars from Yahoo; nothing written")
                continue
            plan = plan_ticker(
                ticker, stored, drop_unfinished_session(yahoo, now), last_session
            )

            if dry_run:
                counts[ticker] = len(plan.rows)
                verb = "to insert"
            else:
                rows = [{**row, "inserted_at": batch} for row in plan.rows]
                counts[ticker] = insert_bars(rows)
                verb = "written"
            _report(ticker, plan, counts[ticker], verb)
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
        description="Backfill Yahoo daily bars for trading days that have none"
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
