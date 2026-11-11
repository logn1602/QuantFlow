"""pipelines/trending.py: one run of the trending tier (docs/trending.md, 7.1).

  1. Fetch every list; a list that answered and offered nobody is read as
     failed (rules.offering_nobody).
  2. Apply the rules and choose the active set; store every item in
     trending_snapshots, each list's outcome in trending_fetches and the
     selection in tickers, in one transaction.
  3. Prices for every active symbol: its daily bars from Yahoo replace its
     rows in trending_prices, one transaction per symbol (7.2).
  4. Indicators and anomaly flags on those same bars, by the core's own
     functions: a summary on each symbol's tickers row (7.4).
  5. Sentiment: each symbol's RSS feed, FinBERT, trending_news (7.5).
  6. Exit 1 if any list failed (T8), a write failed, or a step did nothing
     for any active symbol, once 2 to 5 have run on what the lists that
     answered returned.

Discovery and analytics only: no forecast, ensemble or backtest, and no write
to a table the core or the evaluation reads. The stages' run() functions are
never called: they read and write the core's tables, and each turns an empty
ticker list into the core watchlist (tickers or TICKERS).

Usage:
    python -m quantflow.pipelines.trending            # one run
    python -m quantflow.pipelines.trending --cap 3    # at most 3 discovered

Either writes to the database config names. --cap is a run like any other:
the active symbols past it are deactivated, until a run without it.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import socket
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quantflow import config
from quantflow.db import trending as db_trending
from quantflow.features import sentiment
from quantflow.features.trending_analytics import analyse
from quantflow.ingestion import trending_rules as rules
from quantflow.ingestion.split_guard import split_sized_jumps
from quantflow.ingestion.trending_sources import TrendingSource, sources_for
from quantflow.ingestion.yfinance_fetcher import (
    as_daily_bars,
    bars_to_rows,
    download,
    drop_unfinished_session,
)
from quantflow.utils.logger import get_logger

logger = get_logger("trending")

# A new symbol's first-sight history (section 7.2).
FIRST_SIGHT_PERIOD = "6mo"
# trending_prices' prices are NUMERIC(12, 4): a value this large cannot be
# stored (a heavily reverse-split name's adjusted history can reach it).
MAX_STORABLE_PRICE = 1e8
# Between RSS fetches, as the core's sentiment stage does.
RSS_PAUSE_SECONDS = 1.0
# Each wait for a feed's data. feedparser has no timeout of its own: it reads
# through urllib, which waits as long as the socket default allows, which is
# forever unless one is set.
RSS_TIMEOUT_SECONDS = 20


@dataclass
class RunResult:
    """What a run did, for the log and the exit code."""

    session: dt.date
    failed_lists: list[str] = field(default_factory=list)
    active: tuple[str, ...] = ()
    prices_replaced: list[str] = field(default_factory=list)
    prices_skipped: list[str] = field(default_factory=list)
    prices_failed: list[str] = field(default_factory=list)
    analysed: list[str] = field(default_factory=list)
    analytics_failed: list[str] = field(default_factory=list)
    articles: dict[str, int] = field(default_factory=dict)  # rows written
    # Headlines read and scored, by symbol, stored before or not: what shows
    # whether the feeds and the model worked at all.
    scored: dict[str, int] = field(default_factory=dict)
    sentiment_failed: list[str] = field(default_factory=list)
    news_failed: list[str] = field(default_factory=list)
    finbert_failed: bool = False

    @property
    def failed_steps(self) -> list[str]:
        """The steps that did nothing for any active symbol: no bars
        replaced, no summary made, no headline read and scored. Each
        symbol's miss is a warning, but all of them is the step failing.

        Sentiment is judged by the headlines scored, not by errors: a feed
        host answering with an error status raises nothing (feedparser
        returns no entries and fetch_rss no articles), and analyze_batch
        leaves a headline it could not score as None."""
        if not self.active:
            return []
        steps = []
        if not self.prices_replaced:
            steps.append("prices")
        elif not self.analysed:
            steps.append("analytics")
        if not self.finbert_failed and not any(
            self.scored.get(symbol) for symbol in self.active
        ):
            steps.append("sentiment")
        return steps

    @property
    def exit_code(self) -> int:
        """1 when any list failed (T8), a write failed (a price replacement,
        a symbol's news), FinBERT could not load, or a step failed for every
        active symbol; one symbol's skipped download, analytics or feed is a
        warning only (section 14)."""
        failed = (
            self.failed_lists
            or self.prices_failed
            or self.news_failed
            or self.finbert_failed
            or self.failed_steps
        )
        return 1 if failed else 0


def _utcnow() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


# ── Prices (7.2) ──────────────────────────────────────────────────────────────


def price_rows(bars: pd.DataFrame, symbol: str) -> list[dict[str, Any]]:
    """trending_prices rows from a daily frame (as_daily_bars), shaped by the
    core's fetcher (bars_to_rows: no row without a close, one per date, UTC)
    without its source column, which trending_prices does not have."""
    return [
        {column: row[column] for column in db_trending.PRICE_ROW_COLUMNS}
        for row in bars_to_rows(bars, symbol)
    ]


def _storable(rows: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        row[column] is None or abs(float(row[column])) < MAX_STORABLE_PRICE
        for row in rows
        for column in ("open", "high", "low", "close")
    )


def _first_date(rows: Sequence[Mapping[str, Any]]) -> dt.date:
    return min(pd.Timestamp(row["ts"]) for row in rows).date()


def fetch_prices(
    symbol: str,
    row: Mapping[str, Any],
    *,
    listed: dt.date | None,
    now: pd.Timestamp,
    get: Callable[..., pd.DataFrame] = download,
) -> list[dict[str, Any]] | None:
    """The symbol's daily bars from Yahoo, as trending_prices rows, or None
    when they must not replace the stored ones (logged): nothing came back,
    the download does not reach back to the symbol's first_bar_date or ends
    before its stored summary_session (a partial answer must never shorten
    a history, at either end), or a price cannot be stored. From
    first_bar_date, or six months back for a new symbol; bars before a
    listing date dropped (section 5)."""
    stored_first = row.get("first_bar_date")
    first: dt.date | None = None if rules.missing(stored_first) else stored_first
    if first is None:
        data = get(symbol, period=FIRST_SIGHT_PERIOD, interval="1d", auto_adjust=False)
    else:
        data = get(symbol, start=first.isoformat(), interval="1d", auto_adjust=False)
    data = as_daily_bars(drop_unfinished_session(data, now))
    if data.empty:
        logger.warning(f"{symbol}: Yahoo returned no daily bars; its stored bars stay")
        return None
    yahoo_first = pd.Timestamp(data.index[0]).date()
    data, dropped = rules.cut_before_listing(data, listed)
    if dropped:
        logger.info(
            f"{symbol}: dropped {dropped} bars before its listing on {listed} "
            f"(Yahoo's first bar: {yahoo_first})"
        )
    rows = price_rows(data, symbol)
    if not rows:
        logger.warning(f"{symbol}: no bars from its listing on; its stored bars stay")
        return None
    start = _first_date(rows)
    expected = first if listed is None or first is None else max(first, listed)
    if expected is not None and start > expected:
        logger.warning(
            f"{symbol}: the download starts {start}, after its first stored bar "
            f"{expected}; the stored bars stay rather than lose their start"
        )
        return None
    end = max(pd.Timestamp(r["ts"]) for r in rows).date()
    summary_session = row.get("summary_session")
    stored_end = None
    if not rules.missing(summary_session):
        stored_end = pd.Timestamp(summary_session).date()
    if stored_end is not None and end < stored_end:
        logger.warning(
            f"{symbol}: the download ends {end}, before its stored bars' "
            f"{stored_end}; the stored bars stay rather than lose their end"
        )
        return None
    if not _storable(rows):
        logger.warning(f"{symbol}: a price from Yahoo is too large to store; skipped")
        return None
    if first is None:
        logger.info(
            f"{symbol}: first stored bar {start} (Yahoo's first: {yahoo_first})"
        )
    return rows


# ── Analytics (7.4) ───────────────────────────────────────────────────────────


def summarize(
    rows: Sequence[Mapping[str, Any]], symbol: str
) -> tuple[dict[str, Any], pd.DataFrame]:
    """The tab's summary from the symbol's fresh bars (7.4): the latest
    session and close, RSI(14), the anomaly flags of the last
    rules.RECENT_DATES trading dates, and the latest big move, by the core's
    own functions on one row per date (trending_analytics.analyse, which
    the tab's chart uses too). Also returns the big moves found
    (split_sized_jumps), which warn and never block for a trending symbol
    (7.3)."""
    analysis = analyse(pd.DataFrame(list(rows)), symbol)
    daily, indicators, flags = analysis.daily, analysis.indicators, analysis.flags
    jumps = split_sized_jumps(daily[["date", "close"]])

    session = daily["date"].iloc[-1].date()
    recent = pd.Timestamp(rules.recent_start(session))
    flagged = 0
    if not flags.empty:
        flagged = int((flags.index.tz_localize(None) >= recent).sum())
    rsi = None
    if not indicators.empty and pd.notna(indicators["rsi_14"].iloc[-1]):
        rsi = round(float(indicators["rsi_14"].iloc[-1]), 2)
    summary = {
        "summary_session": session,
        "last_close": round(float(daily["close"].iloc[-1]), 4),
        "rsi_14": rsi,
        "recent_flags": flagged,
        "big_move_date": jumps["date"].max().date() if not jumps.empty else None,
    }
    return summary, jumps


# ── The run ───────────────────────────────────────────────────────────────────


def _select(
    fetches, *, core, settings, now: pd.Timestamp, session: dt.date, listing
) -> tuple[rules.Selection, dict[str, dict[str, Any]]]:
    """Step 2: choose the active set, then store the fetch and the selection
    together. Returns the selection and the tickers rows written, by
    symbol."""
    complete = bool(fetches) and all(f.succeeded for f in fetches)

    found = rules.candidates(fetches, core=core, settings=settings)
    pinned_found = rules.pinned_items(fetches, settings)
    stored = {
        row["symbol"]: row
        for row in db_trending.load_tickers(
            [c.symbol for c in found] + list(settings.pinned)
        )
    }
    counted = rules.counted_with(
        db_trending.counted_sessions(settings.inactive_after),
        session,
        complete=complete,
        limit=settings.inactive_after,
    )
    selection = rules.choose_active(
        found,
        stored,
        settings=settings,
        complete=complete,
        counted=counted,
        core=core,
    )
    rows = rules.ticker_rows(
        selection, found, pinned_found, stored, now=now, listing=listing
    )
    snapshots, outcomes, tickers = db_trending.save_run(
        rules.snapshot_rows(fetches, snapshot_ts=now, session=session),
        rules.fetch_rows(fetches, snapshot_ts=now, session=session),
        rows,
        db_trending.SELECTION_COLUMNS,
    )
    logger.info(
        f"Stored {snapshots} list items, {outcomes} list outcomes and {tickers} "
        f"tickers rows"
    )

    for symbol, item in pinned_found.items():
        if item.quote_type != "EQUITY":
            logger.warning(f"Pinned {symbol} has quote type {item.quote_type}")
    logger.info(
        f"{len(found)} candidates; active: {', '.join(selection.active) or 'none'}"
    )
    for label, symbols in (
        ("joined", selection.joined),
        ("new, history to download", selection.new),
        ("waiting for a later run (new-symbol limit)", selection.waiting),
        ("kept while absent", selection.kept_absent),
        ("deactivated", selection.deactivated),
    ):
        if symbols:
            logger.info(f"  {label}: {', '.join(symbols)}")
    if not complete:
        logger.warning(
            "A list failed: nobody deactivated, and this fetch does not count "
            "towards anyone's absence (T8)"
        )
    return selection, {row["symbol"]: row for row in rows}


def _prices_and_analytics(
    selection, rows, result: RunResult, *, listing, now, get
) -> None:
    """Steps 3 and 4, symbol by symbol: one that fails is logged and the
    rest go on."""
    summaries = []
    for symbol in selection.active:
        row = rows[symbol]
        try:
            bars = fetch_prices(
                symbol, row, listed=listing.get(symbol), now=now, get=get
            )
        except Exception as error:
            logger.warning(f"{symbol}: price fetch failed: {error}")
            bars = None
        if bars is None:
            result.prices_skipped.append(symbol)
            continue
        try:
            written = db_trending.replace_prices(symbol, bars, _first_date(bars))
        except Exception as error:
            logger.error(f"{symbol}: price replacement failed, old bars kept: {error}")
            result.prices_failed.append(symbol)
            continue
        result.prices_replaced.append(symbol)
        logger.info(f"  {symbol}: {written} daily bars replace the stored ones")
        try:
            summary, jumps = summarize(bars, symbol)
        except Exception as error:
            logger.warning(f"{symbol}: indicators or flags failed: {error}")
            result.analytics_failed.append(symbol)
            continue
        if summary["big_move_date"] is not None and summary["big_move_date"] != row.get(
            "big_move_date"
        ):
            move = jumps.iloc[-1]
            logger.warning(
                f"{symbol}: big move on {summary['big_move_date']}, close "
                f"{move['ratio']:.2f}x the day before (badged, never blocking)"
            )
        summaries.append(row | summary | {"first_bar_date": _first_date(bars)})
        result.analysed.append(symbol)
    if summaries:
        db_trending.upsert_tickers(summaries, db_trending.SUMMARY_COLUMNS)


@contextmanager
def _socket_timeout(seconds: float) -> Iterator[None]:
    """The socket default timeout set for a block, and put back after."""
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(seconds)
    try:
        yield
    finally:
        socket.setdefaulttimeout(previous)


def _news(selection, result: RunResult, *, pause: float) -> None:
    """Step 5: RSS only (D18), FinBERT loaded once, trending_news. A feed or
    the model failing for a symbol is a warning; the write failing is a
    fault, as a failed price replacement is."""
    if sentiment.get_finbert() is None:
        logger.error("FinBERT could not be loaded: no trending sentiment this run")
        result.finbert_failed = True
        return
    for symbol in selection.active:
        try:
            with _socket_timeout(RSS_TIMEOUT_SECONDS):
                articles = sentiment.fetch_rss(symbol, feeds=[rules.rss_feed(symbol)])
            scores = sentiment.analyze_batch([a["headline"] for a in articles])
            scored = [
                {**article, **score}
                for article, score in zip(articles, scores, strict=True)
                if score is not None
            ]
        except Exception as error:
            logger.warning(f"{symbol}: sentiment failed: {error}")
            result.sentiment_failed.append(symbol)
        else:
            result.scored[symbol] = len(scored)
            try:
                result.articles[symbol] = sentiment.save_sentiment(
                    scored, symbol, table="trending_news"
                )
            except Exception as error:
                logger.error(f"{symbol}: its news could not be stored: {error}")
                result.news_failed.append(symbol)
        time.sleep(pause)


def run(
    *,
    core: Sequence[str],
    settings: rules.TrendingSettings,
    sources: Sequence[TrendingSource] | None = None,
    listing: Mapping[str, dt.date] | None = None,
    now: pd.Timestamp | None = None,
    get: Callable[..., pd.DataFrame] = download,
    pause: float = RSS_PAUSE_SECONDS,
) -> RunResult:
    """One run of the tier; see the module docstring. core is the core
    watchlist (for rule 3), passed in so nothing here reads TICKERS."""
    now = _utcnow() if now is None else now
    session = rules.session_date(now)
    sources = sources_for(settings.lists) if sources is None else sources
    listing = rules.listing_dates() if listing is None else listing
    result = RunResult(session=session)
    logger.info(f"Trending run for the {session} session")

    fetches = rules.offering_nobody(
        [fetch for source in sources for fetch in source.fetch()],
        core=core,
        settings=settings,
    )
    result.failed_lists = [f.list_name for f in fetches if not f.succeeded]
    selection, rows = _select(
        fetches,
        core=core,
        settings=settings,
        now=now,
        session=session,
        listing=listing,
    )
    result.active = selection.active
    if not selection.active:
        # Never handed on: each stage reads an empty list as the core's.
        logger.info("No active symbol: prices, analytics and sentiment skipped")
    else:
        _prices_and_analytics(
            selection, rows, result, listing=listing, now=now, get=get
        )
        _news(selection, result, pause=pause)

    if result.failed_lists:
        logger.error(
            f"Lists failed: {', '.join(result.failed_lists)}; the others were "
            f"stored and used, and the run ends red (T8)"
        )
    for step in result.failed_steps:
        logger.error(f"The {step} step did nothing for any active symbol: red")
    logger.info(
        f"Done: {len(result.prices_replaced)} symbols' bars replaced, "
        f"{len(result.prices_skipped)} skipped, {len(result.prices_failed)} failed; "
        f"{sum(result.articles.values())} articles stored"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One run of the trending tier")
    parser.add_argument(
        "--cap",
        type=int,
        help="at most this many discovered symbols, for a local check "
        "(default: TRENDING_MAX_ACTIVE). Writes to the configured database "
        "like any run: the active symbols past the cap are deactivated until "
        "a run without it",
    )
    args = parser.parse_args(argv)
    if args.cap is not None and args.cap < 0:
        parser.error("--cap must be 0 or more")

    config.require_or_exit()
    try:
        settings = rules.settings(config.TICKERS)
        listing = rules.listing_dates()
    except ValueError as error:  # TrendingConfigError, or the listing file
        logger.error(f"Trending settings: {error}")
        print(f"TRENDING FAILED: {error}", file=sys.stderr)
        return 1
    if args.cap is not None:
        settings = dataclasses.replace(settings, max_active=args.cap)

    missing = db_trending.missing_tables()
    if missing:
        message = (
            f"the trending tables are not there yet ({', '.join(missing)}): "
            f"run sql/migrations/002_trending.sql first"
        )
        logger.error(message)
        print(f"TRENDING FAILED: {message}", file=sys.stderr)
        return 1

    result = run(core=config.TICKERS, settings=settings, listing=listing)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
