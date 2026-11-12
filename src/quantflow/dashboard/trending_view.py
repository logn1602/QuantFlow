"""dashboard/trending_view.py: the Trending tab's reads and shaping
(docs/trending.md, section 10).

Apart from app.py so it can be tested: app.py runs Streamlit as it is
imported (Known problem 5), and this module does not. Every read takes an
open connection, so a test hands it a fake one; app.py wraps each in
caching.cached, like every loader. The tables read are the tier's own, and
nothing here writes.

A table that is not there yet (before the owner runs 002_trending.sql) reads
as an empty frame, as an empty table does, and the tab shows its empty state
instead of an error. Dates are the stored DATE columns or UTC instants, never
a timestamp cast in the session's time zone (decision K).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from quantflow.db.prices import PRICE_COLUMNS
from quantflow.features.trending_analytics import WINDOW_BARS, analyse
from quantflow.ingestion import trending_rules as rules

CAPTION = "Discovery only: not forecast or backtested."
EMPTY = "No trending data yet."
# Shown when a trending setting cannot be read: the tab is public, so the
# setting's raw value goes to the log, not to every visitor.
SETTINGS_WARNING = "The trending settings cannot be read, so this tab is off for now."

# Postgres' code for a table that does not exist.
UNDEFINED_TABLE = "42P01"

# One grouped query for the whole table, not one per row: each active symbol
# with its summary from the last run, its place on the latest fetch's lists,
# its days on the lists and its news of the last 7 days.
TABLE_QUERY = text("""
    WITH latest AS (
        SELECT max(snapshot_ts) AS snapshot_ts FROM trending_snapshots
    ),
    listed AS (
        SELECT s.symbol,
               min(s.rank) AS best_rank,
               array_agg(DISTINCT s.list_name) AS lists,
               (array_agg(s.pct_change ORDER BY s.rank, s.list_name))[1] AS pct_change,
               (array_agg(s.volume ORDER BY s.rank, s.list_name))[1] AS volume
        FROM trending_snapshots s
        JOIN latest ON s.snapshot_ts = latest.snapshot_ts
        GROUP BY s.symbol
    ),
    days AS (
        SELECT symbol, count(DISTINCT session_date) AS days_trending
        FROM trending_snapshots
        WHERE symbol IN (SELECT symbol FROM tickers WHERE active)
        GROUP BY symbol
    ),
    news AS (
        SELECT ticker AS symbol, avg(compound) AS sentiment_7d, count(*) AS articles_7d
        FROM trending_news
        WHERE published_at >= now() - interval '7 days'
          AND ticker IN (SELECT symbol FROM tickers WHERE active)
        GROUP BY ticker
    )
    SELECT t.symbol, t.short_name, t.pinned, t.first_trade_date,
           t.summary_session, t.last_close, t.rsi_14, t.recent_flags,
           t.big_move_date,
           listed.best_rank, listed.lists, listed.pct_change, listed.volume,
           coalesce(days.days_trending, 0) AS days_trending,
           news.sentiment_7d, coalesce(news.articles_7d, 0) AS articles_7d
    FROM tickers t
    LEFT JOIN listed ON listed.symbol = t.symbol
    LEFT JOIN days ON days.symbol = t.symbol
    LEFT JOIN news ON news.symbol = t.symbol
    WHERE t.active
    ORDER BY listed.best_rank NULLS LAST, t.symbol
""")

# Each list's last success, last attempt, and whether that attempt answered.
LISTS_QUERY = text("""
    SELECT list_name,
           max(fetched_at) FILTER (WHERE succeeded) AS last_success,
           max(fetched_at) AS last_fetch,
           (array_agg(succeeded ORDER BY fetched_at DESC))[1] AS latest_succeeded
    FROM trending_fetches
    GROUP BY list_name
""")

# The chart's bars: the window the run's summary analyses too
# (trending_analytics.WINDOW_BARS, about a year), so the table's flag count
# and the chart's flags agree, and no more is read: a symbol's history grows a
# bar a day for as long as it is tracked.
CHART_BARS = WINDOW_BARS

# The latest CHART_BARS bars, oldest first, as the candles are drawn.
BARS_QUERY = text("""
    SELECT ts, open, high, low, close, volume
    FROM (
        SELECT ts, open, high, low, close, volume
        FROM trending_prices
        WHERE ticker = :symbol
        ORDER BY ts DESC LIMIT :limit
    ) latest
    ORDER BY ts
""")


def _read(conn, query, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """A read, or an empty frame when its table is not there yet. Any other
    error is raised: the tab must not hide a real fault as "no data"."""
    try:
        return pd.read_sql(query, conn, params=params or {})
    except ProgrammingError as error:
        if getattr(error.orig, "pgcode", None) != UNDEFINED_TABLE:
            raise
        conn.rollback()
        return pd.DataFrame()


def load_table(conn) -> pd.DataFrame:
    """The active symbols' rows for the tab's table (TABLE_QUERY)."""
    return _read(conn, TABLE_QUERY)


def load_list_status(conn) -> pd.DataFrame:
    """Each list's last success and latest outcome (LISTS_QUERY)."""
    return _read(conn, LISTS_QUERY)


def load_bars(conn, symbol: str) -> pd.DataFrame:
    """One symbol's latest CHART_BARS stored daily bars, for its chart."""
    return _read(conn, BARS_QUERY, {"symbol": symbol, "limit": CHART_BARS})


def load_chart(
    conn, symbol: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[()]:
    """One symbol's chart frames (chart_frames on load_bars), or () with no
    bars: app.py keeps the frames an hour, computed once rather than on
    every rerun, and caching.cached keeps no empty result."""
    frames = chart_frames(load_bars(conn, symbol), symbol)
    return () if frames[0].empty else frames


# ── Shaping ───────────────────────────────────────────────────────────────────


def _when(value: Any) -> str:
    return pd.Timestamp(value).tz_convert("UTC").strftime("%Y-%m-%d %H:%M UTC")


def list_lines(status: pd.DataFrame, lists: Sequence[str]) -> list[str]:
    """One short line per list in TRENDING_LISTS, in its order: when it last
    succeeded, "never succeeded" or "never fetched", and "failed since" on a
    list whose latest fetch failed (T8)."""
    rows = {} if status.empty else {r["list_name"]: r for _, r in status.iterrows()}
    lines = []
    for name in lists:
        row = rows.get(name)
        if row is None:
            lines.append(f"{name}: never fetched")
        elif pd.isna(row["last_success"]):
            lines.append(
                f"{name}: never succeeded (last tried {_when(row['last_fetch'])})"
            )
        elif not row["latest_succeeded"]:
            lines.append(
                f"{name}: last succeeded {_when(row['last_success'])}; failed since"
            )
        else:
            lines.append(f"{name}: last succeeded {_when(row['last_success'])}")
    return lines


def list_caption(lines: Sequence[str]) -> str:
    """The lines as one caption, a line each: joined with Markdown's hard
    line break, so a list's entry never wraps into the next one's (ISSUE-001,
    /qa 2026-11-10). The lines hold list names and times only."""
    return "  \n".join(lines)


def _date(value: Any) -> dt.date | None:
    """A stored date as a date, None for NULL. pd.isna first: NaT is itself
    an instance of datetime.date."""
    if rules.missing(value):
        return None
    return pd.Timestamp(value).date()


def badges(row: pd.Series, *, settings: rules.TrendingSettings, today: dt.date) -> str:
    """The row's badges, as words: pinned, promotion-eligible (section 8),
    and a big move in the last rules.RECENT_DATES trading dates (7.3)."""
    words = []
    pinned = bool(row.get("pinned"))
    if pinned:
        words.append("pinned")
    session = _date(row.get("summary_session")) or today
    if rules.promotion_eligible(
        days_trending=int(row.get("days_trending") or 0),
        pinned=pinned,
        first_trade_date=_date(row.get("first_trade_date")),
        as_of=session,
        settings=settings,
    ):
        words.append("promotion-eligible")
    moved = _date(row.get("big_move_date"))
    if moved is not None and moved >= rules.recent_start(session):
        words.append(f"big move {moved.isoformat()}")
    return ", ".join(words)


# Symbol and Badges first, and pinned there by app.py (table_column_config):
# with Badges last of fourteen, a 1,400-pixel window showed about nine
# columns, and the badges were cut off on the right (2026-11-11). Pinned, the
# two stay in view while the other twelve scroll.
PINNED_COLUMNS = ("Symbol", "Badges")
TABLE_COLUMNS = [
    *PINNED_COLUMNS,
    "Name",
    "Best rank",
    "Lists",
    "% change",
    "Volume",
    "Days trending",
    "Sentiment (7d)",
    "Articles (7d)",
    f"Flags ({rules.RECENT_DATES}d)",
    "RSI (14)",
    "Last close",
    "Session",
]


def table_column_config() -> dict[str, Any]:
    """st.dataframe's column_config for the table: PINNED_COLUMNS pinned at
    the left, and Badges 200 pixels wide ("medium"). Left to itself, the
    grid sizes a column to its first rows, which mostly carry no badge, and
    cut "big move 2026-10-30" to "big move"; at 200 pixels two badges fit.
    Streamlit is imported here, not at the top, so the module stays
    importable without it."""
    import streamlit as st

    return {
        "Symbol": st.column_config.Column(pinned=True),
        "Badges": st.column_config.Column(pinned=True, width="medium"),
    }


def _number(value: Any, digits: int) -> float | None:
    return None if rules.missing(value) else round(float(value), digits)


def _whole(value: Any) -> int | None:
    return None if rules.missing(value) else int(value)


def table_rows(
    raw: pd.DataFrame, *, settings: rules.TrendingSettings, today: dt.date
) -> pd.DataFrame:
    """The tab's table from TABLE_QUERY's rows: one per active symbol, its
    lists in TRENDING_LISTS order, numbers rounded for reading, badges as
    words. Values only; app.py shows the frame as text, never as HTML."""
    if raw.empty:
        return pd.DataFrame(columns=TABLE_COLUMNS)
    priority = rules.list_priority(settings.lists)
    rows = []
    for _, row in raw.iterrows():
        lists = row.get("lists")
        named = sorted(lists, key=priority) if lists else []
        rows.append(
            {
                "Symbol": row["symbol"],
                "Name": row.get("short_name") or "",
                "Best rank": _whole(row.get("best_rank")),
                "Lists": ", ".join(named),
                "% change": _number(row.get("pct_change"), 2),
                "Volume": _whole(row.get("volume")),
                "Days trending": int(row.get("days_trending") or 0),
                "Sentiment (7d)": _number(row.get("sentiment_7d"), 3),
                "Articles (7d)": int(row.get("articles_7d") or 0),
                f"Flags ({rules.RECENT_DATES}d)": _whole(row.get("recent_flags")),
                "RSI (14)": _number(row.get("rsi_14"), 1),
                "Last close": _number(row.get("last_close"), 2),
                "Session": _date(row.get("summary_session")),
                "Badges": badges(row, settings=settings, today=today),
            }
        )
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def chart_frames(
    bars: pd.DataFrame, symbol: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A symbol's stored bars as app.py's price_chart and rsi_macd_chart
    take them: one row per date, with its indicators and anomaly flags
    computed as the run computes them (trending_analytics.analyse), each
    with a date column."""
    empty = pd.DataFrame(columns=["date"])
    if bars.empty:
        return pd.DataFrame(columns=["date", *PRICE_COLUMNS]), empty, empty
    analysis = analyse(bars, symbol)

    def dated(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return empty
        index = pd.DatetimeIndex(frame.index).tz_convert("UTC").tz_localize(None)
        return frame.reset_index(drop=True).assign(date=index)

    return analysis.daily, dated(analysis.indicators), dated(analysis.flags)
