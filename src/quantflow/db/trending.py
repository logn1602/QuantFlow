"""Reads and writes for the trending tier's tables (docs/trending.md, section 6).

The tier's storage is its own: tickers, trending_prices, trending_snapshots,
trending_fetches and trending_news (the last written by db.sentiment). No
function here touches a table the core or the evaluation reads, and no core
module reads these.

Each write that must land whole is one transaction:
  - a fetch's snapshots, each list's outcome and the selection made from
    them (save_run), so the three never disagree: a fetch stored without
    the tickers rows would count its session towards absence for symbols
    whose last_seen never moved;
  - a symbol's price replacement with its first_bar_date (replace_prices),
    so a reader finds its old bars or its new ones, never none, as the
    forecasts since PR #19.

Inserts are batched through db.batch.insert_rows, and count only rows
Postgres wrote.
"""

import datetime as dt
from collections.abc import Sequence
from typing import Any

import pandas as pd
from sqlalchemy import column, table, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from quantflow.db.batch import CHUNK_ROWS, frame_to_rows, insert_rows
from quantflow.db.connection import get_engine

# The tier's tables, as sql/migrations/002_trending.sql creates them.
TRENDING_TABLES = (
    "tickers",
    "trending_prices",
    "trending_snapshots",
    "trending_fetches",
    "trending_news",
)

SNAPSHOT_COLUMNS = (
    "snapshot_ts",
    "session_date",
    "symbol",
    "source",
    "list_name",
    "rank",
    "pct_change",
    "volume",
    "market_cap",
)
FETCH_COLUMNS = (
    "fetched_at",
    "session_date",
    "source",
    "list_name",
    "succeeded",
    "items",
    "error",
)
# Every tickers column the run writes. tier and promoted_at are not among
# them: they change by hand only, at a promotion (section 8).
TICKER_COLUMNS = (
    "symbol",
    "short_name",
    "quote_type",
    "first_seen",
    "last_seen",
    "active",
    "pinned",
    "first_bar_date",
    "first_trade_date",
    "summary_session",
    "last_close",
    "rsi_14",
    "recent_flags",
    "big_move_date",
)
# What a run may overwrite on a stored row: never its key, and never
# first_seen, which only the first insert sets.
UPDATABLE_TICKER_COLUMNS = frozenset(TICKER_COLUMNS) - {"symbol", "first_seen"}
# The columns the selection sets (4.3) and the ones the analytics set (7.4).
SELECTION_COLUMNS = (
    "short_name",
    "quote_type",
    "last_seen",
    "active",
    "pinned",
    "first_trade_date",
)
SUMMARY_COLUMNS = (
    "summary_session",
    "last_close",
    "rsi_14",
    "recent_flags",
    "big_move_date",
)
# A trending_prices row, as the run builds it (not db.prices.PRICE_COLUMNS,
# the OHLCV columns a frame of bars carries).
PRICE_ROW_COLUMNS = ("ticker", "ts", "open", "high", "low", "close", "volume")


def missing_tables() -> list[str]:
    """The tier's tables not there yet, in TRENDING_TABLES order: all of
    them until the owner runs sql/migrations/002_trending.sql."""
    query = text("""
        SELECT name
        FROM unnest(CAST(:names AS text[])) WITH ORDINALITY AS t(name, place)
        WHERE to_regclass(name) IS NULL
        ORDER BY place
    """)
    with get_engine().connect() as conn:
        return list(
            conn.execute(query, {"names": list(TRENDING_TABLES)}).scalars().all()
        )


def save_run(
    snapshot_rows: list[dict[str, Any]],
    fetch_rows: list[dict[str, Any]],
    ticker_rows: list[dict[str, Any]] | None = None,
    update: Sequence[str] = (),
) -> tuple[int, int, int]:
    """Store one run's fetch and the selection made from it, in one
    transaction: every item's snapshot row, every list's outcome and the
    tickers rows (upserted as upsert_tickers does, update the columns a
    stored row takes). Returns (snapshot rows, outcome rows, tickers rows)
    written; a snapshot or outcome already stored for the same key is
    skipped and not counted. The tickers rows are checked before anything
    is sent."""
    ticker_rows = ticker_rows or []
    _check_ticker_rows(ticker_rows, update)
    with get_engine().begin() as conn:
        snapshots = insert_rows(
            conn,
            "trending_snapshots",
            snapshot_rows,
            skip_conflicts=("snapshot_ts", "symbol", "source", "list_name"),
        )
        outcomes = insert_rows(
            conn,
            "trending_fetches",
            fetch_rows,
            skip_conflicts=("fetched_at", "source", "list_name"),
        )
        tickers = _upsert(conn, ticker_rows, update)
    return snapshots, outcomes, tickers


def _check_ticker_rows(rows: list[dict[str, Any]], update: Sequence[str]) -> None:
    """upsert_tickers' rules, checked before anything is sent."""
    if not rows:
        return
    columns = list(rows[0])
    unknown = set(columns) - set(TICKER_COLUMNS)
    if unknown:
        raise ValueError(f"not tickers columns the run writes: {sorted(unknown)}")
    if any(list(row) != columns for row in rows):
        raise ValueError("every tickers row must carry the same columns")
    refused = set(update) - UPDATABLE_TICKER_COLUMNS
    if refused or not set(update) <= set(columns):
        raise ValueError(
            f"cannot update {sorted(set(update) - set(columns) | refused)}"
        )
    symbols = [row["symbol"] for row in rows]
    if len(set(symbols)) != len(symbols):
        raise ValueError("a symbol is given twice")


def _upsert(conn, rows: list[dict[str, Any]], update: Sequence[str]) -> int:
    if not rows:
        return 0
    target = table("tickers", *(column(name) for name in rows[0]))
    written = 0
    for start in range(0, len(rows), CHUNK_ROWS):
        stmt = pg_insert(target).values(rows[start : start + CHUNK_ROWS])
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol"],
            set_={name: stmt.excluded[name] for name in update},
        )
        written += conn.execute(stmt).rowcount
    return written


def upsert_tickers(rows: list[dict[str, Any]], update: Sequence[str]) -> int:
    """Insert tickers rows, or overwrite the update columns of those already
    stored, in one transaction; returns rows inserted or updated.

    Every row carries the same columns, first_seen and last_seen among them
    (NOT NULL, checked even for a row that will update). A symbol given twice
    is refused before anything is sent: one INSERT ... ON CONFLICT DO UPDATE
    cannot change a row twice.
    """
    _check_ticker_rows(rows, update)
    if not rows:
        return 0
    with get_engine().begin() as conn:
        return _upsert(conn, rows, update)


_DELETE_PRICES = text("DELETE FROM trending_prices WHERE ticker = ANY(:tickers)")
_SET_FIRST_BAR = text(
    "UPDATE tickers SET first_bar_date = :first_bar_date WHERE symbol = :symbol"
)


def replace_prices(
    ticker: str, rows: list[dict[str, Any]], first_bar_date: dt.date
) -> int:
    """Replace the symbol's rows in trending_prices with rows, and record
    first_bar_date on its tickers row, in one transaction; returns rows
    written.

    A failed insert rolls the delete back with it, so the symbol keeps its
    previous bars; no rows is refused, so a bad answer from Yahoo can never
    leave it with none. Rows are inserted without skipping conflicts: a
    repeated date is a bug in the caller, and fails the whole replacement.
    """
    if not rows:
        raise ValueError(f"{ticker}: no bars to replace the stored ones")
    if any(row["ticker"] != ticker for row in rows):
        raise ValueError(f"{ticker}: every bar must be the symbol's own")
    with get_engine().begin() as conn:
        conn.execute(_DELETE_PRICES, {"tickers": [ticker]})
        written = insert_rows(conn, "trending_prices", rows)
        conn.execute(
            _SET_FIRST_BAR, {"first_bar_date": first_bar_date, "symbol": ticker}
        )
    return written


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """A read's rows as plain dicts, NULL as None (not NaN or NaT)."""
    if frame.empty:
        return []
    return frame_to_rows(frame, list(frame.columns))


def load_tickers(symbols: Sequence[str]) -> list[dict[str, Any]]:
    """The stored tickers rows a run needs: every active or pinned symbol,
    and any of symbols (today's candidates). Not the whole table, which
    keeps a row for every symbol ever seen."""
    query = text(f"""
        SELECT {", ".join(TICKER_COLUMNS)}
        FROM tickers
        WHERE active OR pinned OR symbol = ANY(:symbols)
        ORDER BY symbol
    """)
    with get_engine().connect() as conn:
        frame = pd.read_sql(query, conn, params={"symbols": list(symbols)})
    return _records(frame)


def counted_sessions(limit: int) -> list[dt.date]:
    """The latest session dates, newest first, at most limit of them, that
    count towards absence (4.3): those with a fetch at which every list
    answered. A fetch with a failed list counts for nobody (T8)."""
    query = text("""
        SELECT DISTINCT session_date
        FROM (
            SELECT session_date
            FROM trending_fetches
            GROUP BY fetched_at, session_date
            HAVING bool_and(succeeded)
        ) complete
        ORDER BY session_date DESC
        LIMIT :limit
    """)
    with get_engine().connect() as conn:
        frame = pd.read_sql(query, conn, params={"limit": limit})
    return [pd.Timestamp(day).date() for day in frame["session_date"]]
