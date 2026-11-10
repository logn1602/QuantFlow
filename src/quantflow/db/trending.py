"""Reads and writes for the trending tier's tables (docs/trending.md, section 6).

The tier's storage is its own: tickers, trending_prices, trending_snapshots,
trending_fetches and trending_news (the last written by db.sentiment). No
function here touches a table the core or the evaluation reads, and no core
module reads these.

Each write that must land whole is one transaction:
  - a fetch's snapshots with each list's outcome (save_fetch), so the two
    never disagree;
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
PRICE_COLUMNS = ("ticker", "ts", "open", "high", "low", "close", "volume")


def save_fetch(
    snapshot_rows: list[dict[str, Any]], fetch_rows: list[dict[str, Any]]
) -> tuple[int, int]:
    """Store one fetch: every item's snapshot row and every list's outcome,
    in one transaction. Returns (snapshot rows, outcome rows) written; a row
    already stored for the same key is skipped and not counted."""
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
    return snapshots, outcomes


def upsert_tickers(rows: list[dict[str, Any]], update: Sequence[str]) -> int:
    """Insert tickers rows, or overwrite the update columns of those already
    stored, in one transaction; returns rows inserted or updated.

    Every row carries the same columns, first_seen and last_seen among them
    (NOT NULL, checked even for a row that will update). A symbol given twice
    is refused before anything is sent: one INSERT ... ON CONFLICT DO UPDATE
    cannot change a row twice.
    """
    if not rows:
        return 0
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

    target = table("tickers", *(column(name) for name in columns))
    written = 0
    with get_engine().begin() as conn:
        for start in range(0, len(rows), CHUNK_ROWS):
            stmt = pg_insert(target).values(rows[start : start + CHUNK_ROWS])
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol"],
                set_={name: stmt.excluded[name] for name in update},
            )
            written += conn.execute(stmt).rowcount
    return written


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
