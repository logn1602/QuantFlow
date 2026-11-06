"""Writer for the technical_indicators table."""

from sqlalchemy import text

from quantflow.db.batch import insert_rows
from quantflow.db.connection import get_engine

_DELETE_FOR_TICKERS = text(
    "DELETE FROM technical_indicators WHERE ticker = ANY(:tickers)"
)


def replace_indicators(ticker: str, rows: list[dict]) -> int:
    """Replace every indicator row for one ticker; return rows written.

    Why replace rather than upsert: indicators are fully recomputed from
    raw_prices on each run, so the newest computation is the whole truth, and
    an upsert leaves behind rows that no longer belong. When indicators moved
    from one row per bar to one row per trading date, every 15-min timestamp
    the old runs wrote would otherwise have stayed in the table, and the
    dashboard charts the table directly.

    The delete and the inserts share one transaction, so a failure part-way
    rolls back the delete too: the ticker keeps its previous rows rather than
    ending up with none.
    """
    with get_engine().begin() as conn:
        conn.execute(_DELETE_FOR_TICKERS, {"tickers": [ticker]})
        return insert_rows(conn, "technical_indicators", rows)
