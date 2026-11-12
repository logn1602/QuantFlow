"""Batched INSERTs that report what Postgres actually wrote.

Every write path in the project used to INSERT one row at a time, count each
attempt as a write, and catch errors per row. Two things were wrong with that.
Rows skipped by ON CONFLICT DO NOTHING were counted as inserted. And on
Postgres the first failed INSERT aborts the surrounding transaction, so every
later row fails too and the commit rolls the whole batch back, while the
returned count still claims success.

This module sends one multi-row INSERT per chunk and sums the rowcount
Postgres returns for each statement, which counts only rows actually written.
It deliberately does not use executemany: for a Core insert on psycopg2,
SQLAlchemy 2.0 turns executemany into its insertmanyvalues batching, whose
rowcount is the number of rows *sent* per batch, so a conflict skip would be
counted as a write again. Errors other than a skipped conflict propagate.
"""

from collections.abc import Sequence
from typing import Any

import pandas as pd
from sqlalchemy import column, table
from sqlalchemy.dialects.postgresql import insert as pg_insert

# Postgres allows 65,535 bind parameters per statement. 1,000 rows of the
# widest rows written in chunks of it (tickers', 14 columns: 14,000
# parameters) stays well inside that.
CHUNK_ROWS = 1000


def frame_to_rows(df: pd.DataFrame, columns: Sequence[str]) -> list[dict[str, Any]]:
    """Plain row dicts ready to bind: native Python values, NaN as None.

    NaN must not reach the driver. Bound into a BIGINT it is a write error,
    and bound into a NUMERIC it is stored as the value NaN rather than NULL.
    """
    subset = df.loc[:, list(columns)]
    subset = subset.astype(object).where(subset.notna(), None)
    return subset.to_dict(orient="records")


def insert_rows(
    conn,
    table_name: str,
    rows: list[dict[str, Any]],
    *,
    skip_conflicts: Sequence[str] | bool = False,
    chunk_rows: int = CHUNK_ROWS,
) -> int:
    """INSERT rows in chunks on an open connection; return rows written.

    skip_conflicts:
      False           a plain INSERT; a duplicate key raises.
      True            ON CONFLICT DO NOTHING against any unique constraint.
      column names    ON CONFLICT (columns) DO NOTHING.

    Skipped rows are expected and are not counted. Every other error raises,
    and because the caller owns the transaction, the whole write rolls back.
    """
    if not rows:
        return 0

    columns = list(rows[0])
    target = table(table_name, *(column(name) for name in columns))
    written = 0
    for start in range(0, len(rows), chunk_rows):
        stmt = pg_insert(target).values(rows[start : start + chunk_rows])
        if skip_conflicts is True:
            stmt = stmt.on_conflict_do_nothing()
        elif skip_conflicts:
            stmt = stmt.on_conflict_do_nothing(index_elements=list(skip_conflicts))
        written += conn.execute(stmt).rowcount
    return written
