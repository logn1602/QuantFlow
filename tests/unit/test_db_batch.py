"""tests/unit/test_db_batch.py

Covers quantflow.db.batch, the one INSERT path every writer now shares.

The count it returns must be what Postgres wrote, not what was sent. That is
why it issues one multi-row statement per chunk instead of executemany: on
psycopg2, SQLAlchemy 2.0's insertmanyvalues batching reports rows sent per
batch, which would count ON CONFLICT skips as writes again.
"""

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.exc import IntegrityError

from quantflow.db.batch import frame_to_rows, insert_rows


def _rows(n, ticker="AAPL"):
    base = pd.Timestamp("2026-01-02", tz="UTC")
    return [
        {"ticker": ticker, "source": "yfinance", "ts": base + pd.Timedelta(days=i)}
        for i in range(n)
    ]


def test_large_writes_are_split_into_chunks(fake_db):
    with_engine = _engine(fake_db)
    with with_engine.begin() as conn:
        written = insert_rows(
            conn, "raw_prices", _rows(2500), skip_conflicts=True, chunk_rows=1000
        )

    assert written == 2500
    assert len(fake_db.inserts("raw_prices")) == 3


def test_the_count_is_rows_written_across_every_chunk(fake_db):
    rows = _rows(25)
    fake_db.seed("raw_prices", rows[::5])  # one stored row in every chunk of 5
    with _engine(fake_db).begin() as conn:
        written = insert_rows(
            conn,
            "raw_prices",
            rows,
            skip_conflicts=("ticker", "source", "ts"),
            chunk_rows=5,
        )
    assert written == 20


def test_named_conflict_columns_become_the_conflict_target(fake_db):
    with _engine(fake_db).begin() as conn:
        insert_rows(
            conn, "raw_prices", _rows(1), skip_conflicts=("ticker", "source", "ts")
        )
    sql, _ = fake_db.inserts("raw_prices")[0]
    assert "ON CONFLICT (ticker, source, ts) DO NOTHING" in sql


def test_skip_conflicts_true_skips_any_unique_violation(fake_db):
    with _engine(fake_db).begin() as conn:
        insert_rows(conn, "forecasts", [{"ticker": "AAPL"}], skip_conflicts=True)
    sql, _ = fake_db.inserts("forecasts")[0]
    assert sql.rstrip().endswith("ON CONFLICT DO NOTHING")


def test_a_plain_insert_raises_on_a_duplicate_and_rolls_back(fake_db):
    rows = _rows(3)
    fake_db.seed("raw_prices", rows[2:])
    with pytest.raises(IntegrityError), _engine(fake_db).begin() as conn:
        insert_rows(conn, "raw_prices", rows)

    assert fake_db.rollbacks == 1
    assert len(fake_db.tables["raw_prices"]) == 1  # only the seeded row


def test_no_rows_means_no_statement(fake_db):
    with _engine(fake_db).begin() as conn:
        assert insert_rows(conn, "raw_prices", []) == 0
    assert fake_db.statements == []


def test_frame_to_rows_binds_nan_as_none_and_numpy_as_python():
    df = pd.DataFrame(
        {"close": [1.5, np.nan], "volume": [np.int64(7), np.int64(8)], "x": ["a", "b"]}
    )
    rows = frame_to_rows(df, ["close", "volume"])

    assert rows[0] == {"close": 1.5, "volume": 7}
    assert rows[1]["close"] is None
    assert type(rows[0]["volume"]) is int


def _engine(fake_db):
    from quantflow.db import connection

    return connection.get_engine()
