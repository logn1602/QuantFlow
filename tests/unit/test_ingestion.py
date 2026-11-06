"""tests/unit/test_ingestion.py

Covers what the price fetchers write to raw_prices and what they report.

Both fetchers used to count every row they attempted, including rows that
ON CONFLICT DO NOTHING skipped, and wrapped each INSERT in its own try/except
inside one transaction. On Postgres the first failed INSERT aborts that
transaction, so every later row fails too and the batch rolls back while the
returned count still claims success. These tests hold the fetchers to the
count Postgres reports, and to raising on any other write error.
"""

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.exc import DataError

from quantflow.ingestion import alpha_vantage_fetcher, yfinance_fetcher


def _yf_frame(n, start="2026-11-02 09:30"):
    """n 15-min bars shaped like yf.download output: capitalised columns and a
    tz-aware New York index."""
    index = pd.date_range(start, periods=n, freq="15min", tz="America/New_York")
    close = 100.0 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "Open": close - 0.1,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Adj Close": close,
            "Volume": np.full(n, 1000.0),
        },
        index=index,
    )


def _av_frame(n):
    ts = pd.date_range("2026-11-02 09:30", periods=n, freq="15min", tz="US/Eastern")
    close = 100.0 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "ts": ts,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1000, dtype=int),
        }
    )


def _stored(source, timestamps):
    return [
        {"ticker": "AAPL", "source": source, "ts": pd.Timestamp(t).tz_convert("UTC")}
        for t in timestamps
    ]


# ── yfinance ──────────────────────────────────────────────────────────────────


def test_yfinance_counts_only_rows_the_database_wrote(fake_db):
    frame = _yf_frame(5)
    fake_db.seed("raw_prices", _stored("yfinance", frame.index[:3]))

    assert yfinance_fetcher._upsert_prices(frame, "AAPL") == 2


def test_yfinance_writes_one_batched_insert_not_one_per_row(fake_db):
    yfinance_fetcher._upsert_prices(_yf_frame(26), "AAPL")
    assert len(fake_db.inserts("raw_prices")) == 1


def test_yfinance_write_error_raises_instead_of_being_logged(fake_db):
    fake_db.fail_next_insert = DataError("INSERT", {}, Exception("bigint out of range"))
    with pytest.raises(DataError):
        yfinance_fetcher._upsert_prices(_yf_frame(3), "AAPL")


def test_yfinance_missing_volume_is_stored_as_null_not_nan(fake_db):
    """NaN bound into a BIGINT column is a write error; None is a NULL."""
    frame = _yf_frame(2)
    frame.loc[frame.index[1], "Volume"] = np.nan
    yfinance_fetcher._upsert_prices(frame, "AAPL")

    volumes = [row["volume"] for row in fake_db.tables["raw_prices"]]
    assert volumes[1] is None


def test_yfinance_rows_without_a_close_are_not_written(fake_db):
    frame = _yf_frame(3)
    frame.loc[frame.index[0], "Close"] = np.nan
    assert yfinance_fetcher._upsert_prices(frame, "AAPL") == 2


# ── Alpha Vantage ─────────────────────────────────────────────────────────────


def test_alpha_vantage_counts_only_rows_the_database_wrote(fake_db):
    frame = _av_frame(4)
    fake_db.seed("raw_prices", _stored("alpha_vantage", frame["ts"][:1]))

    assert alpha_vantage_fetcher._upsert_prices(frame, "AAPL") == 3


def test_alpha_vantage_write_error_raises_instead_of_being_logged(fake_db):
    fake_db.fail_next_insert = DataError("INSERT", {}, Exception("value too long"))
    with pytest.raises(DataError):
        alpha_vantage_fetcher._upsert_prices(_av_frame(2), "AAPL")
