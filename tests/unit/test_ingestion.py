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


# ── Raw prices and finished sessions ─────────────────────────────────────────


@pytest.fixture
def downloads(monkeypatch):
    """Capture yf.download calls; serve a frame chosen by interval."""
    calls = []
    frames = {}

    def fake_download(ticker, **kwargs):
        calls.append({"ticker": ticker, **kwargs})
        return frames.get(kwargs.get("interval"), pd.DataFrame()).copy()

    monkeypatch.setattr(yfinance_fetcher.yf, "download", fake_download)
    return calls, frames


def _daily_frame(days):
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in days])  # naive, as yfinance
    close = 100.0 + np.arange(len(days), dtype=float)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Adj Close": close * 0.99,
            "Volume": np.full(len(days), 5e6),
        },
        index=index,
    )


def test_both_fetches_request_raw_prices_not_dividend_adjusted(fake_db, downloads):
    """15-min bars are never dividend-adjusted by Yahoo, so adjusted daily bars
    would sit on a different basis from the 15-min bars of the same date, and
    a row's value would depend on when it was fetched."""
    calls, _ = downloads
    yfinance_fetcher.fetch_historical(["AAPL"], period="5d")
    yfinance_fetcher.fetch_intraday(["AAPL"])

    assert [c["auto_adjust"] for c in calls] == [False, False]


def test_a_daily_bar_from_a_session_still_trading_is_not_stored(
    fake_db, downloads, monkeypatch
):
    """With ON CONFLICT DO NOTHING a mid-session snapshot, once stored, would
    never be replaced by the final bar."""
    _, frames = downloads
    frames["1d"] = _daily_frame(["2026-11-05", "2026-11-06"])
    monkeypatch.setattr(
        yfinance_fetcher,
        "_now",
        lambda: pd.Timestamp("2026-11-06 15:00", tz="America/New_York"),
    )

    yfinance_fetcher.fetch_historical(["AAPL"], period="5d")

    stored = [r["ts"] for r in fake_db.tables["raw_prices"]]
    assert stored == [pd.Timestamp("2026-11-05", tz="UTC")]


def test_todays_daily_bar_is_stored_once_the_session_is_final(
    fake_db, downloads, monkeypatch
):
    _, frames = downloads
    frames["1d"] = _daily_frame(["2026-11-05", "2026-11-06"])
    monkeypatch.setattr(
        yfinance_fetcher,
        "_now",
        lambda: pd.Timestamp("2026-11-06 17:30", tz="America/New_York"),
    )

    assert yfinance_fetcher.fetch_historical(["AAPL"], period="5d") == {"AAPL": 2}


def test_daily_bars_are_stored_at_utc_midnight(fake_db, downloads, monkeypatch):
    _, frames = downloads
    frames["1d"] = _daily_frame(["2026-11-05"])
    monkeypatch.setattr(
        yfinance_fetcher, "_now", lambda: pd.Timestamp("2026-11-06 18:00", tz="UTC")
    )
    yfinance_fetcher.fetch_historical(["AAPL"], period="5d")

    (row,) = fake_db.tables["raw_prices"]
    assert row["ts"] == pd.Timestamp("2026-11-05 00:00", tz="UTC")
    assert row["close"] == 100.0  # the raw close, not Adj Close


def test_cli_daily_fetches_five_days_of_daily_bars(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        yfinance_fetcher,
        "fetch_historical",
        lambda tickers=None, period="2y": seen.setdefault("period", period) and {},
    )
    assert yfinance_fetcher.main(["--daily"]) == 0
    assert seen["period"] == "5d"


def test_cli_intraday_fetches_15min_bars(monkeypatch):
    seen = []
    monkeypatch.setattr(
        yfinance_fetcher, "fetch_intraday", lambda tickers=None: seen.append(1) or {}
    )
    assert yfinance_fetcher.main(["--intraday"]) == 0
    assert seen == [1]


# ── Alpha Vantage ─────────────────────────────────────────────────────────────


def test_alpha_vantage_counts_only_rows_the_database_wrote(fake_db):
    frame = _av_frame(4)
    fake_db.seed("raw_prices", _stored("alpha_vantage", frame["ts"][:1]))

    assert alpha_vantage_fetcher._upsert_prices(frame, "AAPL") == 3


def test_alpha_vantage_write_error_raises_instead_of_being_logged(fake_db):
    fake_db.fail_next_insert = DataError("INSERT", {}, Exception("value too long"))
    with pytest.raises(DataError):
        alpha_vantage_fetcher._upsert_prices(_av_frame(2), "AAPL")
