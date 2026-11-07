"""tests/integration/test_writes_on_postgres.py

The write-path claims the unit tests can only fake, checked on real Postgres:

  - a multi-row INSERT ... ON CONFLICT DO NOTHING reports only the rows it
    wrote, across chunks, which is what every writer's count now relies on;
  - the backfill's undo query deletes exactly the rows one batch wrote;
  - replacing a ticker's indicators rolls back the delete if an insert fails;
  - the daily-bar rule works on the types the real driver returns.

Needs a disposable local Postgres, never the live database:

    uv run pytest -m integration --pg-url postgresql://user@127.0.0.1:5432/scratch

Each test gets its own schema, created from sql/schema*.sql and dropped after.
"""

import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DataError

from quantflow.db import prices
from quantflow.db.batch import insert_rows
from quantflow.db.indicators import replace_indicators
from quantflow.ingestion import backfill, yfinance_fetcher

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def pg(request, install_engine):
    url = request.config.getoption("--pg-url")
    if not url:
        pytest.skip("needs --pg-url pointing at a disposable local Postgres")
    if make_url(url).host not in ("localhost", "127.0.0.1"):
        pytest.fail("--pg-url must be a local database, never the live one")

    schema = f"qf_test_{uuid.uuid4().hex[:8]}"
    admin = create_engine(url)
    with admin.begin() as conn:
        conn.exec_driver_sql(f"CREATE SCHEMA {schema}")
    engine = create_engine(url, connect_args={"options": f"-csearch_path={schema}"})
    with engine.begin() as conn:
        for name in ("schema.sql", "schema_sentiment.sql"):
            conn.exec_driver_sql((ROOT / "sql" / name).read_text(encoding="utf-8"))
    install_engine(engine)

    yield engine

    engine.dispose()
    with admin.begin() as conn:
        conn.exec_driver_sql(f"DROP SCHEMA {schema} CASCADE")
    admin.dispose()


def _bars(days, ticker="AAPL"):
    return [
        {
            "ticker": ticker,
            "source": "yfinance",
            "ts": pd.Timestamp(day, tz="UTC"),
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1,
        }
        for day in days
    ]


def _count(engine, sql, **params):
    with engine.connect() as conn:
        return conn.execute(text(sql), params).scalar_one()


def test_on_conflict_skips_are_not_counted_across_chunks(pg):
    days = pd.bdate_range("2026-01-05", periods=10)
    assert prices.insert_bars(_bars(days[:4])) == 4

    with pg.begin() as conn:
        written = insert_rows(
            conn,
            "raw_prices",
            _bars(days),
            skip_conflicts=("ticker", "source", "ts"),
            chunk_rows=3,
        )
    assert written == 6
    assert _count(pg, "SELECT count(*) FROM raw_prices") == 10


def test_a_key_repeated_inside_one_insert_is_written_once(pg):
    """What the sentiment writer's own de-duplication guards against."""
    row = _bars(["2026-01-05"])
    with pg.begin() as conn:
        written = insert_rows(
            conn, "raw_prices", row + row, skip_conflicts=("ticker", "source", "ts")
        )
    assert written == 1


def test_the_backfill_undo_deletes_exactly_the_rows_it_wrote(pg, monkeypatch, capsys):
    # A seeded daily bar, two dates with only 15-min bars, and a daily bar
    # some other run wrote with DEFAULT NOW().
    session = [
        pd.Timestamp(f"{day} 09:30", tz="America/New_York").tz_convert("UTC")
        for day in ("2026-04-13", "2026-04-14")
    ]
    rows = _bars(["2026-04-10"]) + [
        {**_bars(["2026-04-10"])[0], "ts": ts} for ts in session
    ]
    prices.insert_bars(rows + _bars(["2026-04-15"]))
    before = _count(pg, "SELECT count(*) FROM raw_prices")

    days = pd.bdate_range("2026-04-13", "2026-04-15")
    yahoo = pd.DataFrame(
        {c: np.full(3, 200.0) for c in ("Open", "High", "Low", "Close", "Adj Close")}
        | {"Volume": np.full(3, 5e6)},
        index=days,
    )
    monkeypatch.setattr(yfinance_fetcher.yf, "download", lambda t, **k: yahoo.copy())

    now = pd.Timestamp("2026-11-09 18:00:00.123456", tz="UTC")
    assert backfill.run(["AAPL"], now=now) == {"AAPL": 2}
    assert _count(pg, "SELECT count(*) FROM raw_prices") == before + 2

    undo = backfill.undo_sql(now.to_pydatetime())
    delete = undo[undo.index("DELETE FROM") :]
    with pg.begin() as conn:
        deleted = conn.execute(text(delete)).rowcount
    assert deleted == 2
    assert _count(pg, "SELECT count(*) FROM raw_prices") == before


def test_a_failed_indicator_replace_keeps_the_previous_rows(pg):
    ts = pd.Timestamp("2026-04-10", tz="UTC")
    columns = ("rsi_14", "macd", "macd_signal", "macd_hist", "bb_upper", "bb_middle")
    good = {"ticker": "AAPL", "ts": ts} | dict.fromkeys(columns, 1.0)
    good["bb_lower"] = 1.0
    assert replace_indicators("AAPL", [good]) == 1

    overflow = good | {"ts": ts + pd.Timedelta(days=1), "rsi_14": 1e9}  # NUMERIC(8,4)
    with pytest.raises(DataError):
        replace_indicators("AAPL", [good | {"rsi_14": 2.0}, overflow])

    assert _count(pg, "SELECT count(*) FROM technical_indicators") == 1
    assert _count(pg, "SELECT rsi_14 FROM technical_indicators") == 1.0


def test_two_runs_replacing_one_tickers_indicators_at_once_both_succeed(pg):
    """The second run's DELETE waits on the first and cannot see the rows the
    first commits; its INSERT must skip them rather than fail the refresh."""
    import threading
    import time

    ts = pd.Timestamp("2026-04-10", tz="UTC")
    columns = ("rsi_14", "macd", "macd_signal", "macd_hist", "bb_upper", "bb_middle")
    row = {"ticker": "AAPL", "ts": ts} | dict.fromkeys(columns, 1.0)
    row["bb_lower"] = 1.0
    assert replace_indicators("AAPL", [row]) == 1

    first = pg.connect()
    first_txn = first.begin()
    first.execute(
        text("DELETE FROM technical_indicators WHERE ticker = ANY(:t)"),
        {"t": ["AAPL"]},
    )
    insert_rows(first, "technical_indicators", [row | {"rsi_14": 2.0}])

    outcome = {}

    def second_run():
        try:
            outcome["written"] = replace_indicators("AAPL", [row | {"rsi_14": 3.0}])
        except Exception as e:
            outcome["error"] = e

    worker = threading.Thread(target=second_run)
    worker.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:  # until the second DELETE is blocked
        if _count(pg, "SELECT count(*) FROM pg_locks WHERE NOT granted"):
            break
        time.sleep(0.05)
    else:
        pytest.fail("the second run never blocked on the first")

    first_txn.commit()
    first.close()
    worker.join(timeout=10)

    assert "error" not in outcome, outcome.get("error")
    assert outcome["written"] == 0
    assert _count(pg, "SELECT rsi_14 FROM technical_indicators") == 2.0


def test_daily_bars_on_the_types_the_driver_returns(pg):
    session = [
        {
            **_bars(["2026-04-13"])[0],
            "ts": pd.Timestamp(f"2026-04-13 {t}", tz="America/New_York"),
            "close": close,
        }
        for t, close in (("09:30", 100.0), ("15:45", 102.5))
    ]
    prices.insert_bars(_bars(["2026-04-10"]) + session)

    daily = prices.load_daily_close("AAPL")
    assert list(daily["ds"]) == [pd.Timestamp("2026-04-10"), pd.Timestamp("2026-04-13")]
    assert list(daily["y"]) == [1.0, 102.5]
