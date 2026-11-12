"""tests/unit/test_db_trending.py

db/trending.py on the fake database (tests/conftest.py): what each write
reports against what it wrote, the price replacement's transaction, and the
shape of the reads. The SQL itself runs on real Postgres in
tests/integration/test_trending_on_postgres.py.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.exc import IntegrityError

from quantflow.db import trending

pytestmark = pytest.mark.trending

T0 = pd.Timestamp("2026-11-10 23:10", tz="UTC")
SESSION = dt.date(2026, 11, 10)


def _snapshot(symbol, list_name="most_actives", rank=1, source="yahoo_screener"):
    return {
        "snapshot_ts": T0,
        "session_date": SESSION,
        "symbol": symbol,
        "source": source,
        "list_name": list_name,
        "rank": rank,
        "pct_change": 1.5,
        "volume": 1000,
        "market_cap": None,
    }


def _outcome(list_name, succeeded=True, items=1, source="yahoo_screener"):
    return {
        "fetched_at": T0,
        "session_date": SESSION,
        "source": source,
        "list_name": list_name,
        "succeeded": succeeded,
        "items": items,
        "error": None if succeeded else "HTTPError: 503",
    }


def _ticker(symbol, **columns):
    row = dict.fromkeys(trending.TICKER_COLUMNS)
    row.update(symbol=symbol, first_seen=T0, last_seen=T0, active=True, pinned=False)
    row.update(columns)
    return row


def _bars(ticker, days, close=10.0):
    return [
        {
            "ticker": ticker,
            "ts": pd.Timestamp(day, tz="UTC"),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 100,
        }
        for day in days
    ]


# ── save_run ──────────────────────────────────────────────────────────────────


def test_a_fetch_stores_its_snapshots_and_outcomes_and_counts_them(fake_db):
    snapshots = [_snapshot("SPCX"), _snapshot("HIMS", rank=2)]
    outcomes = [_outcome("most_actives", items=2), _outcome("day_gainers", False, 0)]
    assert trending.save_run(snapshots, outcomes) == (2, 2, 0)
    assert len(fake_db.tables["trending_snapshots"]) == 2
    assert [r["succeeded"] for r in fake_db.tables["trending_fetches"]] == [True, False]
    # One batched insert per table.
    assert len(fake_db.inserts("trending_snapshots")) == 1
    assert len(fake_db.inserts("trending_fetches")) == 1


def test_a_snapshot_already_stored_is_skipped_and_not_counted(fake_db):
    fake_db.seed("trending_snapshots", [_snapshot("SPCX")])
    written = trending.save_run(
        [_snapshot("SPCX"), _snapshot("HIMS")], [_outcome("most_actives")]
    )
    assert written == (1, 1, 0)


def test_a_failed_outcome_insert_rolls_the_snapshots_back(fake_db, monkeypatch):
    """One transaction: the snapshots never land without the outcomes that
    say which lists answered."""
    calls = []
    real = trending.insert_rows

    def failing(conn, table_name, rows, **kwargs):
        calls.append(table_name)
        if table_name == "trending_fetches":
            raise IntegrityError("INSERT", {}, Exception("boom"))
        return real(conn, table_name, rows, **kwargs)

    monkeypatch.setattr(trending, "insert_rows", failing)
    with pytest.raises(IntegrityError):
        trending.save_run([_snapshot("SPCX")], [_outcome("most_actives")])
    assert calls == ["trending_snapshots", "trending_fetches"]
    assert fake_db.tables["trending_snapshots"] == []
    assert fake_db.rollbacks == 1


def test_a_run_stores_its_fetch_and_its_tickers_rows_together(fake_db):
    written = trending.save_run(
        [_snapshot("SPCX")],
        [_outcome("most_actives")],
        [_ticker("SPCX"), _ticker("HIMS")],
        trending.SELECTION_COLUMNS,
    )
    assert written == (1, 1, 2)
    assert {r["symbol"] for r in fake_db.tables["tickers"]} == {"SPCX", "HIMS"}


def test_a_failed_tickers_write_rolls_the_fetch_back(fake_db, monkeypatch):
    """A fetch stored without the selection made from it would count its
    session towards absence for symbols whose last_seen never moved: one
    transaction, so neither lands without the other."""
    real = fake_db.execute

    def failing(statement, params=None):
        if str(statement).lstrip().startswith("INSERT INTO tickers"):
            raise IntegrityError("INSERT", {}, Exception("boom"))
        return real(statement, params)

    monkeypatch.setattr(fake_db, "execute", failing)
    with pytest.raises(IntegrityError):
        trending.save_run(
            [_snapshot("SPCX")],
            [_outcome("most_actives")],
            [_ticker("SPCX")],
            trending.SELECTION_COLUMNS,
        )
    assert fake_db.tables["trending_snapshots"] == []
    assert fake_db.tables["trending_fetches"] == []
    assert fake_db.rollbacks == 1


def test_tickers_rows_are_checked_before_the_fetch_is_sent(fake_db):
    with pytest.raises(ValueError, match="given twice"):
        trending.save_run(
            [_snapshot("SPCX")],
            [_outcome("most_actives")],
            [_ticker("SPCX"), _ticker("SPCX")],
            trending.SELECTION_COLUMNS,
        )
    assert fake_db.statements == []


# ── upsert_tickers ────────────────────────────────────────────────────────────


def test_new_tickers_are_inserted_and_stored_ones_updated(fake_db):
    earlier = T0 - pd.Timedelta(days=3)
    fake_db.seed(
        "tickers",
        [_ticker("SPCX", first_seen=earlier, last_seen=earlier, active=False)],
    )
    rows = [
        _ticker("SPCX", short_name="SpaceX", pinned=True),
        _ticker("HIMS", short_name="Hims"),
    ]
    assert trending.upsert_tickers(rows, trending.SELECTION_COLUMNS) == 2

    stored = {r["symbol"]: r for r in fake_db.tables["tickers"]}
    assert stored["SPCX"]["active"] is True and stored["SPCX"]["pinned"] is True
    assert stored["SPCX"]["last_seen"] == T0
    # first_seen is the first insert's, whatever the row says.
    assert stored["SPCX"]["first_seen"] == earlier
    assert stored["HIMS"]["first_seen"] == T0
    assert len(fake_db.inserts("tickers")) == 1


def test_an_upsert_updates_only_the_columns_named(fake_db):
    fake_db.seed("tickers", [_ticker("SPCX", short_name="SpaceX", rsi_14=40.0)])
    row = _ticker("SPCX", short_name="changed", rsi_14=55.5, last_close=141.3)
    trending.upsert_tickers([row], trending.SUMMARY_COLUMNS)
    (stored,) = fake_db.tables["tickers"]
    assert stored["rsi_14"] == 55.5 and stored["last_close"] == 141.3
    assert stored["short_name"] == "SpaceX"


@pytest.mark.parametrize(
    ("rows", "update", "message"),
    [
        ([_ticker("A"), _ticker("A")], ("active",), "given twice"),
        ([_ticker("A")], ("first_seen",), "cannot update"),
        ([_ticker("A")], ("symbol",), "cannot update"),
        ([_ticker("A")], ("tier",), "cannot update"),
        ([{"symbol": "A", "promoted_at": T0}], (), "not tickers columns"),
        ([_ticker("A"), {"symbol": "B", "active": True}], ("active",), "same columns"),
        ([{"symbol": "A", "first_seen": T0}], ("active",), "cannot update"),
    ],
)
def test_an_upsert_the_table_could_not_take_is_refused_before_sending(
    fake_db, rows, update, message
):
    with pytest.raises(ValueError, match=message):
        trending.upsert_tickers(rows, update)
    assert fake_db.statements == []


def test_no_rows_is_no_statement(fake_db):
    assert trending.upsert_tickers([], trending.SELECTION_COLUMNS) == 0
    assert fake_db.statements == []


# ── replace_prices ────────────────────────────────────────────────────────────


def test_a_replacement_deletes_only_its_symbols_bars_and_sets_first_bar_date(
    fake_db,
):
    fake_db.seed("trending_prices", _bars("SPCX", ["2026-11-05", "2026-11-06"]))
    fake_db.seed("trending_prices", _bars("HIMS", ["2026-11-06"]))
    new = _bars("SPCX", ["2026-11-06", "2026-11-09", "2026-11-10"], close=12.0)

    assert trending.replace_prices("SPCX", new, dt.date(2026, 11, 6)) == 3

    stored = fake_db.tables["trending_prices"]
    assert sorted((r["ticker"], r["close"]) for r in stored) == [
        ("HIMS", 10.0),
        ("SPCX", 12.0),
        ("SPCX", 12.0),
        ("SPCX", 12.0),
    ]
    ((_, delete_params),) = fake_db.deletes("trending_prices")
    assert delete_params == [{"tickers": ["SPCX"]}]
    updates = [(s, p) for s, p in fake_db.statements if s.lstrip().startswith("UPDATE")]
    assert updates == [
        (
            "UPDATE tickers SET first_bar_date = :first_bar_date WHERE symbol = :symbol",
            [{"first_bar_date": dt.date(2026, 11, 6), "symbol": "SPCX"}],
        )
    ]


def test_a_failed_replacement_keeps_the_previous_bars(fake_db):
    old = _bars("SPCX", ["2026-11-05", "2026-11-06"])
    fake_db.seed("trending_prices", old)
    fake_db.fail_next_insert = IntegrityError("INSERT", {}, Exception("boom"))
    with pytest.raises(IntegrityError):
        trending.replace_prices(
            "SPCX", _bars("SPCX", ["2026-11-10"]), dt.date(2026, 11, 10)
        )
    assert fake_db.tables["trending_prices"] == old


def test_a_repeated_date_fails_the_whole_replacement(fake_db):
    """No conflict skipping: a caller that sends one date twice has a bug,
    and the symbol keeps its stored bars."""
    old = _bars("SPCX", ["2026-11-05"])
    fake_db.seed("trending_prices", old)
    twice = _bars("SPCX", ["2026-11-10", "2026-11-10"])
    with pytest.raises(IntegrityError):
        trending.replace_prices("SPCX", twice, dt.date(2026, 11, 10))
    assert fake_db.tables["trending_prices"] == old


def test_a_replacement_with_no_bars_is_refused(fake_db):
    with pytest.raises(ValueError, match="no bars"):
        trending.replace_prices("SPCX", [], dt.date(2026, 11, 10))
    assert fake_db.statements == []


def test_a_replacement_with_another_symbols_bars_is_refused(fake_db):
    with pytest.raises(ValueError, match="the symbol's own"):
        trending.replace_prices(
            "SPCX", _bars("HIMS", ["2026-11-10"]), dt.date(2026, 11, 10)
        )
    assert fake_db.statements == []


# ── Reads ─────────────────────────────────────────────────────────────────────


def test_load_tickers_reads_the_run_rows_with_null_as_none(read_tables):
    seen = {}

    def frame(params):
        seen.update(params)
        return pd.DataFrame(
            {
                "symbol": ["HIMS", "SPCX"],
                "short_name": ["Hims", None],
                "quote_type": ["EQUITY", "EQUITY"],
                "first_seen": [T0, T0],
                "last_seen": [T0, T0],
                "active": [True, True],
                "pinned": [False, True],
                "first_bar_date": [dt.date(2026, 5, 11), None],
                "first_trade_date": [None, dt.date(2026, 6, 12)],
                "summary_session": [None, None],
                "last_close": [np.nan, 141.3],
                "rsi_14": [np.nan, np.nan],
                "recent_flags": [np.nan, 2.0],
                "big_move_date": [None, None],
            }
        )

    read_tables["tickers"] = frame
    rows = trending.load_tickers(("SPCX", "HIMS"))
    assert seen == {"symbols": ["SPCX", "HIMS"]}
    assert [r["symbol"] for r in rows] == ["HIMS", "SPCX"]
    assert rows[0]["last_close"] is None and rows[0]["rsi_14"] is None
    assert rows[1]["short_name"] is None and rows[1]["first_bar_date"] is None
    assert rows[1]["first_trade_date"] == dt.date(2026, 6, 12)


def test_load_tickers_with_nothing_stored_is_empty(read_tables):
    read_tables["tickers"] = pd.DataFrame(columns=list(trending.TICKER_COLUMNS))
    assert trending.load_tickers([]) == []


def test_counted_sessions_come_back_as_dates_newest_first(read_tables):
    seen = {}

    def frame(params):
        seen.update(params)
        return pd.DataFrame(
            {"session_date": [dt.date(2026, 11, 10), dt.date(2026, 11, 6)]}
        )

    read_tables["trending_fetches"] = frame
    assert trending.counted_sessions(5) == [dt.date(2026, 11, 10), dt.date(2026, 11, 6)]
    assert seen == {"limit": 5}


def test_tickers_rows_with_no_update_columns_are_inserted_only(fake_db):
    """An empty update passed the checks but built ON CONFLICT DO UPDATE
    with an empty SET, which SQLAlchemy refuses: upsert_tickers raised, and
    save_run, whose default it is, raised inside its transaction after
    sending the fetch (found in the full-diff review, 2026-11-12). A stored
    row is now left as it is."""
    assert trending.upsert_tickers([_ticker("SPCX")], ()) == 1
    assert trending.upsert_tickers([_ticker("SPCX", short_name="Renamed")], ()) == 0
    assert fake_db.tables["tickers"][0]["short_name"] is None
    stored = trending.save_run(
        [_snapshot("HIMS")], [_outcome("most_actives")], [_ticker("HIMS")]
    )
    assert stored == (1, 1, 1)
