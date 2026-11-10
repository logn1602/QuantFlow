"""tests/integration/test_trending_on_postgres.py

The trending tier's storage on real Postgres (docs/trending.md, sections 6
and 7.2), what the unit suite can only fake:

  - the migration runs twice, creating the five tables and nothing else, and
    agrees with sql/schema_trending.sql;
  - a role granted SELECT by default privileges, as the read-only role is
    (sql/schema_roles.sql, step 4), reads every new table and writes none;
  - trending_news has news_sentiment's columns and key;
  - a fetch, the tickers upsert and the reads on the types the driver
    returns;
  - each symbol's price replacement is one transaction: a failed insert
    keeps its bars and first_bar_date, and a second connection polling
    during replacements always finds the symbol's bars.

Needs a disposable local Postgres, never the live database:

    uv run pytest -m integration --pg-url postgresql://user@127.0.0.1:5432/scratch
"""

import datetime as dt
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DataError, IntegrityError, ProgrammingError

from quantflow.db import trending

pytestmark = [pytest.mark.integration, pytest.mark.trending]

ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "sql" / "migrations" / "002_trending.sql"
SCHEMA = ROOT / "sql" / "schema_trending.sql"
TABLES = (
    "tickers",
    "trending_fetches",
    "trending_news",
    "trending_prices",
    "trending_snapshots",
)
T0 = pd.Timestamp("2026-11-10 23:10", tz="UTC")


@contextmanager
def _fresh_schema(pg_url, schema_engine):
    """An empty schema of its own and an engine on it, dropped after."""
    schema = f"qf_fresh_{uuid.uuid4().hex[:8]}"
    admin = create_engine(pg_url)
    with admin.begin() as conn:
        conn.exec_driver_sql(f"CREATE SCHEMA {schema}")
    engine = schema_engine(pg_url, schema)
    try:
        yield engine, schema
    finally:
        engine.dispose()
        with admin.begin() as conn:
            conn.exec_driver_sql(f"DROP SCHEMA {schema} CASCADE")
        admin.dispose()


def _run_script(engine, path):
    """A SQL file as psql or Supabase's editor runs it: one script, its own
    BEGIN and COMMIT, on a connection outside any SQLAlchemy transaction."""
    raw = engine.raw_connection()
    try:
        raw.driver_connection.autocommit = True
        with raw.cursor() as cursor:
            cursor.execute(path.read_text(encoding="utf-8"))
    finally:
        raw.invalidate()  # the connection's autocommit stays out of the pool


def _tables(engine, schema):
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = :s ORDER BY table_name"
            ),
            {"s": schema},
        )
        return tuple(row[0] for row in rows)


def _columns(engine, schema, table_name):
    with engine.connect() as conn:
        return pd.read_sql(
            text("""
                SELECT column_name, data_type, is_nullable,
                       character_maximum_length, numeric_precision, numeric_scale
                FROM information_schema.columns
                WHERE table_schema = :s AND table_name = :t
                ORDER BY ordinal_position
            """),
            conn,
            params={"s": schema, "t": table_name},
        )


def _unique_keys(engine, schema, table_name):
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT con.contype, pg_get_constraintdef(con.oid)
                FROM pg_constraint con
                JOIN pg_class rel ON rel.oid = con.conrelid
                JOIN pg_namespace ns ON ns.oid = rel.relnamespace
                WHERE ns.nspname = :s AND rel.relname = :t
                  AND con.contype IN ('p', 'u')
                ORDER BY con.contype
            """),
            {"s": schema, "t": table_name},
        )
        return [tuple(row) for row in rows]


# ── The migration ─────────────────────────────────────────────────────────────


def test_the_migration_runs_twice_and_creates_the_five_tables(pg_url, schema_engine):
    with _fresh_schema(pg_url, schema_engine) as (engine, schema):
        _run_script(engine, MIGRATION)
        assert _tables(engine, schema) == TABLES
        first = {t: _columns(engine, schema, t) for t in TABLES}

        _run_script(engine, MIGRATION)  # idempotent: nothing changes
        _run_script(engine, SCHEMA)  # make setup's file: the same tables
        assert _tables(engine, schema) == TABLES
        for name in TABLES:
            pd.testing.assert_frame_equal(_columns(engine, schema, name), first[name])


def test_a_role_with_default_select_reads_every_new_table_and_writes_none(
    pg_url, schema_engine
):
    """The read-only role reads tables postgres creates after it through its
    default privileges (sql/schema_roles.sql, step 4), so the dashboard needs
    no grant after the migration. Modelled here on the test schema."""
    role = f"qf_ro_{uuid.uuid4().hex[:8]}"
    with _fresh_schema(pg_url, schema_engine) as (engine, schema):
        with engine.begin() as conn:
            conn.exec_driver_sql(f"CREATE ROLE {role} NOLOGIN")
            conn.exec_driver_sql(f"GRANT USAGE ON SCHEMA {schema} TO {role}")
            conn.exec_driver_sql(
                f"ALTER DEFAULT PRIVILEGES IN SCHEMA {schema} "
                f"GRANT SELECT ON TABLES TO {role}"
            )
        try:
            _run_script(engine, MIGRATION)
            with engine.connect() as conn:
                conn.exec_driver_sql(f"SET ROLE {role}")
                for name in TABLES:
                    count = conn.exec_driver_sql(f"SELECT count(*) FROM {name}")
                    assert count.scalar_one() == 0
                with pytest.raises(ProgrammingError, match="permission denied"):
                    conn.exec_driver_sql(
                        "INSERT INTO trending_fetches VALUES "
                        "(now(), current_date, 's', 'l', true, 0, null)"
                    )
                conn.rollback()
        finally:
            with engine.begin() as conn:
                conn.exec_driver_sql(
                    f"ALTER DEFAULT PRIVILEGES IN SCHEMA {schema} "
                    f"REVOKE SELECT ON TABLES FROM {role}"
                )
                conn.exec_driver_sql(f"DROP OWNED BY {role}")
                conn.exec_driver_sql(f"DROP ROLE {role}")


def test_trending_news_has_news_sentiments_columns_and_key(pg):
    with pg.connect() as conn:
        schema = conn.exec_driver_sql("SELECT current_schema()").scalar_one()
    news = _columns(pg, schema, "news_sentiment")
    twin = _columns(pg, schema, "trending_news")
    pd.testing.assert_frame_equal(twin, news)
    assert _unique_keys(pg, schema, "trending_news") == [
        ("p", "PRIMARY KEY (id)"),
        ("u", "UNIQUE (ticker, headline, published_at)"),
    ]
    assert _unique_keys(pg, schema, "news_sentiment") == _unique_keys(
        pg, schema, "trending_news"
    )


def test_trending_news_draws_ids_from_its_own_sequence(pg):
    """LIKE ... INCLUDING DEFAULTS would have shared news_sentiment's."""
    with pg.connect() as conn:
        default = conn.exec_driver_sql(
            "SELECT column_default FROM information_schema.columns "
            "WHERE table_schema = current_schema() "
            "AND table_name = 'trending_news' AND column_name = 'id'"
        ).scalar_one()
    assert "trending_news_id_seq" in default


# ── Writes and reads on real types ────────────────────────────────────────────


def _snapshot(symbol, list_name, rank, **columns):
    return {
        "snapshot_ts": T0,
        "session_date": dt.date(2026, 11, 10),
        "symbol": symbol,
        "source": "yahoo_screener",
        "list_name": list_name,
        "rank": rank,
        "pct_change": -6.486535,
        "volume": 345697762,
        "market_cap": 4034492891136,
    } | columns


def _outcome(fetched_at, session, list_name, succeeded):
    return {
        "fetched_at": fetched_at,
        "session_date": session,
        "source": "yahoo_screener",
        "list_name": list_name,
        "succeeded": succeeded,
        "items": 25 if succeeded else 0,
        "error": None if succeeded else "HTTPError: 503 Server Error",
    }


def test_a_fetch_is_stored_on_real_types_and_counted_once(pg):
    snapshots = [
        _snapshot("NVDA", "most_actives", 1),
        _snapshot("SPCX", "most_actives", 3, market_cap=None, pct_change=None),
        _snapshot("NVDA", "trending_tickers", 1, source="yahoo_trending"),
    ]
    outcomes = [
        _outcome(T0, dt.date(2026, 11, 10), "most_actives", True),
        _outcome(T0, dt.date(2026, 11, 10), "day_gainers", False),
    ]
    assert trending.save_fetch(snapshots, outcomes) == (3, 2)
    assert trending.save_fetch(snapshots, outcomes) == (0, 0)
    with pg.connect() as conn:
        stored = pd.read_sql(
            text("SELECT * FROM trending_snapshots ORDER BY list_name, rank"), conn
        )
    assert float(stored["pct_change"].iloc[0]) == pytest.approx(-6.4865)
    assert stored["market_cap"].isna().sum() == 1


def _ticker(symbol, **columns):
    row = dict.fromkeys(trending.TICKER_COLUMNS)
    row.update(symbol=symbol, first_seen=T0, last_seen=T0, active=True, pinned=False)
    return row | columns


def test_the_tickers_upsert_keeps_first_seen_tier_and_promoted_at(pg):
    earlier = T0 - pd.Timedelta(days=30)
    first = [_ticker("SPCX", first_seen=earlier, last_seen=earlier, pinned=True)]
    assert trending.upsert_tickers(first, trending.SELECTION_COLUMNS) == 1
    with pg.begin() as conn:  # a promotion, by hand (section 8)
        conn.exec_driver_sql("UPDATE tickers SET promoted_at = now()")

    again = [
        _ticker("SPCX", short_name="SpaceX", first_trade_date=dt.date(2026, 6, 12)),
        _ticker("HIMS", active=False),
    ]
    assert trending.upsert_tickers(again, trending.SELECTION_COLUMNS) == 2
    with pg.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT symbol, tier, first_seen, last_seen, pinned, short_name, "
                "first_trade_date, promoted_at IS NOT NULL AS promoted "
                "FROM tickers ORDER BY symbol"
            )
        ).mappings()
        stored = {row["symbol"]: dict(row) for row in rows}
    spcx = stored["SPCX"]
    assert spcx["first_seen"] == earlier and spcx["last_seen"] == T0
    assert spcx["tier"] == "trending" and spcx["promoted"]
    assert spcx["pinned"] is False and spcx["short_name"] == "SpaceX"
    assert spcx["first_trade_date"] == dt.date(2026, 6, 12)
    assert stored["HIMS"]["tier"] == "trending" and not stored["HIMS"]["promoted"]


def test_the_summary_upsert_writes_numerics_at_their_scale(pg):
    trending.upsert_tickers([_ticker("SPCX")], trending.SELECTION_COLUMNS)
    summary = _ticker(
        "SPCX",
        summary_session=dt.date(2026, 11, 10),
        last_close=141.29654,
        rsi_14=55.555,
        recent_flags=3,
        big_move_date=dt.date(2026, 10, 1),
    )
    trending.upsert_tickers([summary], trending.SUMMARY_COLUMNS)
    (row,) = trending.load_tickers([])
    assert row["last_close"] == pytest.approx(141.2965)
    assert row["rsi_14"] == pytest.approx(55.56)
    assert row["recent_flags"] == 3
    assert row["summary_session"] == dt.date(2026, 11, 10)


def test_load_tickers_reads_active_pinned_and_the_named_symbols(pg):
    trending.upsert_tickers(
        [
            _ticker("ACTV"),
            _ticker("PINN", active=False, pinned=True),
            _ticker("GONE", active=False),
            _ticker("NAMED", active=False),
        ],
        trending.SELECTION_COLUMNS,
    )
    rows = trending.load_tickers(["NAMED", "NEVER"])
    assert [r["symbol"] for r in rows] == ["ACTV", "NAMED", "PINN"]
    assert all(r["first_bar_date"] is None for r in rows)


def test_only_sessions_with_a_fetch_where_every_list_answered_count(pg):
    nov = {d: dt.date(2026, 11, d) for d in (4, 5, 6, 9)}
    outcomes = [
        # The 4th: every list answered.
        *[_outcome(T0 - pd.Timedelta(days=6), nov[4], n, True) for n in ("a", "b")],
        # The 5th: one list failed, and no second fetch.
        _outcome(T0 - pd.Timedelta(days=5), nov[5], "a", True),
        _outcome(T0 - pd.Timedelta(days=5), nov[5], "b", False),
        # The 6th: a partial fetch, then a complete one of the same session.
        _outcome(T0 - pd.Timedelta(days=4), nov[6], "a", False),
        _outcome(T0 - pd.Timedelta(days=4), nov[6], "b", True),
        *[_outcome(T0 - pd.Timedelta(days=3), nov[6], n, True) for n in ("a", "b")],
        # The 9th: complete, fetched twice.
        *[_outcome(T0 - pd.Timedelta(days=1), nov[9], n, True) for n in ("a", "b")],
        *[_outcome(T0, nov[9], n, True) for n in ("a", "b")],
    ]
    trending.save_fetch([], outcomes)
    assert trending.counted_sessions(5) == [nov[9], nov[6], nov[4]]
    assert trending.counted_sessions(2) == [nov[9], nov[6]]


# ── The price replacement (7.2) ───────────────────────────────────────────────


def _bars(ticker, days, close):
    return [
        {
            "ticker": ticker,
            "ts": pd.Timestamp(day, tz="UTC"),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1000,
        }
        for day in days
    ]


def _stored_bars(pg, ticker):
    with pg.connect() as conn:
        return pd.read_sql(
            text("SELECT ts, close FROM trending_prices WHERE ticker = :t ORDER BY ts"),
            conn,
            params={"t": ticker},
        )


def _first_bar_date(pg, ticker):
    with pg.connect() as conn:
        return conn.execute(
            text("SELECT first_bar_date FROM tickers WHERE symbol = :t"), {"t": ticker}
        ).scalar_one()


DAYS = pd.bdate_range("2026-05-11", "2026-11-10")


@pytest.mark.parametrize(
    ("break_it", "error"),
    [
        (lambda row: row.update(ts="not a date"), DataError),
        (lambda row: row.update(close=None), IntegrityError),
        (lambda row: row.update(close=1e9), DataError),  # beyond NUMERIC(12, 4)
    ],
)
def test_a_failed_replacement_keeps_the_bars_and_first_bar_date(pg, break_it, error):
    trending.upsert_tickers([_ticker("SPCX"), _ticker("HIMS")], ("active",))
    trending.replace_prices("SPCX", _bars("SPCX", DAYS, 100.0), DAYS[0].date())
    trending.replace_prices("HIMS", _bars("HIMS", DAYS, 20.0), DAYS[0].date())
    before = _stored_bars(pg, "SPCX")

    broken = _bars("SPCX", DAYS[5:], 110.0)
    break_it(broken[-1])
    with pytest.raises(error):
        trending.replace_prices("SPCX", broken, DAYS[5].date())

    pd.testing.assert_frame_equal(_stored_bars(pg, "SPCX"), before)
    assert _first_bar_date(pg, "SPCX") == DAYS[0].date()
    assert len(_stored_bars(pg, "HIMS")) == len(DAYS)  # another symbol untouched


def test_a_replacement_swaps_the_bars_and_records_first_bar_date(pg):
    trending.upsert_tickers([_ticker("SPCX")], ("active",))
    assert trending.replace_prices(
        "SPCX", _bars("SPCX", DAYS, 100.0), DAYS[0].date()
    ) == len(DAYS)
    later = DAYS[DAYS >= "2026-06-12"]
    assert trending.replace_prices(
        "SPCX", _bars("SPCX", later, 101.0), later[0].date()
    ) == len(later)
    stored = _stored_bars(pg, "SPCX")
    assert len(stored) == len(later) and set(stored["close"]) == {101.0}
    assert _first_bar_date(pg, "SPCX") == dt.date(2026, 6, 12)


def test_a_reader_always_finds_the_symbols_bars_during_replacements(pg):
    """As PR #19's check did for forecasts: a second connection polls the
    symbol's row count while its bars are replaced again and again. Deleted
    and inserted in one transaction, the count is never 0."""
    trending.upsert_tickers([_ticker("SPCX")], ("active",))
    trending.replace_prices("SPCX", _bars("SPCX", DAYS, 100.0), DAYS[0].date())

    counts: list[int] = []
    stop = threading.Event()

    def poll():
        with pg.connect() as conn:
            while not stop.is_set():
                counts.append(
                    conn.execute(
                        text(
                            "SELECT count(*) FROM trending_prices WHERE ticker = 'SPCX'"
                        )
                    ).scalar_one()
                )
                conn.commit()  # a fresh snapshot for every poll

    reader = threading.Thread(target=poll)
    reader.start()
    try:
        for i in range(30):
            trending.replace_prices(
                "SPCX", _bars("SPCX", DAYS, 100.0 + i), DAYS[0].date()
            )
            time.sleep(0.005)
    finally:
        stop.set()
        reader.join(timeout=30)

    assert len(counts) > 30, "the reader barely ran; the check proves nothing"
    assert min(counts) == len(DAYS)
