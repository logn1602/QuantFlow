"""tests/integration/test_trending_on_postgres.py

The trending tier's storage on real Postgres (docs/trending.md, sections 6
and 7.2), what the unit suite can only fake:

  - the migration runs twice, creating the five tables and nothing else,
    and builds the same catalog as sql/schema_trending.sql in a schema of
    its own (columns, defaults, constraints, indexes, sequences), a
    comparison shown to catch each kind of difference;
  - a role granted SELECT by default privileges, as the read-only role is
    (sql/schema_roles.sql, step 4), reads every new table and writes none;
  - trending_news has news_sentiment's columns and key;
  - a fetch, the tickers upsert and the reads on the types the driver
    returns; a run's fetch and tickers rows land together or not at all;
  - missing_tables names the tables not there yet;
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
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.exc import DataError, IntegrityError, ProgrammingError

from quantflow.dashboard import trending_view
from quantflow.db import trending
from quantflow.db.batch import insert_rows

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


def _run_script(engine, path, script=None):
    """A SQL file (or script, its text) as psql or Supabase's editor runs
    it: one script, its own BEGIN and COMMIT, on a connection outside any
    SQLAlchemy transaction."""
    raw = engine.raw_connection()
    try:
        raw.driver_connection.autocommit = True
        with raw.cursor() as cursor:
            cursor.execute(script or path.read_text(encoding="utf-8"))
    finally:
        raw.invalidate()  # the connection's autocommit stays out of the pool


def _catalog(engine, schema):
    """What a script built in schema: every column (type, length, precision,
    nullability, default), constraint, index and sequence, with the
    schema's own name taken out, so two schemas' catalogs compare."""
    queries = {
        "columns": """
            SELECT table_name, ordinal_position, column_name, data_type,
                   character_maximum_length, numeric_precision, numeric_scale,
                   is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = :s
            ORDER BY table_name, ordinal_position
        """,
        "constraints": """
            SELECT rel.relname, con.conname, con.contype,
                   pg_get_constraintdef(con.oid)
            FROM pg_constraint con
            JOIN pg_class rel ON rel.oid = con.conrelid
            JOIN pg_namespace ns ON ns.oid = rel.relnamespace
            WHERE ns.nspname = :s
            ORDER BY rel.relname, con.conname
        """,
        "indexes": """
            SELECT tablename, indexname, indexdef
            FROM pg_indexes
            WHERE schemaname = :s
            ORDER BY tablename, indexname
        """,
        "sequences": """
            SELECT sequence_name, data_type, start_value, increment
            FROM information_schema.sequences
            WHERE sequence_schema = :s
            ORDER BY sequence_name
        """,
    }

    def plain(value):
        return value.replace(f"{schema}.", "") if isinstance(value, str) else value

    with engine.connect() as conn:
        return {
            part: [
                tuple(plain(v) for v in row)
                for row in conn.execute(text(query), {"s": schema})
            ]
            for part, query in queries.items()
        }


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


def test_the_migration_runs_twice_and_creates_the_five_tables(scratch_schema):
    assert set(TABLES) == set(trending.TRENDING_TABLES)
    with scratch_schema() as (engine, schema):
        _run_script(engine, MIGRATION)
        assert _tables(engine, schema) == TABLES
        first = _catalog(engine, schema)

        _run_script(engine, MIGRATION)  # idempotent: nothing changes
        assert _catalog(engine, schema) == first


def test_the_migration_and_the_schema_file_build_the_same_catalog(scratch_schema):
    """Each in a schema of its own: run in one, the second script's CREATE
    ... IF NOT EXISTS would skip every table and compare nothing."""
    with (
        scratch_schema() as (migrated, one),
        scratch_schema() as (set_up, other),
    ):
        _run_script(migrated, MIGRATION)
        _run_script(set_up, SCHEMA)
        assert _catalog(migrated, one) == _catalog(set_up, other)


@pytest.mark.parametrize(
    ("old", "new", "part"),
    [
        ("error         VARCHAR(200)", "error         VARCHAR(201)", "columns"),
        ("NOT NULL DEFAULT 'trending'", "NOT NULL DEFAULT 'core'", "columns"),
        (
            "close       NUMERIC(12, 4)  NOT NULL",
            "close       NUMERIC(12, 4)",
            "columns",
        ),
        (
            "CHECK (tier IN ('core', 'trending'))",
            "CHECK (tier IN ('trending', 'core'))",
            "constraints",
        ),
        (
            "UNIQUE (snapshot_ts, symbol, source, list_name)",
            "UNIQUE (snapshot_ts, symbol, list_name, source)",
            "constraints",
        ),
        (
            "ON trending_snapshots (symbol, session_date)",
            "ON trending_snapshots (session_date, symbol)",
            "indexes",
        ),
        ("BIGSERIAL PRIMARY KEY", "SERIAL PRIMARY KEY", "sequences"),
    ],
)
def test_the_catalog_comparison_sees_each_kind_of_difference(
    scratch_schema, old, new, part
):
    """The comparison above proves something only if a drift would fail it:
    one change of each kind to the schema file, each caught."""
    script = SCHEMA.read_text(encoding="utf-8")
    assert script.count(old) == 1, f"the schema file no longer has: {old}"
    with (
        scratch_schema() as (migrated, one),
        scratch_schema() as (drifted, other),
    ):
        _run_script(migrated, MIGRATION)
        _run_script(drifted, SCHEMA, script.replace(old, new))
        assert _catalog(migrated, one)[part] != _catalog(drifted, other)[part]


def test_a_role_with_default_select_reads_every_new_table_and_writes_none(
    scratch_schema,
):
    """The read-only role reads tables postgres creates after it through its
    default privileges (sql/schema_roles.sql, step 4), so the dashboard needs
    no grant after the migration. Modelled here on the test schema."""
    role = f"qf_ro_{uuid.uuid4().hex[:8]}"
    with scratch_schema() as (engine, schema):
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
                # The Trending tab's reads, as the dashboard runs them.
                assert trending_view.load_table(conn).empty
                assert trending_view.load_list_status(conn).empty
                assert trending_view.load_bars(conn, "SPCX").empty
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
    assert trending.save_run(snapshots, outcomes) == (3, 2, 0)
    assert trending.save_run(snapshots, outcomes) == (0, 0, 0)
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


def test_a_runs_fetch_and_tickers_rows_land_together_or_not_at_all(pg):
    """One transaction on Postgres: a tickers row the table refuses (no
    first_seen: NOT NULL) leaves no snapshot and no outcome behind."""
    row = dict.fromkeys(trending.TICKER_COLUMNS) | {
        "symbol": "SPCX",
        "last_seen": T0,
        "active": True,
        "pinned": True,
    }
    snapshots = [_snapshot("SPCX", "most_actives", 1)]
    outcomes = [_outcome(T0, dt.date(2026, 11, 10), "most_actives", True)]
    with pytest.raises(IntegrityError):
        trending.save_run(snapshots, outcomes, [row], trending.SELECTION_COLUMNS)
    with pg.connect() as conn:
        for name in ("trending_snapshots", "trending_fetches", "tickers"):
            count = conn.exec_driver_sql(f"SELECT count(*) FROM {name}")
            assert count.scalar_one() == 0, name

    row["first_seen"] = T0
    written = trending.save_run(snapshots, outcomes, [row], trending.SELECTION_COLUMNS)
    assert written == (1, 1, 1)


def test_missing_tables_names_the_tables_not_there_yet(scratch_schema, install_engine):
    with scratch_schema() as (engine, _):
        install_engine(engine)
        assert trending.missing_tables() == list(trending.TRENDING_TABLES)
        _run_script(engine, MIGRATION)
        assert trending.missing_tables() == []


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
    trending.save_run([], outcomes)
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


# ── The Trending tab's reads (section 10) ─────────────────────────────────────


def _seed_the_tab(pg):
    """Two sessions of fetches, four symbols, some prices and news."""
    earlier = T0 - pd.Timedelta(days=1)
    day = {T0: dt.date(2026, 11, 10), earlier: dt.date(2026, 11, 9)}

    def snap(ts, symbol, list_name, rank, pct):
        return _snapshot(symbol, list_name, rank, pct_change=pct) | {
            "snapshot_ts": ts,
            "session_date": day[ts],
        }

    snapshots = [
        snap(earlier, "HIMS", "most_actives", 4, -1.0),
        snap(earlier, "GONE", "most_actives", 1, 3.0),
        snap(T0, "HIMS", "trending_tickers", 2, -17.98),
        snap(T0, "HIMS", "day_losers", 1, -17.99),
        snap(T0, "TTWO", "trending_tickers", 1, 8.06),
    ]
    outcomes = [
        _outcome(earlier, day[earlier], "most_actives", True),
        _outcome(T0, day[T0], "trending_tickers", True),
        _outcome(T0, day[T0], "day_losers", True),
        _outcome(T0, day[T0], "most_actives", False),
    ]
    trending.save_run(snapshots, outcomes)
    trending.upsert_tickers(
        [
            _ticker("HIMS", short_name="Hims & Hers"),
            _ticker("TTWO"),
            _ticker("SPCX", pinned=True, first_trade_date=dt.date(2026, 6, 12)),
            _ticker("GONE", active=False),
        ],
        trending.SELECTION_COLUMNS,
    )
    trending.upsert_tickers(
        [_ticker("HIMS", summary_session=dt.date(2026, 11, 9), rsi_14=28.61,
                 recent_flags=4, last_close=18.8151)],
        trending.SUMMARY_COLUMNS,
    )  # fmt: skip
    trending.replace_prices("SPCX", _bars("SPCX", DAYS[-30:], 140.0), DAYS[-30].date())
    now = pd.Timestamp.now(tz="UTC")
    news = [
        {"ticker": "HIMS", "source": "rss", "headline": f"h{i}",
         "published_at": now - pd.Timedelta(days=age), "sentiment": "negative",
         "compound": score}
        for i, (age, score) in enumerate([(1, -0.5), (2, -0.3), (10, 0.9)])
    ]  # fmt: skip
    with pg.begin() as conn:
        insert_rows(conn, "trending_news", news)


def test_the_tabs_table_is_one_grouped_read_of_the_active_symbols(pg):
    _seed_the_tab(pg)
    with pg.connect() as conn:
        table = trending_view.load_table(conn)

    # Active only, by today's best rank, the pinned symbol off the lists last.
    assert list(table["symbol"]) == ["HIMS", "TTWO", "SPCX"]
    hims = table.iloc[0]
    assert hims["best_rank"] == 1 and sorted(hims["lists"]) == [
        "day_losers",
        "trending_tickers",
    ]
    assert float(hims["pct_change"]) == pytest.approx(-17.99)  # the best-ranked item
    assert hims["days_trending"] == 2  # two sessions, counted once each
    assert hims["articles_7d"] == 2  # the 10-day-old article is out of the window
    assert float(hims["sentiment_7d"]) == pytest.approx(-0.4)
    assert hims["short_name"] == "Hims & Hers" and hims["recent_flags"] == 4
    spcx = table.iloc[2]
    assert pd.isna(spcx["best_rank"]) and spcx["days_trending"] == 0
    assert spcx["articles_7d"] == 0 and bool(spcx["pinned"])

    shown = trending_view.table_rows(
        table, settings=_view_settings(), today=dt.date(2026, 11, 10)
    )
    assert shown["Lists"].iloc[0] == "trending_tickers, day_losers"
    assert list(shown["Badges"]) == ["", "", "pinned"]


def _view_settings():
    from quantflow.ingestion import trending_rules

    return trending_rules.TrendingSettings(
        pinned=("SPCX",),
        lists=trending_rules.KNOWN_LISTS,
        max_active=15,
        max_new_per_run=5,
        inactive_after=5,
        min_price=5.0,
        min_market_cap=None,
        promotion_min_days=20,
        promotion_min_history_days=504,
    )


def test_the_tabs_list_line_shows_each_lists_last_success(pg):
    _seed_the_tab(pg)
    with pg.connect() as conn:
        status = trending_view.load_list_status(conn)
    lines = trending_view.list_lines(status, _view_settings().lists)
    assert lines == [
        "trending_tickers: last succeeded 2026-11-10 23:10 UTC",
        "most_actives: last succeeded 2026-11-09 23:10 UTC; failed since",
        "day_gainers: never fetched",
        "day_losers: last succeeded 2026-11-10 23:10 UTC",
    ]


def test_the_tabs_chart_reads_one_symbols_bars(pg):
    _seed_the_tab(pg)
    with pg.connect() as conn:
        bars = trending_view.load_bars(conn, "SPCX")
        none = trending_view.load_bars(conn, "HIMS")
    assert len(bars) == 30 and none.empty
    prices, indicators, _ = trending_view.chart_frames(bars, "SPCX")
    assert len(prices) == 30 and not indicators.empty


def test_before_the_migration_the_tab_reads_its_empty_state(scratch_schema):
    """The tables are not there yet: each read is an empty frame, not an
    error, and the connection is usable after."""
    with (
        scratch_schema() as (engine, _),
        engine.connect() as conn,
    ):
        assert trending_view.load_table(conn).empty
        assert trending_view.load_list_status(conn).empty
        assert trending_view.load_bars(conn, "SPCX").empty
        assert conn.exec_driver_sql("SELECT 1").scalar_one() == 1
