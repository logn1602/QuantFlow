"""tests/integration/test_trending_on_postgres.py

The trending tier's storage on real Postgres (docs/trending.md, sections 6
and 7.2), what the unit suite can only fake:

  - the migration creates the five tables and nothing else, a second run
    changes nothing, and it builds the same catalog as
    sql/schema_trending.sql in a schema of its own (columns, defaults,
    constraints, indexes, sequences, row-level security), a comparison
    shown to catch each kind of difference;
  - after it, the read-only role, granted SELECT by default privileges
    (sql/schema_roles.sql, step 4), reads every row through its policy and
    writes none; a role with the same SELECT and no policy sees no row; and
    anon and authenticated, granted everything by Supabase's default
    privileges, hold nothing;
  - trending_news has news_sentiment's columns and key;
  - a fetch, the tickers upsert and the reads on the types the driver
    returns; a run's fetch and tickers rows land together or not at all;
  - missing_tables names the tables not there yet;
  - each symbol's price replacement is one transaction: a failed insert
    keeps its bars and first_bar_date, and a second connection polling
    during replacements always finds the symbol's bars;
  - the tier's writer, quantflow_trending (sql/schema_roles.sql, steps 11
    and 13, run as written): two whole runs, logged in as it, store what
    they should, and every command outside what a run uses is refused.

Needs a disposable local Postgres, never the live database:

    uv run pytest -m integration --pg-url postgresql://user@127.0.0.1:5432/scratch
"""

import datetime as dt
import re
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2.errors
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DataError, IntegrityError, ProgrammingError

from quantflow.dashboard import trending_view
from quantflow.db import trending
from quantflow.db.batch import insert_rows
from quantflow.features import sentiment
from quantflow.ingestion import trending_rules as rules
from quantflow.ingestion.trending_sources import ListFetch, TrendingItem
from quantflow.pipelines import trending as tier

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

# The roles the migration names: Supabase's anon and authenticated, and the
# owner's quantflow_readonly (sql/schema_roles.sql). Roles are cluster-wide:
# on the disposable cluster each is made once, with no login, and left for
# the next run, which finds it.
MIGRATION_ROLES = ("anon", "authenticated", "quantflow_readonly")


@pytest.fixture
def migration_roles(pg_url):
    admin = create_engine(pg_url)
    try:
        with admin.begin() as conn:
            existing = {
                row[0]
                for row in conn.execute(
                    text("SELECT rolname FROM pg_roles WHERE rolname = ANY(:names)"),
                    {"names": list(MIGRATION_ROLES)},
                )
            }
            for role in MIGRATION_ROLES:
                if role not in existing:
                    conn.exec_driver_sql(f"CREATE ROLE {role} NOLOGIN")
    finally:
        admin.dispose()


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
        "row_security": """
            SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity
            FROM pg_class c
            JOIN pg_namespace ns ON ns.oid = c.relnamespace
            WHERE ns.nspname = :s AND c.relkind = 'r'
            ORDER BY c.relname
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


def _access(engine, schema):
    """The schema's policies, and what anon, authenticated and
    quantflow_readonly were granted on its tables and sequences."""
    with engine.connect() as conn:
        policies = conn.execute(
            text("""
                SELECT tablename, policyname, permissive, roles::text, cmd,
                       qual, with_check
                FROM pg_policies
                WHERE schemaname = :s
                ORDER BY tablename, policyname
            """),
            {"s": schema},
        )
        grants = conn.execute(
            text("""
                SELECT c.relname, r.rolname, acl.privilege_type
                FROM pg_class c
                JOIN pg_namespace ns ON ns.oid = c.relnamespace
                CROSS JOIN LATERAL aclexplode(c.relacl) AS acl
                JOIN pg_roles r ON r.oid = acl.grantee
                WHERE ns.nspname = :s AND r.rolname = ANY(:roles)
                ORDER BY 1, 2, 3
            """),
            {"s": schema, "roles": list(MIGRATION_ROLES)},
        )
        return {
            "policies": [tuple(row) for row in policies],
            "grants": [tuple(row) for row in grants],
        }


def _one_row_each(engine):
    """A row in each of the five tables, written as their owner."""
    with engine.begin() as conn:
        for statement in (
            "INSERT INTO tickers (symbol, first_seen, last_seen) "
            "VALUES ('SPCX', now(), now())",
            "INSERT INTO trending_prices (ticker, ts, close) "
            "VALUES ('SPCX', '2026-11-10 00:00+00', 1)",
            "INSERT INTO trending_snapshots (snapshot_ts, session_date, symbol, "
            "source, list_name, rank) VALUES (now(), '2026-11-10', 'SPCX', "
            "'yahoo_screener', 'most_actives', 1)",
            "INSERT INTO trending_fetches (fetched_at, session_date, source, "
            "list_name, succeeded, items) VALUES (now(), '2026-11-10', "
            "'yahoo_screener', 'most_actives', true, 1)",
            "INSERT INTO trending_news (ticker, source, published_at, headline, "
            "sentiment) VALUES ('SPCX', 'rss', now(), 'a headline', 'neutral')",
        ):
            conn.exec_driver_sql(statement)


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


@pytest.mark.usefixtures("migration_roles")
def test_the_migration_creates_the_five_tables_and_a_second_run_changes_nothing(
    scratch_schema,
):
    """A policy has no IF NOT EXISTS: a second run stops at its first CREATE
    POLICY, and being one transaction, leaves everything as it was."""
    assert set(TABLES) == set(trending.TRENDING_TABLES)
    with scratch_schema() as (engine, schema):
        _run_script(engine, MIGRATION)
        assert _tables(engine, schema) == TABLES
        first = _catalog(engine, schema), _access(engine, schema)

        with pytest.raises(psycopg2.errors.DuplicateObject, match="already exists"):
            _run_script(engine, MIGRATION)
        assert (_catalog(engine, schema), _access(engine, schema)) == first


@pytest.mark.usefixtures("migration_roles")
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
        (
            "ALTER TABLE trending_news ENABLE ROW LEVEL SECURITY;",
            "",
            "row_security",
        ),
    ],
)
@pytest.mark.usefixtures("migration_roles")
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


@pytest.mark.usefixtures("migration_roles")
def test_only_the_read_only_role_reads_the_rows_and_the_api_roles_hold_nothing(
    scratch_schema,
):
    """After the migration, modelled on the test schema as the live database
    has it: quantflow_readonly, granted SELECT by default privileges
    (sql/schema_roles.sql, step 4), reads every row through its policy, so
    the dashboard needs no grant, and writes none. A role granted the same
    SELECT but no policy sees no row, and no error: how every core table
    looked to quantflow_readonly before step 7. Supabase's default
    privileges grant anon and authenticated everything; the revokes leave
    them nothing."""
    other = f"qf_nopolicy_{uuid.uuid4().hex[:8]}"
    readers = f"quantflow_readonly, {other}"
    with scratch_schema() as (engine, schema):
        with engine.begin() as conn:
            conn.exec_driver_sql(f"CREATE ROLE {other} NOLOGIN")
            conn.exec_driver_sql(
                f"GRANT USAGE ON SCHEMA {schema} TO {readers}, anon, authenticated"
            )
            conn.exec_driver_sql(
                f"ALTER DEFAULT PRIVILEGES IN SCHEMA {schema} "
                f"GRANT SELECT ON TABLES TO {readers}"
            )
            for kind in ("TABLES", "SEQUENCES"):
                conn.exec_driver_sql(
                    f"ALTER DEFAULT PRIVILEGES IN SCHEMA {schema} "
                    f"GRANT ALL ON {kind} TO anon, authenticated"
                )
        try:
            _run_script(engine, MIGRATION)
            _one_row_each(engine)
            with engine.connect() as conn:
                conn.exec_driver_sql("SET ROLE quantflow_readonly")
                for name in TABLES:
                    count = conn.exec_driver_sql(f"SELECT count(*) FROM {name}")
                    assert count.scalar_one() == 1, name
                # The Trending tab's reads, as the dashboard runs them.
                assert len(trending_view.load_table(conn)) == 1
                assert len(trending_view.load_list_status(conn)) == 1
                assert len(trending_view.load_bars(conn, "SPCX")) == 1
                with pytest.raises(ProgrammingError, match="permission denied"):
                    conn.exec_driver_sql(
                        "INSERT INTO trending_fetches VALUES "
                        "(now(), current_date, 's', 'l', true, 0, null)"
                    )
                conn.rollback()
                conn.exec_driver_sql(f"SET ROLE {other}")
                for name in TABLES:
                    count = conn.exec_driver_sql(f"SELECT count(*) FROM {name}")
                    assert count.scalar_one() == 0, name
                conn.rollback()
            access = _access(engine, schema)
            # Of the three roles, only the read-only one holds anything: SELECT.
            assert access["grants"] == [
                (name, "quantflow_readonly", "SELECT") for name in sorted(TABLES)
            ]
            assert access["policies"] == [
                (name, "quantflow_readonly_read", "PERMISSIVE",
                 "{quantflow_readonly}", "SELECT", "true", None)
                for name in sorted(TABLES)
            ]  # fmt: skip
            with engine.connect() as conn:
                for role in ("anon", "authenticated"):
                    held = conn.execute(
                        text("""
                            SELECT has_table_privilege(:r, 'tickers',
                                       'SELECT, INSERT, UPDATE, DELETE, TRUNCATE')
                                OR has_table_privilege(:r, 'trending_news',
                                       'SELECT, INSERT, UPDATE, DELETE, TRUNCATE')
                                OR has_sequence_privilege(:r,
                                       'trending_news_id_seq', 'USAGE, SELECT, UPDATE')
                        """),
                        {"r": role},
                    ).scalar_one()
                    assert held is False, role
        finally:
            with engine.begin() as conn:
                conn.exec_driver_sql(
                    f"ALTER DEFAULT PRIVILEGES IN SCHEMA {schema} "
                    f"REVOKE ALL ON TABLES FROM {readers}, anon, authenticated"
                )
                conn.exec_driver_sql(
                    f"ALTER DEFAULT PRIVILEGES IN SCHEMA {schema} "
                    f"REVOKE ALL ON SEQUENCES FROM anon, authenticated"
                )
                conn.exec_driver_sql(f"DROP OWNED BY {other}")
                conn.exec_driver_sql(f"DROP ROLE {other}")


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


@pytest.mark.usefixtures("migration_roles")
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


def test_the_tabs_chart_reads_the_latest_bars_oldest_first(pg):
    days = pd.bdate_range(end="2026-11-10", periods=trending_view.CHART_BARS + 40)
    with pg.begin() as conn:
        insert_rows(conn, "trending_prices", _bars("SPCX", days, 20.0))
    with pg.connect() as conn:
        bars = trending_view.load_bars(conn, "SPCX")
    assert len(bars) == trending_view.CHART_BARS
    assert bars["ts"].is_monotonic_increasing
    assert bars["ts"].iloc[-1] == pd.Timestamp("2026-11-10", tz="UTC")
    assert bars["ts"].iloc[0] == pd.Timestamp(days[40], tz="UTC")


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


# ── The tier's writer (sql/schema_roles.sql, steps 11 to 14) ──────────────────

WRITER = "quantflow_trending"
ROLES = ROOT / "sql" / "schema_roles.sql"
CORE_SCHEMAS = (
    "schema.sql",
    "schema_sentiment.sql",
    "schema_metrics.sql",
    "schema_backtest.sql",
)


def _sql_statements(sql):
    """Each statement, comments dropped and whitespace made single spaces
    (as tests/unit/test_schema_roles.py reads the file)."""
    without_comments = re.sub(r"--[^\n]*", "", sql)
    return [
        " ".join(statement.split())
        for statement in without_comments.split(";")
        if statement.strip()
    ]


def _writer_statements():
    statements = _sql_statements(ROLES.read_text(encoding="utf-8"))
    return [s for s in statements if re.search(rf"\b{WRITER}\b", s)]


def _step_13():
    """Step 13 as the owner runs it: its BEGIN to its COMMIT."""
    statements = _sql_statements(ROLES.read_text(encoding="utf-8"))
    first = next(
        i
        for i, s in enumerate(statements)
        if s.startswith("CREATE POLICY ") and f" TO {WRITER} " in s
    )
    begin = max(i for i in range(first) if statements[i] == "BEGIN")
    end = statements.index("COMMIT", first)
    return ";\n".join(statements[begin : end + 1]) + ";\n"


@pytest.fixture
def writer_role(pg_url):
    """quantflow_trending, made once on the disposable cluster with no
    login, as migration_roles makes the others: the tests reach it with
    SET ROLE. Its own statements (step 11) are run, as written, by the
    test below."""
    admin = create_engine(pg_url)
    try:
        with admin.begin() as conn:
            found = conn.execute(
                text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": WRITER}
            ).first()
            if found is None:
                conn.exec_driver_sql(f"CREATE ROLE {WRITER} NOLOGIN")
    finally:
        admin.dispose()


def test_step_11_makes_a_login_that_can_reach_the_database_and_do_nothing_else(
    pg_url,
):
    """As written, under a name of its own (roles are cluster-wide, and
    quantflow_trending may be there already), in a transaction rolled back
    after the checks: CREATE ROLE, ALTER ROLE and GRANT are all
    transactional."""
    other = f"qf_writer_{uuid.uuid4().hex[:8]}"
    statements = [
        re.sub(rf"\b{WRITER}\b", other, s)
        for s in _writer_statements()
        if s.startswith(("CREATE ROLE", "ALTER ROLE", "GRANT CONNECT", "GRANT USAGE ON SCHEMA"))
    ]  # fmt: skip
    assert len(statements) == 5
    admin = create_engine(pg_url)
    try:
        with admin.connect() as conn:
            for statement in statements:
                conn.exec_driver_sql(statement)
            role = conn.execute(
                text("""
                    SELECT rolcanlogin, rolsuper, rolcreatedb, rolcreaterole,
                           rolreplication, rolbypassrls, rolconnlimit,
                           rolconfig, rolinherit
                    FROM pg_roles WHERE rolname = :r
                """),
                {"r": other},
            ).one()
            reach = conn.execute(
                text("""
                    SELECT has_database_privilege(:r, current_database(), 'CONNECT'),
                           has_schema_privilege(:r, 'public', 'USAGE'),
                           has_schema_privilege(:r, 'public', 'CREATE')
                """),
                {"r": other},
            ).one()
            conn.rollback()
    finally:
        admin.dispose()
    assert tuple(role)[:7] == (True, False, False, False, False, False, 5)
    assert sorted(role.rolconfig) == [
        "idle_in_transaction_session_timeout=60s",
        "statement_timeout=120s",
    ]
    assert tuple(reach) == (True, True, False)


NOW = pd.Timestamp("2026-11-10 23:10", tz="UTC")  # Tuesday 18:10 New York
NEXT = pd.Timestamp("2026-11-11 23:10", tz="UTC")  # the next session's run
SETTINGS = rules.TrendingSettings(
    pinned=("SPCX",),
    lists=rules.KNOWN_LISTS,
    max_active=3,
    max_new_per_run=5,
    # One counted session absent deactivates a symbol, so the second run
    # also writes a deactivation.
    inactive_after=1,
    min_price=5.0,
    min_market_cap=None,
    promotion_min_days=20,
    promotion_min_history_days=504,
)


class _Lists:
    """A source whose four lists answered: trending_tickers and most_actives
    with the given symbols, the day's gainers and losers with none."""

    name = "fake"

    def __init__(self, now, *symbols):
        self.fetches = [
            ListFetch(
                source="yahoo_trending" if name == "trending_tickers" else "yahoo_screener",
                list_name=name,
                items=tuple(
                    TrendingItem(
                        symbol=symbol,
                        source="yahoo_trending" if name == "trending_tickers" else "yahoo_screener",
                        list_name=name,
                        rank=rank,
                        pct_change=2.5,
                        volume=1_000_000,
                        market_cap=10**10,
                        price=50.0,
                        quote_type="EQUITY",
                        short_name=f"{symbol} Inc",
                        first_trade_date=dt.date(2015, 3, 2),
                        fetched_at=now.to_pydatetime(),
                    )
                    for rank, symbol in enumerate(listed, start=1)
                ),
            )
            for name, listed in (
                ("trending_tickers", symbols),
                ("most_actives", symbols[::-1]),
                ("day_gainers", ()),
                ("day_losers", ()),
            )
        ]  # fmt: skip

    def fetch(self):
        return list(self.fetches)


def _yahoo(last_day):
    """yfinance_fetcher.download, as the run calls it: every symbol's daily
    bars, as Yahoo sends them (naive exchange dates), up to last_day."""
    days = pd.bdate_range("2026-05-11", last_day)
    steps = np.arange(len(days))

    def get(symbol, **kwargs):
        closes = 20.0 + 0.05 * steps + np.sin(steps)
        frame = pd.DataFrame(
            {
                "Open": closes,
                "High": closes + 1,
                "Low": closes - 1,
                "Close": closes,
                "Adj Close": closes,
                "Volume": 1_000_000,
            },
            index=pd.DatetimeIndex(days),
        )
        if "start" in kwargs:
            frame = frame[frame.index >= pd.Timestamp(kwargs["start"])]
        return frame

    return get


@pytest.mark.usefixtures("migration_roles", "writer_role")
def test_two_whole_runs_work_as_the_writer_and_nothing_else_does(
    pg_url, scratch_schema, schema_engine, install_engine, monkeypatch
):
    """The live database's shape: the core's tables and the tier's, owned
    by the role that made them (postgres there), with row-level security on
    the tier's, then step 13 as written. USAGE on this schema stands in for
    step 11's on public. Then two whole runs logged in as the writer, the
    lists, Yahoo, RSS and FinBERT faked: the second, a session later, takes
    every path a run writes by (the upsert's update, a replacement of stored
    bars, a deactivation, a headline skipped as stored). Then every command
    outside those, refused."""
    run_at = {"now": NOW}

    def fetch_rss(symbol, feeds=None):
        # One headline every run sends, one new each run.
        return [
            {"headline": f"{symbol} again", "url": None,
             "published_at": NOW.to_pydatetime(), "source": "rss"},
            {"headline": f"{symbol} on {run_at['now'].date()}", "url": None,
             "published_at": run_at["now"].to_pydatetime(), "source": "rss"},
        ]  # fmt: skip

    def analyze_batch(headlines):
        score = {
            "sentiment": "positive",
            "score_pos": 0.6,
            "score_neg": 0.1,
            "score_neu": 0.3,
            "compound": 0.5,
        }
        return [dict(score) for _ in headlines]

    monkeypatch.setattr(sentiment, "get_finbert", object)
    monkeypatch.setattr(sentiment, "fetch_rss", fetch_rss)
    monkeypatch.setattr(sentiment, "analyze_batch", analyze_batch)

    with scratch_schema() as (owner, schema):
        for name in CORE_SCHEMAS:
            _run_script(owner, ROOT / "sql" / name)
        _run_script(owner, MIGRATION)
        _run_script(owner, None, _step_13())
        with owner.begin() as conn:
            conn.exec_driver_sql(f"GRANT USAGE ON SCHEMA {schema} TO {WRITER}")
        writer = schema_engine(pg_url, schema, role=WRITER)
        try:
            install_engine(writer)
            with writer.connect() as conn:
                assert conn.exec_driver_sql("SELECT current_user").scalar() == WRITER
            assert trending.missing_tables() == []

            first = tier.run(
                core=("AAPL", "MSFT"),
                settings=SETTINGS,
                sources=[_Lists(NOW, "HIMS", "TTWO", "AAON")],
                listing={},
                now=NOW,
                get=_yahoo("2026-11-10"),
                pause=0,
            )
            assert first.exit_code == 0, first
            assert set(first.active) == {"SPCX", "HIMS", "TTWO", "AAON"}
            assert set(first.prices_replaced) == set(first.active)

            run_at["now"] = NEXT
            second = tier.run(
                core=("AAPL", "MSFT"),
                settings=SETTINGS,
                sources=[_Lists(NEXT, "HIMS", "TTWO")],
                listing={},
                now=NEXT,
                get=_yahoo("2026-11-11"),
                pause=0,
            )
            assert second.exit_code == 0, second
            assert set(second.active) == {"SPCX", "HIMS", "TTWO"}
            assert set(second.prices_replaced) == set(second.active)

            # A read no policy allows finds no row, with no error: so every
            # table is counted as the writer, against the owner's counts
            # below, and the run's own reads are checked by what they return.
            assert trending.counted_sessions(5) == [
                dt.date(2026, 11, 11),
                dt.date(2026, 11, 10),
            ]
            assert {r["symbol"] for r in trending.load_tickers([])} == {
                "SPCX",
                "HIMS",
                "TTWO",
            }
            with writer.connect() as conn:
                seen = {
                    name: conn.exec_driver_sql(f"SELECT count(*) FROM {name}").scalar()
                    for name in TABLES
                }

            refused = (
                # No core table, read or written.
                "SELECT count(*) FROM raw_prices",
                "INSERT INTO news_sentiment (ticker, source, published_at, "
                "headline, sentiment) VALUES ('AAPL', 'rss', now(), 'h', 'neutral')",
                "UPDATE forecasts SET predicted_close = 0",
                # No command a run does not use.
                "DELETE FROM tickers",
                "DELETE FROM trending_snapshots",
                "DELETE FROM trending_news",
                "UPDATE trending_prices SET close = 1",
                "UPDATE trending_fetches SET succeeded = true",
                "TRUNCATE trending_news",
                # Never a promotion, which is set by hand.
                "UPDATE tickers SET tier = 'core'",
                "UPDATE tickers SET promoted_at = now()",
                "UPDATE tickers SET first_seen = now()",
                "INSERT INTO tickers (symbol, tier, first_seen, last_seen) "
                "VALUES ('ZZZZ', 'core', now(), now())",
                # Nothing it does not own.
                "CREATE TABLE writer_owned (a int)",
                "ALTER TABLE tickers DISABLE ROW LEVEL SECURITY",
                f"DROP POLICY {WRITER}_read ON tickers",
            )
            with writer.connect() as conn:
                for statement in refused:
                    with pytest.raises(
                        ProgrammingError, match=r"permission denied|must be owner"
                    ):
                        conn.exec_driver_sql(statement)
                    conn.rollback()
        finally:
            writer.dispose()

        with owner.connect() as conn:
            tickers = pd.read_sql(
                text("SELECT symbol, tier, active, promoted_at FROM tickers"), conn
            ).set_index("symbol")
            fetches = conn.exec_driver_sql(
                "SELECT count(*), bool_and(succeeded) FROM trending_fetches"
            ).one()
            bars = dict(
                conn.exec_driver_sql(
                    "SELECT ticker, max(ts)::date FROM trending_prices GROUP BY ticker"
                ).all()
            )
            news = conn.exec_driver_sql("SELECT count(*) FROM trending_news").scalar()
            stored = {
                name: conn.exec_driver_sql(f"SELECT count(*) FROM {name}").scalar()
                for name in TABLES
            }
            core = {
                name: conn.exec_driver_sql(f"SELECT count(*) FROM {name}").scalar()
                for name in ("raw_prices", "news_sentiment", "forecasts")
            }
    assert dict(tickers["active"]) == {
        "SPCX": True, "HIMS": True, "TTWO": True, "AAON": False,
    }  # fmt: skip
    assert set(tickers["tier"]) == {"trending"}
    assert tickers["promoted_at"].isna().all()
    assert tuple(fetches) == (8, True)
    # AAON's bars stay as the first run stored them; the rest moved on.
    assert bars == {
        "SPCX": dt.date(2026, 11, 11),
        "HIMS": dt.date(2026, 11, 11),
        "TTWO": dt.date(2026, 11, 11),
        "AAON": dt.date(2026, 11, 10),
    }
    # Two headlines for each of four symbols, then one new for each of three.
    assert news == 4 * 2 + 3
    assert seen == stored and all(stored.values()), (seen, stored)
    assert core == {"raw_prices": 0, "news_sentiment": 0, "forecasts": 0}
