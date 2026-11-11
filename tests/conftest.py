"""Shared pytest configuration.

The sys.path manipulation this file used to perform is gone: quantflow is an
installed package, so tests import it the same way anything else does.

Tests are offline by design: no database, no network, no model training. The
integration and eval markers exist for suites that need those; see
pyproject.toml.

Offline unit support:

  _no_database (autouse)  swaps get_engine everywhere for one that raises, so a
                          test that forgets to fake the database fails instead
                          of reaching the live Supabase instance that .env
                          points at.
  _no_network (autouse)   makes yf.download, yf.Ticker, every other yfinance
                          request and every requests call raise, so a test
                          that forgets to fake Yahoo, Alpha Vantage or NewsAPI
                          never reaches them.
  _no_committed_acknowledgements (autouse)
                          points config.SPLIT_JUMP_ACKNOWLEDGED_FILE at an
                          empty list, so no test depends on what
                          config/split_jump_acknowledged.txt lists or breaks
                          when it is edited. A test that needs entries writes
                          its own file (see test_split_jump_check.py).
  fake_db                 an in-memory stand-in that records every statement
                          and applies INSERT ... ON CONFLICT DO NOTHING, an
                          upsert's DO UPDATE SET and a keyed UPDATE the way
                          Postgres does, so a write path can be checked for
                          what it reports against what it actually wrote.
  read_tables             canned pd.read_sql results by table name, for code
                          that reads through pandas.
  price_bars              raw_prices rows in the two shapes yfinance stores:
                          daily bars at 00:00 UTC and 15-min bars on the New
                          York session grid.

Integration support (tests/integration, a disposable local Postgres):

  --pg-url                the database to use; nothing runs without it.
  is_local_pg_url         the check that it can only reach this machine.
  install_engine          routes every get_engine call to the test's engine.
  pg_url                  --pg-url after those checks, or a skip without it.
  pg                      a schema of the test's own, created from the schema
                          files the write paths use (SCHEMA_FILES), installed
                          as every get_engine and dropped after the test.
  schema_engine           an engine on a given schema with session settings of
                          its own (a New York TimeZone, say).
  scratch_schema          a context manager: an empty schema of the test's own
                          and an engine on it, dropped after (pg builds on it).
"""

import copy
import functools
import re
import sys
import uuid
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql.elements import TextClause

# Unique keys of the tables the write paths touch, as declared in sql/*.sql.
CONFLICT_KEYS = {
    "raw_prices": ("ticker", "source", "ts"),
    "news_sentiment": ("ticker", "headline", "published_at"),
    "forecasts": ("ticker", "model", "forecast_date", "run_at"),
    "technical_indicators": ("ticker", "ts"),
    # sql/schema.sql's key. The live table predates it and has none yet (see
    # sql/migrations/001_dedupe_anomalies.sql), so nothing conflicts there.
    "anomalies": ("ticker", "ts"),
    # sql/schema_trending.sql.
    "tickers": ("symbol",),
    "trending_prices": ("ticker", "ts"),
    "trending_snapshots": ("snapshot_ts", "symbol", "source", "list_name"),
    "trending_fetches": ("fetched_at", "source", "list_name"),
    "trending_news": ("ticker", "headline", "published_at"),
}

# ON CONFLICT ... DO UPDATE SET <column> = excluded.<column>, as SQLAlchemy
# compiles an upsert.
_UPDATED_COLUMN = re.compile(r"(\w+) = excluded\.(\w+)")

# A multi-row Core INSERT binds row i's columns as "<column>_m<i>".
_MULTI_ROW_PARAM = re.compile(r"^(?P<column>.+)_m(?P<row>\d+)$")


def _statement_rows(statement, params):
    """SQL text plus parameter rows, for text() and Core statements alike."""
    if isinstance(statement, TextClause) or not hasattr(statement, "compile"):
        sql = str(statement)
    else:
        compiled = statement.compile(dialect=postgresql.dialect())
        sql = str(compiled)
        if params is None:
            grouped = defaultdict(dict)
            for key, value in compiled.params.items():
                match = _MULTI_ROW_PARAM.match(key)
                if match:
                    grouped[int(match["row"])][match["column"]] = value
                else:
                    grouped[0][key] = value
            return sql, [grouped[i] for i in sorted(grouped)]
    if params is None:
        return sql, []
    return sql, [params] if isinstance(params, dict) else list(params)


class FakeResult:
    def __init__(self, rowcount):
        self.rowcount = rowcount


class FakeDatabase:
    """Records statements and keeps the rows that INSERTs actually wrote."""

    def __init__(self) -> None:
        self.tables: dict[str, list[dict]] = defaultdict(list)
        self.statements: list[tuple[str, list[dict]]] = []
        self.rollbacks = 0
        self.fail_next_insert: Exception | None = None

    def seed(self, table, rows):
        self.tables[table].extend(dict(row) for row in rows)

    def inserts(self, table):
        return [
            (sql, rows)
            for sql, rows in self.statements
            if re.match(rf"\s*INSERT INTO\s+{table}\b", sql, re.IGNORECASE)
        ]

    def deletes(self, table):
        return [
            (sql, rows)
            for sql, rows in self.statements
            if re.match(rf"\s*DELETE FROM\s+{table}\b", sql, re.IGNORECASE)
        ]

    def execute(self, statement, params=None):
        sql, rows = _statement_rows(statement, params)
        self.statements.append((sql, rows))
        verb = sql.lstrip().split(None, 1)[0].upper()
        if verb == "INSERT":
            return self._insert(sql, rows)
        if verb == "DELETE":
            return self._delete(sql, rows)
        if verb == "UPDATE":
            return self._update(sql, rows)
        return FakeResult(0)

    def _insert(self, sql, rows):
        if self.fail_next_insert is not None:
            error, self.fail_next_insert = self.fail_next_insert, None
            raise error
        match = re.search(r"INSERT INTO\s+(\w+)", sql, re.IGNORECASE)
        assert match, f"cannot find the target table in: {sql}"
        table = match.group(1)
        key_cols = CONFLICT_KEYS.get(table, ())
        stored = {tuple(r.get(c) for c in key_cols) for r in self.tables[table]}
        skip_conflicts = "ON CONFLICT" in sql.upper()
        # An upsert overwrites the SET columns of the stored row and, as in
        # Postgres, counts it.
        update = sql.upper().partition(" DO UPDATE SET ")
        updated = [
            target
            for target, source in _UPDATED_COLUMN.findall(sql[len(update[0]) :])
            if target == source
        ]
        written = 0
        for row in rows:
            key = tuple(row.get(c) for c in key_cols)
            if key_cols and key in stored:
                if updated:
                    existing = next(
                        r
                        for r in self.tables[table]
                        if tuple(r.get(c) for c in key_cols) == key
                    )
                    existing.update({column: row[column] for column in updated})
                    written += 1
                    continue
                if skip_conflicts:
                    continue
                raise IntegrityError(sql, row, Exception("duplicate key value"))
            stored.add(key)
            self.tables[table].append(dict(row))
            written += 1
        return FakeResult(written)

    def _update(self, sql, rows):
        """UPDATE <table> SET <column> = :<param>, ... WHERE <key> = :<param>,
        the one form the write paths send; anything else fails the test."""
        match = re.fullmatch(
            r"\s*UPDATE\s+(\w+)\s+SET\s+(.+?)\s+WHERE\s+(\w+)\s*=\s*:(\w+)\s*",
            sql,
            re.IGNORECASE | re.DOTALL,
        )
        assert match, f"the fake database cannot apply: {sql}"
        table, assignments, key, key_param = match.groups()
        pairs = re.findall(r"(\w+)\s*=\s*:(\w+)", assignments)
        changed = 0
        for params in rows:
            for row in self.tables[table]:
                if row.get(key) == params[key_param]:
                    row.update({column: params[param] for column, param in pairs})
                    changed += 1
        return FakeResult(changed)

    def _delete(self, sql, rows):
        match = re.search(r"DELETE FROM\s+(\w+)", sql, re.IGNORECASE)
        assert match, f"cannot find the target table in: {sql}"
        table = match.group(1)
        params = rows[0] if rows else {}
        tickers = params.get("tickers")
        if tickers is None:
            return FakeResult(0)
        models = params.get("models")
        before = len(self.tables[table])
        self.tables[table] = [
            r
            for r in self.tables[table]
            if r.get("ticker") not in tickers
            or (models is not None and r.get("model") not in models)
        ]
        return FakeResult(before - len(self.tables[table]))


class _FakeConnection:
    def __init__(self, db):
        self._db = db

    def execute(self, statement, params=None):
        return self._db.execute(statement, params)


class _FakeEngine:
    """begin() is transactional: an exception restores every table."""

    def __init__(self, db):
        self._db = db

    @contextmanager
    def begin(self):
        snapshot = copy.deepcopy(self._db.tables)
        try:
            yield _FakeConnection(self._db)
        except BaseException:
            self._db.tables = snapshot
            self._db.rollbacks += 1
            raise

    @contextmanager
    def connect(self):
        yield _FakeConnection(self._db)


def _replace_get_engine(monkeypatch, replacement):
    """Swap get_engine in quantflow.db.connection and in every loaded module
    that imported it by name. Modules imported later read the patched
    connection attribute, so they pick up the replacement too."""
    import quantflow.db.connection as connection

    monkeypatch.setattr(connection, "get_engine", replacement)
    for name, module in list(sys.modules.items()):
        if (
            name.startswith("quantflow.")
            and module is not connection
            and getattr(module, "get_engine", None) is not None
        ):
            monkeypatch.setattr(module, "get_engine", replacement)


def pytest_addoption(parser):
    parser.addoption(
        "--pg-url",
        default=None,
        help="a disposable local Postgres for the integration suite; "
        "never the live database (only localhost is accepted)",
    )


def _is_local_pg_url(url):
    """True only when libpq would connect to this machine.

    Judged on the arguments the psycopg2 dialect hands libpq, not on the URL's
    host alone: the query string can carry host, hostaddr or service, any of
    which sends the connection somewhere else. So can the environment, which
    libpq reads for every setting the URL leaves out: PGHOSTADDR directly, and
    PGSERVICE through a service file that sets hostaddr. No connection is
    opened.
    """
    import os

    from sqlalchemy import create_engine
    from sqlalchemy.engine import make_url

    parsed = make_url(url)
    if parsed.get_backend_name() != "postgresql":
        return False
    if {"host", "hostaddr", "service"} & set(parsed.query):
        return False
    if os.environ.get("PGHOSTADDR") or os.environ.get("PGSERVICE"):
        return False
    engine = create_engine(parsed)
    try:
        _, kwargs = engine.dialect.create_connect_args(parsed)
    finally:
        engine.dispose()
    return kwargs.get("host") in ("localhost", "127.0.0.1")


@pytest.fixture
def is_local_pg_url():
    """The integration suite's local-only check, for testing it."""
    return _is_local_pg_url


@pytest.fixture
def install_engine(monkeypatch):
    """Route every get_engine call to the given engine (integration tests)."""

    def install(engine):
        _replace_get_engine(monkeypatch, lambda: engine)

    return install


ROOT = Path(__file__).resolve().parents[1]

# The tables the write paths touch. schema_roles.sql is left out: it creates
# cluster-wide roles, which a per-test schema cannot contain.
SCHEMA_FILES = ("schema.sql", "schema_sentiment.sql", "schema_trending.sql")


def _schema_engine(url, schema, **settings):
    """An engine whose sessions use the test schema. The lock timeout makes a
    stuck test fail rather than hang the suite."""
    from sqlalchemy import create_engine

    options = {"search_path": schema, "lock_timeout": "15s"} | settings
    return create_engine(
        url,
        connect_args={"options": " ".join(f"-c{k}={v}" for k, v in options.items())},
    )


@pytest.fixture
def schema_engine():
    """_schema_engine, for a test that needs a second session's settings."""
    return _schema_engine


@pytest.fixture
def pg_url(request, monkeypatch):
    """The disposable database's URL, after the local-only checks."""
    url = request.config.getoption("--pg-url")
    if not url:
        pytest.skip("needs --pg-url pointing at a disposable local Postgres")
    if not _is_local_pg_url(url):
        pytest.fail("--pg-url must be a local database, never the live one")
    # Whatever .env's DB_HOST is, code that judges the database's locality
    # (db.readonly) sees this one: a read it took for the live database
    # would log in as the read-only role on an engine of its own, around
    # the test engine installed by pg.
    from quantflow import config

    monkeypatch.setattr(config, "DB_HOST", "127.0.0.1")
    return url


@contextmanager
def _scratch_schema(url, prefix="qf_scratch", **settings):
    """An empty schema of its own on url and an engine on it (with
    _schema_engine's settings), dropped with everything in it after."""
    from sqlalchemy import create_engine

    schema = f"{prefix}_{uuid.uuid4().hex[:8]}"
    admin = create_engine(url)
    with admin.begin() as conn:
        conn.exec_driver_sql(f"CREATE SCHEMA {schema}")
    engine = _schema_engine(url, schema, **settings)
    try:
        yield engine, schema
    finally:
        engine.dispose()
        with admin.begin() as conn:
            conn.exec_driver_sql(f"DROP SCHEMA {schema} CASCADE")
        admin.dispose()


@pytest.fixture
def scratch_schema(pg_url):
    """_scratch_schema on the disposable database: with scratch_schema() as
    (engine, schema): ..."""
    return functools.partial(_scratch_schema, pg_url)


@pytest.fixture
def pg(pg_url, install_engine):
    with _scratch_schema(pg_url, "qf_test") as (engine, _):
        with engine.begin() as conn:
            for name in SCHEMA_FILES:
                conn.exec_driver_sql((ROOT / "sql" / name).read_text(encoding="utf-8"))
        install_engine(engine)
        yield engine


@pytest.fixture(autouse=True)
def _no_database(monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError(
            "unit tests are offline: use the fake_db fixture, never a real database"
        )

    _replace_get_engine(monkeypatch, refuse)
    # The second way to an engine: db.readonly's, for the read-only role.
    # Never installed by install_engine either, so no test reaches it.
    import quantflow.db.connection as connection

    monkeypatch.setattr(connection, "engine_as", refuse)
    for name, module in list(sys.modules.items()):
        if name.startswith("quantflow.") and getattr(module, "engine_as", None):
            monkeypatch.setattr(module, "engine_as", refuse)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    import requests
    import yfinance
    import yfinance.data

    def refuse(*args, **kwargs):
        raise RuntimeError("tests are offline: fake this network call")

    monkeypatch.setattr(yfinance, "download", refuse)
    # yfinance sends its own requests through curl_cffi, not requests, so
    # refusing download and requests alone let yf.Ticker(...).splits and every
    # other yfinance call reach Yahoo. Ticker is refused outright, and
    # YfData._make_request, which every yfinance request goes through (as of
    # 1.7.0), catches the other ways in. Should an upgrade rename it, this
    # setattr fails every test rather than letting requests through.
    monkeypatch.setattr(yfinance, "Ticker", refuse)
    monkeypatch.setattr(yfinance.data.YfData, "_make_request", refuse)
    monkeypatch.setattr(requests.Session, "request", refuse)


@pytest.fixture(scope="session")
def _empty_acknowledged_list(tmp_path_factory):
    """Shared by every test, so never written to: a test that needs entries
    writes a file of its own."""
    path = tmp_path_factory.mktemp("acknowledged") / "split_jump_acknowledged.txt"
    path.write_text(
        "# The unit suite's list: no acknowledged moves.\n", encoding="utf-8"
    )
    return path


@pytest.fixture(autouse=True)
def _no_committed_acknowledgements(monkeypatch, _empty_acknowledged_list):
    import quantflow.config as config

    monkeypatch.setattr(
        config, "SPLIT_JUMP_ACKNOWLEDGED_FILE", _empty_acknowledged_list
    )


@pytest.fixture
def fake_db(monkeypatch):
    db = FakeDatabase()
    engine = _FakeEngine(db)
    _replace_get_engine(monkeypatch, lambda: engine)
    return db


class PriceBars:
    """raw_prices rows in the shapes yfinance stores them."""

    EXCHANGE_TZ = "America/New_York"

    def daily(self, day, close, *, open_=None, high=None, low=None, volume=1e6):
        """A daily bar. yfinance returns these as naive midnights, which a
        UTC session stores at 00:00 UTC."""
        return {
            "ts": pd.Timestamp(day, tz="UTC"),
            "open": close if open_ is None else open_,
            "high": close + 1.0 if high is None else high,
            "low": close - 1.0 if low is None else low,
            "close": close,
            "volume": volume,
        }

    def session(self, day, closes, *, volume=1000.0):
        """15-min bars from 09:30 New York, one per close given."""
        start = pd.Timestamp(f"{day} 09:30", tz=self.EXCHANGE_TZ)
        return [
            {
                "ts": (start + pd.Timedelta(minutes=15 * i)).tz_convert("UTC"),
                "open": close - 0.05,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": volume,
            }
            for i, close in enumerate(closes)
        ]

    def frame(self, *groups):
        """raw_prices as Postgres returns them, ordered by ts."""
        rows = []
        for group in groups:
            rows.extend(group if isinstance(group, list) else [group])
        return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)


@pytest.fixture
def price_bars():
    return PriceBars()


@pytest.fixture
def read_tables(monkeypatch, fake_db):
    """Serve canned frames to pd.read_sql, picked by the table a query reads.

    Assign frames by table name: read_tables["raw_prices"] = frame, or a
    function of the query's params when the rows depend on them (one ticker's
    rows, say). A query against a table with no canned frame fails the test
    rather than silently returning nothing.
    """
    served = {}

    def fake_read_sql(sql, con=None, params=None, **kwargs):
        query = str(sql)
        for name, frame in served.items():
            if re.search(rf"\bFROM\s+{name}\b", query, re.IGNORECASE):
                return (frame(params or {}) if callable(frame) else frame).copy()
        raise AssertionError(f"no canned frame for query:\n{query}")

    monkeypatch.setattr(pd, "read_sql", fake_read_sql)
    return served
