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
  fake_db                 an in-memory stand-in that records every statement
                          and applies INSERT ... ON CONFLICT DO NOTHING the way
                          Postgres does, so a write path can be checked for what
                          it reports against what it actually wrote.
  read_tables             canned pd.read_sql results by table name, for code
                          that reads through pandas.
  price_bars              raw_prices rows in the two shapes yfinance stores:
                          daily bars at 00:00 UTC and 15-min bars on the New
                          York session grid.

Integration support (tests/integration, a disposable local Postgres):

  --pg-url                the database to use; nothing runs without it.
  is_local_pg_url         the check that it can only reach this machine.
  install_engine          routes every get_engine call to the test's engine.
"""

import copy
import re
import sys
from collections import defaultdict
from contextlib import contextmanager

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
}

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
        written = 0
        for row in rows:
            key = tuple(row.get(c) for c in key_cols)
            if key_cols and key in stored:
                if skip_conflicts:
                    continue
                raise IntegrityError(sql, row, Exception("duplicate key value"))
            stored.add(key)
            self.tables[table].append(dict(row))
            written += 1
        return FakeResult(written)

    def _delete(self, sql, rows):
        match = re.search(r"DELETE FROM\s+(\w+)", sql, re.IGNORECASE)
        assert match, f"cannot find the target table in: {sql}"
        table = match.group(1)
        tickers = (rows[0] if rows else {}).get("tickers")
        if tickers is None:
            return FakeResult(0)
        before = len(self.tables[table])
        self.tables[table] = [
            r for r in self.tables[table] if r.get("ticker") not in tickers
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


@pytest.fixture(autouse=True)
def _no_database(monkeypatch):
    def refuse():
        raise RuntimeError(
            "unit tests are offline: use the fake_db fixture, never a real database"
        )

    _replace_get_engine(monkeypatch, refuse)


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
        """raw_prices as Postgres returns them, ordered by ts.

        Carries every column a loader projection could ask for: the bar
        itself, plus the `ts::date AS date`, `ts::date AS ds` and
        `close AS y` aliases that the loaders selected before the daily-bar
        fix. Those aliases are what lets the regression tests fail against the
        pre-fix loaders rather than error out on a missing column.
        """
        rows = []
        for group in groups:
            rows.extend(group if isinstance(group, list) else [group])
        df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
        utc_day = df["ts"].dt.tz_convert("UTC").dt.date
        return df.assign(date=utc_day, ds=utc_day, y=df["close"])


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
