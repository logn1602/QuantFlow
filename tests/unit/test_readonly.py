"""tests/unit/test_readonly.py

Covers db.readonly with a fake engine: the transaction is made read-only
first, checked, timed out, rolled back and disposed, whatever happens; and a
non-local database is read only as the read-only role (sql/schema_roles.sql),
never with the pipeline's credentials, which can write; and a table the
caller expects rows in, found empty, stops the read (require_rows).
"""

import re
from pathlib import Path

import pytest

import quantflow.config as config
from quantflow.db import readonly


@pytest.fixture(autouse=True)
def local_database(monkeypatch):
    """A local database unless a test says otherwise, whatever .env sets."""
    monkeypatch.setattr(config, "DB_HOST", "127.0.0.1")
    monkeypatch.setattr(config, "DB_READONLY_USER", "")
    monkeypatch.setattr(config, "DB_READONLY_PASSWORD", "")


class _Result:
    def __init__(self, value, rows=()):
        self.value, self.rows = value, list(rows)

    def scalar_one(self):
        return self.value

    def all(self):
        return self.rows


class _Txn:
    def __init__(self, log):
        self.log = log

    def rollback(self):
        self.log.append("ROLLBACK")


class _Conn:
    def __init__(self, log, read_only, tables):
        self.log, self.read_only, self.tables = log, read_only, tables

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def begin(self):
        return _Txn(self.log)

    def exec_driver_sql(self, sql):
        self.log.append(sql)
        if sql.startswith("SHOW"):
            return _Result(self.read_only)
        if "row_security_active" in sql:
            # require_rows' one query: per table, whether a row is visible
            # and whether row-level security applies to the role there.
            names = re.findall(r"SELECT '(\w+)'", sql)
            return _Result(None, [(name, *self.tables[name]) for name in names])
        if sql == "SELECT current_user":
            return _Result("quantflow_readonly")
        return _Result(None)


class _Engine:
    def __init__(self, read_only="on", tables=None):
        self.log: list[str] = []
        self.read_only = read_only
        self.tables = tables or {}

    def connect(self):
        return _Conn(self.log, self.read_only, self.tables)

    def dispose(self):
        self.log.append("DISPOSE")


def test_the_transaction_is_read_only_before_anything_runs(monkeypatch):
    engine = _Engine()
    monkeypatch.setattr(readonly, "get_engine", lambda: engine)
    with readonly.read_only_connection() as conn:
        conn.exec_driver_sql("SELECT 1")
    assert engine.log[0] == "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
    assert engine.log[1] == "SHOW transaction_read_only"
    assert engine.log[2] == "SET LOCAL statement_timeout = '120s'"
    assert engine.log[3:] == ["SELECT 1", "ROLLBACK", "DISPOSE"]


def test_a_writable_transaction_stops_before_any_query(monkeypatch):
    engine = _Engine(read_only="off")
    monkeypatch.setattr(readonly, "get_engine", lambda: engine)
    with (
        pytest.raises(RuntimeError, match="transaction_read_only is 'off'"),
        readonly.read_only_connection(),
    ):
        pytest.fail("the body must not run")
    assert engine.log[-2:] == ["ROLLBACK", "DISPOSE"]


def test_an_error_inside_still_rolls_back(monkeypatch):
    engine = _Engine()
    monkeypatch.setattr(readonly, "get_engine", lambda: engine)
    with pytest.raises(ValueError), readonly.read_only_connection():
        raise ValueError("boom")
    assert engine.log[-2:] == ["ROLLBACK", "DISPOSE"]


# ── Tables that always hold rows (require_rows) ───────────────────────────────


def test_a_table_found_empty_where_rows_are_expected_stops_the_read(monkeypatch):
    """Row-level security with no policy for a role shows it every row
    hidden, and no error: quantflow_readonly saw every core table of the
    live database empty (2026-11-11). A read that expects rows stops before
    its caller reads anything, and says why."""
    engine = _Engine(tables={"raw_prices": (False, True), "anomalies": (True, True)})
    monkeypatch.setattr(readonly, "get_engine", lambda: engine)
    with (
        pytest.raises(readonly.EmptyTablesError) as raised,
        readonly.read_only_connection(expect_rows=["raw_prices", "anomalies"]),
    ):
        pytest.fail("the body must not run")
    message = str(raised.value)
    assert message.startswith("raw_prices: no row visible to quantflow_readonly")
    assert "row-level security" in message.lower()
    assert "sql/schema_roles.sql, step 7" in message
    assert "anomalies" not in message
    check = next(i for i, sql in enumerate(engine.log) if "row_security_active" in sql)
    assert check > engine.log.index("SHOW transaction_read_only")
    assert engine.log[-2:] == ["ROLLBACK", "DISPOSE"]


def test_an_empty_table_with_no_row_level_security_is_called_empty(monkeypatch):
    engine = _Engine(tables={"forecasts": (False, False)})
    monkeypatch.setattr(readonly, "get_engine", lambda: engine)
    with (
        pytest.raises(
            readonly.EmptyTablesError,
            match=r"^forecasts: empty, with no row-level security in the way",
        ),
        readonly.read_only_connection(expect_rows=["forecasts"]),
    ):
        pytest.fail("the body must not run")


def test_tables_that_show_rows_let_the_read_go_on(monkeypatch):
    engine = _Engine(
        tables={"raw_prices": (True, True), "news_sentiment": (True, False)}
    )
    monkeypatch.setattr(readonly, "get_engine", lambda: engine)
    expected = ["raw_prices", "news_sentiment"]
    with readonly.read_only_connection(expect_rows=expected) as conn:
        conn.exec_driver_sql("SELECT 1")
    assert engine.log[-3:] == ["SELECT 1", "ROLLBACK", "DISPOSE"]


def test_no_tables_expected_means_no_check(monkeypatch):
    engine = _Engine()
    monkeypatch.setattr(readonly, "get_engine", lambda: engine)
    with readonly.read_only_connection():
        pass
    assert not any("row_security_active" in sql for sql in engine.log)


def test_only_plain_table_names_reach_the_check(monkeypatch):
    engine = _Engine()
    monkeypatch.setattr(readonly, "get_engine", lambda: engine)
    with (
        pytest.raises(ValueError, match="not a plain table name"),
        readonly.read_only_connection(expect_rows=["raw_prices; DROP TABLE x"]),
    ):
        pytest.fail("the body must not run")
    assert not any("DROP" in sql for sql in engine.log)


CORE_SCHEMAS = (
    "schema.sql",
    "schema_sentiment.sql",
    "schema_metrics.sql",
    "schema_backtest.sql",
)


def test_the_tables_expected_to_hold_rows_are_the_core_ones_less_alerts_log():
    """Every core table a daily refresh fills; alerts_log holds a row only
    once an alert fires. A new core table fails this until it is placed."""
    sql = Path(__file__).resolve().parents[2] / "sql"
    core = set()
    for name in CORE_SCHEMAS:
        text = (sql / name).read_text(encoding="utf-8")
        core |= set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", text))
    assert core - {"alerts_log"} == readonly.ROWS_EXPECTED


def test_only_this_machine_counts_as_local(monkeypatch):
    import quantflow.config as config

    monkeypatch.setattr(config, "DB_HOST", "127.0.0.1")
    assert readonly.is_local_database()
    monkeypatch.setattr(config, "DB_HOST", "aws-0.pooler.supabase.com")
    assert not readonly.is_local_database()


def test_errors_are_masked(monkeypatch):
    import quantflow.config as config

    monkeypatch.setattr(config, "DB_PASSWORD", "pw123")
    monkeypatch.setattr(config, "DB_USER", "postgres.xyz")
    message = readonly.masked('auth failed for "postgres.xyz" with pw123')
    assert "pw123" not in message and "postgres.xyz" not in message


# ── The live database, read as the read-only role ─────────────────────────────

REMOTE = "aws-0.pooler.supabase.com"


@pytest.fixture
def remote(monkeypatch):
    monkeypatch.setattr(config, "DB_HOST", REMOTE)
    return monkeypatch


def _with_role(monkeypatch):
    monkeypatch.setattr(config, "DB_READONLY_USER", "quantflow_readonly.abcdefgh")
    monkeypatch.setattr(config, "DB_READONLY_PASSWORD", "ro-secret")


def test_a_remote_database_is_refused_without_the_read_only_role(remote):
    """The pipeline's credentials can write; a read of the live database
    must not depend on every query being checked right."""
    with (
        pytest.raises(readonly.NoReadOnlyRoleError, match="DB_READONLY_USER"),
        readonly.read_only_connection(),
    ):
        pytest.fail("the body must not run")


def test_a_remote_database_is_read_as_the_read_only_role(remote):
    _with_role(remote)
    engine = _Engine()
    logins = []

    def engine_as(user, password):
        logins.append((user, password))
        return engine

    remote.setattr(readonly, "engine_as", engine_as)
    with readonly.read_only_connection() as conn:
        conn.exec_driver_sql("SELECT 1")

    assert logins == [("quantflow_readonly.abcdefgh", "ro-secret")]
    assert engine.log[0].startswith("SET TRANSACTION")
    assert engine.log[-2:] == ["ROLLBACK", "DISPOSE"]


def test_live_needs_the_flag_and_the_read_only_role(remote):
    assert "--live" in readonly.live_refusal(live=False)
    assert "DB_READONLY_USER" in readonly.live_refusal(live=True)
    _with_role(remote)
    assert readonly.live_refusal(live=True) is None
    remote.setattr(config, "DB_HOST", "127.0.0.1")
    assert readonly.live_refusal(live=False) is None


def test_the_read_only_role_is_masked_in_errors(remote):
    _with_role(remote)
    message = readonly.masked(
        'password authentication failed for user "quantflow_readonly.abcdefgh" '
        "(ro-secret)"
    )
    assert "ro-secret" not in message and "quantflow_readonly" not in message
