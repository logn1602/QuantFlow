"""tests/unit/test_readonly.py

Covers db.readonly with a fake engine: the transaction is made read-only
first, checked, timed out, rolled back and disposed, whatever happens.
"""

import pytest

from quantflow.db import readonly


class _Result:
    def __init__(self, value):
        self.value = value

    def scalar_one(self):
        return self.value


class _Txn:
    def __init__(self, log):
        self.log = log

    def rollback(self):
        self.log.append("ROLLBACK")


class _Conn:
    def __init__(self, log, read_only):
        self.log, self.read_only = log, read_only

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def begin(self):
        return _Txn(self.log)

    def exec_driver_sql(self, sql):
        self.log.append(sql)
        return _Result(self.read_only if sql.startswith("SHOW") else None)


class _Engine:
    def __init__(self, read_only="on"):
        self.log: list[str] = []
        self.read_only = read_only

    def connect(self):
        return _Conn(self.log, self.read_only)

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
