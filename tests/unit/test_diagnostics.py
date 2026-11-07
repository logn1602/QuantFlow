"""tests/unit/test_diagnostics.py

Covers scripts/diagnostics/run_1a_diagnostics.py, the read-only queries run
against the live database. What must hold before it is ever pointed there:
the SQL is four queries that only read, the script refuses anything but a
local database unless told --live, and an error never prints the connection
settings.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "diagnostics" / "run_1a_diagnostics.py"
WRITES = r"\b(INSERT|UPDATE|DELETE|MERGE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|COPY|CALL|DO|VACUUM|LOCK)\b"


@pytest.fixture
def diagnostics():
    spec = importlib.util.spec_from_file_location("run_1a_diagnostics", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def _code_only(sql):
    """sql without comments or string literals, where a keyword is only text."""
    sql = re.sub(r"--[^\n]*", "", sql)
    return re.sub(r"'[^']*'", "''", sql)


def test_the_sql_is_four_queries_that_only_read(diagnostics):
    queries = diagnostics.queries()
    assert len(queries) == 4
    for name, sql in queries.items():
        code = _code_only(sql).strip()
        assert re.match(r"(WITH|SELECT)\b", code), name
        assert not re.search(WRITES, code, re.IGNORECASE), name
        assert ";" not in code.rstrip(";"), f"{name}: more than one statement"


def test_a_remote_database_is_refused_without_live(diagnostics, monkeypatch, tmp_path):
    """It must stop before connecting: this suite refuses any real connection,
    so reaching get_engine would fail the run (exit 1), not refuse it (exit 2)."""
    monkeypatch.setattr(diagnostics.config, "DB_HOST", "db.example.supabase.co")
    out = tmp_path / "results.md"
    assert diagnostics.main(["--out", str(out)]) == 2
    assert not out.exists()


def test_an_error_is_printed_with_the_connection_settings_masked(
    diagnostics, monkeypatch, tmp_path, capsys
):
    settings = {
        "DB_HOST": "db.example.supabase.co",
        "DB_USER": "postgres.abcdefgh",
        "DB_PASSWORD": "s3cret-pw",
        "DB_NAME": "quantdb",
    }
    for key, value in settings.items():
        monkeypatch.setattr(diagnostics.config, key, value)

    def refuse(out):
        raise RuntimeError(
            'password authentication failed for user "postgres.abcdefgh" '
            "(s3cret-pw) at db.example.supabase.co/quantdb"
        )

    monkeypatch.setattr(diagnostics, "run", refuse)
    assert diagnostics.main(["--live", "--out", str(tmp_path / "r.md")]) == 1
    err = capsys.readouterr().err
    assert "Failed: RuntimeError" in err
    for value in settings.values():
        assert value not in err
