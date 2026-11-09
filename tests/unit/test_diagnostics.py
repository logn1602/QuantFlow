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


FREE_TIER_SQL = ROOT / "scripts" / "diagnostics" / "free_tier_usage.sql"


@pytest.mark.parametrize("sql_file", [None, FREE_TIER_SQL], ids=["1a", "free_tier"])
def test_the_sql_is_four_queries_that_only_read(diagnostics, sql_file):
    queries = diagnostics.queries(sql_file or diagnostics.SQL_FILE)
    assert len(queries) == 4
    for name, sql in queries.items():
        code = _code_only(sql).strip()
        assert re.match(r"(WITH|SELECT)\b", code), name
        assert not re.search(WRITES, code, re.IGNORECASE), name
        assert ";" not in code.rstrip(";"), f"{name}: more than one statement"


def test_sql_picks_the_query_file_and_a_remote_database_still_needs_live(
    diagnostics, monkeypatch, tmp_path
):
    ran = []
    monkeypatch.setattr(diagnostics, "run", lambda out, sql_file: ran.append(sql_file))
    monkeypatch.setattr(diagnostics.config, "DB_HOST", "db.example.supabase.co")
    out = str(tmp_path / "usage.md")
    assert diagnostics.main(["--sql", str(FREE_TIER_SQL), "--out", out]) == 2
    assert ran == []
    assert diagnostics.main(["--sql", str(FREE_TIER_SQL), "--live", "--out", out]) == 0
    assert diagnostics.main(["--live", "--out", out]) == 0
    assert ran == [FREE_TIER_SQL, diagnostics.SQL_FILE]


BACKSLASH = chr(92)
ESCAPES = {
    "a commit, then a write": "COMMIT; DELETE FROM raw_prices; SELECT 1",
    "a second statement": "SELECT 1; DELETE FROM raw_prices",
    "a write": "DELETE FROM raw_prices",
    "read-write mode": "SET TRANSACTION READ WRITE",
    "a data-modifying CTE": "WITH d AS (DELETE FROM raw_prices RETURNING *) SELECT * FROM d",
    "a select that locks": "SELECT * FROM raw_prices FOR UPDATE",
    # Each hides "; COMMIT; DELETE" inside what a naive scan reads as a string.
    "dollar quoting": "SELECT $$'$$; COMMIT; DELETE FROM raw_prices; SELECT $$'$$",
    "an escape string": f"SELECT E'{BACKSLASH}''; COMMIT; DELETE FROM raw_prices; SELECT ''",
    "a statement after a comment": "SELECT 1 /* ; */ ; COMMIT",
    "an unterminated string": "SELECT 'x; COMMIT",
    # Postgres nests block comments: each of these ends the comment later
    # than a scan that stops at the first "*/", and hides the commit behind
    # a "--" or a quote that Postgres reads as part of the comment.
    "a nested comment hiding a line comment": (
        "SELECT 1 /* /* */ -- */ ; COMMIT ; DELETE FROM raw_prices ; SELECT 1 AS x"
    ),
    "a nested comment hiding a quote": (
        "SELECT 1 /* /* */ ' */ ; COMMIT ; DELETE FROM raw_prices -- '"
    ),
    # A quoted name would hide a function from the check below.
    "a quoted function name": 'SELECT "pg_terminate_backend"(1)',
    # Calls whose effects outlast the read-only transaction.
    "ending another session": "SELECT pg_terminate_backend(pid) FROM pg_stat_activity",
    "lifting the statement timeout": "SELECT set_config('statement_timeout', '0', true)",
    "an advisory lock": "SELECT pg_advisory_lock(1)",
    "a schema-qualified call": "SELECT pg_catalog.pg_cancel_backend (1)",
    "a long sleep": "SELECT pg_sleep(600)",
}


@pytest.mark.parametrize("section", ESCAPES.values(), ids=ESCAPES)
def test_a_query_file_is_refused_unless_each_query_is_one_select(
    diagnostics, monkeypatch, tmp_path, section
):
    """Each query reaches Postgres exactly as written, and Postgres runs every
    statement in it: "COMMIT; DELETE ..." would end the read-only
    transaction and commit the delete. So any file given with --sql is
    checked before connecting, not only the two committed ones."""
    ran = []
    monkeypatch.setattr(diagnostics, "run", lambda out, sql_file: ran.append(sql_file))
    monkeypatch.setattr(diagnostics.config, "DB_HOST", "db.example.supabase.co")
    sql = tmp_path / "escape.sql"
    sql.write_text(f"-- @query escape\n{section}\n", encoding="utf-8")
    out = tmp_path / "out.md"
    assert diagnostics.main(["--sql", str(sql), "--live", "--out", str(out)]) == 2
    assert ran == []
    assert not out.exists()


def test_a_carriage_return_ends_a_line_comment_as_it_does_in_postgres(diagnostics):
    """Postgres ends a "--" comment at a carriage return as well as a line
    feed; a scan that waits for the line feed would read the commit after it
    as comment."""
    sql = f"SELECT 1 -- x{chr(13)}; COMMIT; DELETE FROM raw_prices"
    with pytest.raises(ValueError, match="more than one statement"):
        diagnostics.check_read_only("q", sql)


def test_names_inside_comments_and_strings_are_not_refused(diagnostics):
    """Only code counts: the committed files mention "close" in comments."""
    diagnostics.check_read_only(
        "q", "SELECT 'COMMIT; pg_sleep(1)' AS x -- the \"close\" ; DELETE\n"
    )


@pytest.mark.parametrize(
    "content",
    [None, "SELECT 1\n", "-- @querry size\nSELECT 1\n"],
    ids=["missing file", "no marker", "misspelled marker"],
)
def test_a_query_file_with_no_queries_is_refused_before_connecting(
    diagnostics, monkeypatch, tmp_path, content
):
    """Without a marker the file once parsed to no queries, and the run
    connected and wrote an empty results file."""
    ran = []
    monkeypatch.setattr(diagnostics, "run", lambda out, sql_file: ran.append(sql_file))
    sql = tmp_path / "usage.sql"
    if content is not None:
        sql.write_text(content, encoding="utf-8")
    out = tmp_path / "out.md"
    assert diagnostics.main(["--sql", str(sql), "--live", "--out", str(out)]) == 2
    assert ran == []
    assert not out.exists()


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

    def refuse(out, sql_file):
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
