"""tests/unit/test_trending_schema.py

The trending tier's DDL, read as text (docs/trending.md, section 6): the
migration the live database gets and the schema file make setup runs hold
the same statements, and the migration can only create, so running it twice,
or on a database that already has the tables, changes nothing. Whether it
runs is checked on Postgres in tests/integration/test_trending_on_postgres.py.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.trending

ROOT = Path(__file__).resolve().parents[2]
MIGRATION = (ROOT / "sql" / "migrations" / "002_trending.sql").read_text(
    encoding="utf-8"
)
SCHEMA = (ROOT / "sql" / "schema_trending.sql").read_text(encoding="utf-8")
TABLES = {
    "tickers",
    "trending_prices",
    "trending_snapshots",
    "trending_fetches",
    "trending_news",
}


def _statements(sql):
    """Each statement, comments dropped and whitespace made single spaces."""
    without_comments = re.sub(r"--[^\n]*", "", sql)
    return [
        " ".join(statement.split())
        for statement in without_comments.split(";")
        if statement.strip()
    ]


def test_the_migration_and_the_schema_file_hold_the_same_statements():
    migration = _statements(MIGRATION)
    assert migration[0] == "BEGIN" and migration[-1] == "COMMIT"
    assert migration[1:-1] == _statements(SCHEMA)


def test_the_migration_only_creates_and_every_create_is_idempotent():
    body = _statements(MIGRATION)[1:-1]
    assert body, "no statements found"
    for statement in body:
        assert statement.startswith(
            ("CREATE TABLE IF NOT EXISTS ", "CREATE INDEX IF NOT EXISTS ")
        ), statement
    words = set(re.findall(r"[A-Z]+", " ".join(body)))
    assert not words & {"DROP", "TRUNCATE", "DELETE", "ALTER", "UPDATE", "INSERT"}


def test_the_migration_creates_the_five_tables_and_no_core_one():
    created = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", MIGRATION))
    assert created == TABLES
    indexed = set(re.findall(r"CREATE INDEX IF NOT EXISTS \w+\s+ON (\w+)", MIGRATION))
    assert indexed <= TABLES


def test_make_setup_creates_the_trending_tables():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    setup = makefile.split("\nsetup:", 1)[1].split("\n\n", 1)[0]
    assert "-f sql/schema_trending.sql" in setup


def test_no_percent_sign_in_the_ddl():
    """The integration fixture runs the files through the driver, which reads
    "%" as a placeholder (as schema_metrics.sql's comments show)."""
    assert "%" not in MIGRATION and "%" not in SCHEMA
