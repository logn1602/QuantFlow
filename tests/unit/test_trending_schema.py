"""tests/unit/test_trending_schema.py

The trending tier's DDL, read as text (docs/trending.md, section 6): the
migration the live database gets holds the schema file's statements (the
tables, their indexes, row-level security on each), then the access ones,
which name roles and so stay out of make setup: a SELECT policy for
quantflow_readonly on each table, and no privilege for Supabase's anon and
authenticated roles. sql/schema_roles.sql, step 8, holds the same access
statements for a fresh install. The migration can only create, secure and
revoke, in one transaction. Whether it runs is checked on Postgres in
tests/integration/test_trending_on_postgres.py.
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
ROLES = (ROOT / "sql" / "schema_roles.sql").read_text(encoding="utf-8")
# In the order the files create them.
ORDER = (
    "tickers",
    "trending_prices",
    "trending_snapshots",
    "trending_fetches",
    "trending_news",
)
TABLES = set(ORDER)
POLICY = (
    "CREATE POLICY quantflow_readonly_read ON {} "
    "FOR SELECT TO quantflow_readonly USING (true)"
)
ACCESS = [
    *(POLICY.format(table) for table in ORDER),
    "REVOKE ALL ON tickers, trending_prices, trending_snapshots, trending_fetches, "
    "trending_news FROM anon, authenticated",
    "REVOKE ALL ON SEQUENCE trending_news_id_seq FROM anon, authenticated",
]


def _statements(sql):
    """Each statement, comments dropped and whitespace made single spaces."""
    without_comments = re.sub(r"--[^\n]*", "", sql)
    return [
        " ".join(statement.split())
        for statement in without_comments.split(";")
        if statement.strip()
    ]


def test_the_migration_holds_the_schema_files_statements_then_the_access_ones():
    migration = _statements(MIGRATION)
    assert migration[0] == "BEGIN" and migration[-1] == "COMMIT"
    assert migration[1:-1] == _statements(SCHEMA) + ACCESS


def test_a_fresh_install_gets_the_same_access_from_schema_roles_step_8():
    """make setup cannot name roles a fresh database may not have: its
    tables come up with row-level security on and no policy, and step 8
    of sql/schema_roles.sql, run after it, adds what the migration does."""
    statements = _statements(ROLES)
    first = statements.index(POLICY.format("tickers"))
    begin = max(i for i in range(first) if statements[i] == "BEGIN")
    assert statements[begin + 1 : statements.index("COMMIT", first)] == ACCESS


def test_the_schema_file_turns_row_level_security_on_and_names_no_role():
    schema = _statements(SCHEMA)
    for table in ORDER:
        assert f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY" in schema, table
    words = set(re.findall(r"\w+", " ".join(schema)))
    assert not words & {"quantflow_readonly", "anon", "authenticated", "POLICY"}


# Every statement the migration may hold: creating, row-level security,
# the read policy and the revokes, nothing that changes or removes.
ALLOWED = (
    re.compile(r"CREATE TABLE IF NOT EXISTS \w+ \(.+\)"),
    re.compile(r"CREATE INDEX IF NOT EXISTS \w+ ON \w+ \(.+\)"),
    re.compile(r"ALTER TABLE \w+ ENABLE ROW LEVEL SECURITY"),
    re.compile(
        r"CREATE POLICY quantflow_readonly_read ON \w+ "
        r"FOR SELECT TO quantflow_readonly USING \(true\)"
    ),
    re.compile(r"REVOKE ALL ON (SEQUENCE )?[\w, ]+ FROM anon, authenticated"),
)


def test_the_migration_only_creates_secures_and_revokes():
    body = _statements(MIGRATION)[1:-1]
    assert body, "no statements found"
    for statement in body:
        assert any(form.fullmatch(statement) for form in ALLOWED), statement
    words = set(re.findall(r"[A-Z]+", " ".join(body)))
    assert not words & {
        "DROP",
        "TRUNCATE",
        "DELETE",
        "UPDATE",
        "INSERT",
        "GRANT",
        "DISABLE",
        "FORCE",
    }


def test_the_migration_creates_the_five_tables_and_no_core_one():
    created = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", MIGRATION))
    assert created == TABLES
    indexed = set(re.findall(r"CREATE INDEX IF NOT EXISTS \w+\s+ON (\w+)", MIGRATION))
    assert indexed <= TABLES
    body = " ;".join(_statements(MIGRATION))
    secured = set(re.findall(r"ALTER TABLE (\w+) ENABLE", body))
    policed = set(re.findall(r"CREATE POLICY \w+ ON (\w+)", body))
    assert secured == policed == TABLES
    revoked = re.findall(r"REVOKE ALL ON (?:SEQUENCE )?([\w, ]+) FROM", body)
    assert {name for names in revoked for name in names.split(", ")} == TABLES | {
        "trending_news_id_seq"
    }


def test_make_setup_creates_the_trending_tables():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    setup = makefile.split("\nsetup:", 1)[1].split("\n\n", 1)[0]
    assert "-f sql/schema_trending.sql" in setup


def test_no_percent_sign_in_the_ddl():
    """The integration fixture runs the files through the driver, which reads
    "%" as a placeholder (as schema_metrics.sql's comments show)."""
    assert "%" not in MIGRATION and "%" not in SCHEMA
