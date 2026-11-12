"""tests/unit/test_schema_roles.py

sql/schema_roles.sql read as text. Step 7: row-level security on every core
table, a SELECT-only policy for quantflow_readonly on each, and no privilege
left to Supabase's anon and authenticated roles, in one transaction that
stops rather than queue behind a long read. Without the policies the role
saw every core table of the live database empty (2026-11-11). Step 8, the
trending tables' copy of what sql/migrations/002_trending.sql runs, is
checked in tests/unit/test_trending_schema.py.

Step 10: postgres's default privileges grant anon and authenticated nothing
on what it creates later, as the owner set on the live database. Steps 11
to 14, the trending tier's writer, are checked in
tests/unit/test_trending_writer_role.py: the daily refresh collects this
module, and pytest imports a module before it deselects any test, so
nothing here imports or reads the trending tier.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROLES = (ROOT / "sql" / "schema_roles.sql").read_text(encoding="utf-8")
CORE_SCHEMAS = (
    "schema.sql",
    "schema_sentiment.sql",
    "schema_metrics.sql",
    "schema_backtest.sql",
)
CORE = {
    name
    for schema in CORE_SCHEMAS
    for name in re.findall(
        r"CREATE TABLE IF NOT EXISTS (\w+)",
        (ROOT / "sql" / schema).read_text(encoding="utf-8"),
    )
}
POLICY = (
    "CREATE POLICY quantflow_readonly_read ON {} "
    "FOR SELECT TO quantflow_readonly USING (true)"
)


def _statements(sql):
    """Each statement, comments dropped and whitespace made single spaces."""
    without_comments = re.sub(r"--[^\n]*", "", sql)
    return [
        " ".join(statement.split())
        for statement in without_comments.split(";")
        if statement.strip()
    ]


def _step_7():
    """The statements inside the transaction that holds the core policies."""
    statements = _statements(ROLES)
    first = statements.index(POLICY.format("raw_prices"))
    begin = max(i for i in range(first) if statements[i] == "BEGIN")
    return statements[begin + 1 : statements.index("COMMIT", first)]


def test_the_core_tables_are_the_eight_the_live_database_has():
    assert {
        "raw_prices",
        "technical_indicators",
        "anomalies",
        "forecasts",
        "alerts_log",
        "news_sentiment",
        "model_metrics",
        "backtest_results",
    } == CORE


def test_every_core_table_gets_row_level_security_and_a_select_policy():
    step = _step_7()
    for table in sorted(CORE):
        assert f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY" in step, table
        assert POLICY.format(table) in step, table


def test_step_7_takes_every_privilege_from_the_data_api_roles():
    revokes = [s for s in _step_7() if s.startswith("REVOKE ")]
    listed = re.fullmatch(
        r"REVOKE ALL ON ([\w, ]+) FROM anon, authenticated", revokes[0]
    )
    assert listed and set(listed.group(1).split(", ")) == CORE
    assert revokes[1:] == [
        "REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM anon, authenticated"
    ]


def test_step_7_only_secures_reads_and_revokes_in_one_transaction():
    step = _step_7()
    assert step[0] == "SET LOCAL lock_timeout = '10s'"
    for statement in step[1:]:
        assert statement.startswith(("ALTER TABLE ", "CREATE POLICY ", "REVOKE ")), (
            statement
        )
    assert len(step) == 1 + 2 * len(CORE) + 2
    words = set(re.findall(r"[A-Z]+", " ".join(step)))
    assert not words & {
        "DROP",
        "TRUNCATE",
        "DELETE",
        "INSERT",
        "UPDATE",
        "GRANT",
        "DISABLE",
        "FORCE",
    }


# ── Step 10: what postgres creates later ─────────────────────────────────────

DEFAULT_REVOKES = [
    "ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public "
    f"REVOKE ALL ON {kind} FROM anon, authenticated"
    for kind in ("TABLES", "SEQUENCES", "FUNCTIONS")
]


def test_what_postgres_creates_later_grants_the_data_api_roles_nothing():
    """The three statements the owner ran on the live database on
    2026-11-11: Supabase's default privileges granted anon and
    authenticated everything on each new table, sequence and function, so
    each migration had to revoke it (002 does). Recorded so a fresh install
    gets the same."""
    statements = _statements(ROLES)
    assert DEFAULT_REVOKES[0] in statements
    at = statements.index(DEFAULT_REVOKES[0])
    assert statements[at : at + 3] == DEFAULT_REVOKES


def test_the_only_default_privilege_granted_is_the_read_only_roles_select():
    """A table postgres creates later is readable by quantflow_readonly
    (step 4) and by nobody else, the trending writer included: a new table
    gets its writes from the migration that creates it."""
    granted = [
        s
        for s in _statements(ROLES)
        if s.startswith("ALTER DEFAULT PRIVILEGES") and " GRANT " in s
    ]
    assert granted == [
        "ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public "
        "GRANT SELECT ON TABLES TO quantflow_readonly"
    ]
