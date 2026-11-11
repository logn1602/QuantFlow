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
to 14: quantflow_trending, the trending tier's writer, which holds exactly
the commands one run uses, on the five trending tables only, each with its
policy; the lists come from the code that runs them (db/trending.py). That
the run works as it is checked on Postgres in
tests/integration/test_trending_on_postgres.py.
"""

import re
from pathlib import Path

import pytest

from quantflow.db import trending as db_trending

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


# ── Steps 11 to 14: quantflow_trending, the tier's writer ────────────────────

WRITER = "quantflow_trending"
TRENDING = set(db_trending.TRENDING_TABLES)
TICKERS_SCHEMA = (ROOT / "sql" / "schema_trending.sql").read_text(encoding="utf-8")
GRANT = re.compile(
    r"GRANT (?P<privilege>[A-Z]+)(?: \((?P<columns>[\w, ]+)\))? ON "
    r"(?:SEQUENCE )?(?P<objects>[\w, ]+) TO quantflow_trending"
)
WRITER_POLICY = re.compile(
    r"CREATE POLICY (?P<name>\w+) ON (?P<table>\w+) "
    r"FOR (?P<command>[A-Z]+) TO quantflow_trending (?P<rule>.+)"
)


def _writer_statements():
    return [s for s in _statements(ROLES) if re.search(rf"\b{WRITER}\b", s)]


def _step_13():
    """The statements inside the transaction that holds the writer's
    policies."""
    statements = _statements(ROLES)
    first = next(
        i
        for i, s in enumerate(statements)
        if s.startswith("CREATE POLICY ") and f" TO {WRITER} " in s
    )
    begin = max(i for i in range(first) if statements[i] == "BEGIN")
    return statements[begin + 1 : statements.index("COMMIT", first)]


def _granted():
    """(object, privilege, columns or None) for each object a step 13 GRANT
    names."""
    granted = set()
    for statement in _step_13():
        match = GRANT.fullmatch(statement)
        if statement.startswith("GRANT "):
            assert match, statement
        if not match:
            continue
        columns = match["columns"]
        for name in match["objects"].split(", "):
            granted.add(
                (
                    name,
                    match["privilege"],
                    frozenset(columns.split(", ")) if columns else None,
                )
            )
    return granted


def _tickers_columns():
    """The tickers table's columns in sql/schema_trending.sql: the lines of
    its CREATE TABLE that start with a name (a CHECK continues the line
    above)."""
    body = TICKERS_SCHEMA.split("CREATE TABLE IF NOT EXISTS tickers (", 1)[1]
    body = re.sub(r"--[^\n]*", "", body.split(");", 1)[0])
    return set(re.findall(r"^\s*([a-z_0-9]+)\s", body, re.MULTILINE))


@pytest.mark.trending
def test_the_writer_can_log_in_and_reach_the_schema_and_nothing_more():
    """Owns nothing, so row-level security applies to it, and cannot
    bypass it. Its sessions time out a stuck statement and an idle
    transaction rather than hold the run to the job's 30 minutes."""
    statements = _writer_statements()
    assert statements[:5] == [
        f"CREATE ROLE {WRITER} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE "
        "NOREPLICATION NOBYPASSRLS CONNECTION LIMIT 5",
        f"ALTER ROLE {WRITER} SET statement_timeout = '120s'",
        f"ALTER ROLE {WRITER} SET idle_in_transaction_session_timeout = '60s'",
        f"GRANT CONNECT ON DATABASE postgres TO {WRITER}",
        f"GRANT USAGE ON SCHEMA public TO {WRITER}",
    ]
    # Its password is set by hand, never written in the file.
    assert not [s for s in statements if "PASSWORD" in s]
    # Everything else it is given is in step 13.
    step = _step_13()
    assert statements[5:] == [s for s in step if WRITER in s]


@pytest.mark.trending
def test_the_writer_holds_exactly_the_commands_a_run_uses():
    """From db/trending.py: every table is read (the run reads tickers and
    trending_fetches, and an upsert, an UPDATE or a DELETE with a WHERE reads
    the rows it changes) and appended to; tickers rows are upserted, but
    only in the columns the run writes, so it can never set tier or
    promoted_at (by hand only, at a promotion); each symbol's bars are
    deleted and replaced; trending_news draws its ids from its sequence."""
    tickers = frozenset(db_trending.TICKER_COLUMNS)
    assert _granted() == {
        *((name, "SELECT", None) for name in TRENDING),
        *((name, "INSERT", None) for name in TRENDING - {"tickers"}),
        ("tickers", "INSERT", tickers),
        ("tickers", "UPDATE", frozenset(db_trending.UPDATABLE_TICKER_COLUMNS)),
        ("trending_prices", "DELETE", None),
        ("trending_news_id_seq", "USAGE", None),
    }
    # The run writes every tickers column but the two set by hand.
    assert _tickers_columns() == tickers | {"tier", "promoted_at"}


@pytest.mark.trending
def test_each_command_the_writer_holds_has_its_policy_and_no_other():
    """Row-level security is on for all five tables: a command no policy
    allows finds no row (a DELETE or UPDATE) or fails (an INSERT)."""
    policies = {}
    for statement in _step_13():
        if statement.startswith("CREATE POLICY "):
            match = WRITER_POLICY.fullmatch(statement)
            assert match, statement
            key = (match["table"], match["command"])
            assert key not in policies, key
            policies[key] = (match["name"], match["rule"])
    commands = {
        (name, privilege) for name, privilege, _ in _granted() if name in TRENDING
    }
    assert set(policies) == commands
    rules = {
        "SELECT": ("quantflow_trending_read", "USING (true)"),
        "INSERT": ("quantflow_trending_insert", "WITH CHECK (true)"),
        "UPDATE": ("quantflow_trending_update", "USING (true) WITH CHECK (true)"),
        "DELETE": ("quantflow_trending_delete", "USING (true)"),
    }
    for (name, command), policy in policies.items():
        assert policy == rules[command], (name, command)


@pytest.mark.trending
def test_step_13_only_grants_and_creates_policies_on_the_trending_tables():
    """One transaction, which stops rather than queue behind a long read,
    and names no core table: the tier's login can never write one."""
    step = _step_13()
    assert step[0] == "SET LOCAL lock_timeout = '10s'"
    for statement in step[1:]:
        assert statement.startswith(("GRANT ", "CREATE POLICY ")), statement
        assert WRITER in statement, statement
    words = set(re.findall(r"\w+", " ".join(_writer_statements())))
    assert not words & CORE
    assert not words & {"ALL", "TRUNCATE", "REFERENCES", "TRIGGER", "BYPASSRLS"}
