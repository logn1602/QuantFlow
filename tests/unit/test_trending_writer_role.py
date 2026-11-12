"""tests/unit/test_trending_writer_role.py

sql/schema_roles.sql, steps 11 to 14, read as text: quantflow_trending, the
trending tier's writer, holds exactly the commands one run uses, on the five
trending tables only, each with its policy; the lists come from the code
that runs them (db/trending.py). That a whole run works as the role is
checked on Postgres in tests/integration/test_trending_on_postgres.py.

Apart from tests/unit/test_schema_roles.py on purpose: the daily refresh
collects that module, and pytest imports a module before it deselects any
test, so a trending import or file read there could stop the core's data
steps.
"""

import re
from pathlib import Path

import pytest

from quantflow.db import trending as db_trending

pytestmark = pytest.mark.trending

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


def _statements(sql):
    """Each statement, comments dropped and whitespace made single spaces."""
    without_comments = re.sub(r"--[^\n]*", "", sql)
    return [
        " ".join(statement.split())
        for statement in without_comments.split(";")
        if statement.strip()
    ]


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
