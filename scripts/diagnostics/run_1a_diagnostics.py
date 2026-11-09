"""
scripts/diagnostics/run_1a_diagnostics.py
-----------------------------------------
Run the four daily-close diagnostics (1a_diagnostics.sql), or another query
file given with --sql, in one read-only transaction and write the results as
Markdown tables. free_tier_usage.sql measures the database against the free
tier's limits (docs/architecture.md, "Running on free tiers").

Read-only four ways:
  - before connecting, every query in the file must be one SELECT (or WITH)
    statement written in a plain subset of SQL (check_read_only): no second
    statement; no block comment, quoted identifier, dollar quote or
    backslash, where Postgres's reading and a simple scan's can differ; no
    write, lock or transaction keyword; and no call to a function whose
    effects outlast the transaction. Postgres runs every statement a query
    string holds, so "COMMIT; DELETE ..." would otherwise end the read-only
    transaction and commit the delete;
  - the transaction is opened READ ONLY, so Postgres rejects any write, and
    the script checks transaction_read_only is on before running anything;
  - after every query, the script checks the transaction is still open and
    read-only, and fails if not;
  - the transaction is rolled back at the end, whatever happens.

These protect the database's rows. The check is a scan, not Postgres's own
parser, so the guarantee that does not depend on it is a database role that
can only SELECT (sql/schema_roles.sql): --live runs log in as it
(DB_READONLY_USER and DB_READONLY_PASSWORD in .env), never with the
pipeline's credentials, and refuse to run without it (db.readonly).

A local database is read with the connection settings in .env. Nothing here
prints the URL, host, user or password, and an error message is printed with
those values masked.

A non-local database is refused unless --live is given, so a trial run against
a throwaway local cluster cannot reach the live one by mistake.

Usage, from the repo root:
    uv run python scripts/diagnostics/run_1a_diagnostics.py --out results.md
    uv run python scripts/diagnostics/run_1a_diagnostics.py --live --out results.md
    uv run python scripts/diagnostics/run_1a_diagnostics.py --sql scripts/diagnostics/free_tier_usage.sql --live --out usage.md

The first refuses anything but a local database; --live allows the one in .env,
read as the read-only role.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from psycopg2 import extensions

from quantflow import config  # noqa: F401  (tests patch its settings through here)
from quantflow.db.readonly import (
    is_local_database,
    live_refusal,
    masked,
    read_only_connection,
)

SQL_FILE = Path(__file__).with_name("1a_diagnostics.sql")

# Words no diagnostic needs: writes, row locks (FOR UPDATE, FOR SHARE),
# SELECT INTO, and statements that end or change a transaction. Matched as
# whole words in the query's code, comments and quoted text removed.
FORBIDDEN = re.compile(
    "(?<![A-Za-z0-9_])("
    "INSERT|UPDATE|DELETE|MERGE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|COPY|"
    "CALL|DO|VACUUM|LOCK|INTO|SHARE|COMMIT|ROLLBACK|ABORT|BEGIN|SAVEPOINT|"
    "RELEASE|PREPARE|EXECUTE|SET|RESET|DISCARD|LISTEN|NOTIFY|REFRESH|CLUSTER|"
    "REINDEX|COMMENT|SECURITY|IMPORT|LOAD|CHECKPOINT"
    ")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

# Functions a SELECT can call whose effects outlast the read-only transaction
# or reach past this database: ending or signalling other sessions, session
# locks and settings, replication slots, server files, and extensions that
# open their own connections. Matched as a name followed by "(".
SIDE_EFFECTS = re.compile(
    r"(?<![A-Za-z0-9_])("
    r"pg_terminate_backend|pg_cancel_backend|pg_advisory\w*|pg_try_advisory\w*|"
    r"set_config|pg_sleep\w*|pg_create\w*|pg_drop\w*|pg_replication\w*|"
    r"pg_logical\w*|pg_stat_reset\w*|pg_notify|pg_reload_conf|pg_rotate_logfile|"
    r"pg_switch_wal|pg_promote|pg_backup\w*|pg_read\w*|pg_ls\w*|pg_stat_file|"
    r"pg_file\w*|lo_\w*|dblink\w*|http\w*|net\.\w+"
    r")\s*\(",
    re.IGNORECASE,
)


def _code(sql: str) -> str:
    """sql with comments and quoted text blanked out, scanned left to right
    as Postgres reads it, so a keyword or ";" inside a string or a comment is
    not mistaken for code, nor code for a string. Constructs where Postgres's
    lexer and a simple scan part ways are refused instead of followed: block
    comments (Postgres nests them, so it can end one later than a scan does,
    and a "--" or quote the scan then sees hides real code after it) and
    quoted identifiers (which would hide a function name from the checks)."""
    out, i = [], 0
    while i < len(sql):
        if sql.startswith("--", i):
            # Postgres ends a line comment at a carriage return or a line feed.
            ends = [e for e in (sql.find("\n", i), sql.find("\r", i)) if e >= 0]
            i = min(ends) if ends else len(sql)
            out.append(" ")
        elif sql.startswith("/*", i):
            raise ValueError("a block comment (use -- comments)")
        elif sql[i] == '"':
            raise ValueError("a quoted identifier")
        elif sql[i] == "'":
            j = i + 1
            while True:
                end = sql.find("'", j)
                if end < 0:
                    raise ValueError("unterminated quoted text")
                if sql.startswith("''", end):
                    j = end + 2
                    continue
                break
            i = end + 1
            out.append("''")
        else:
            out.append(sql[i])
            i += 1
    return "".join(out)


def check_read_only(name: str, sql: str) -> None:
    """Raises ValueError unless sql is one SELECT (or WITH) statement in the
    plain subset _code accepts that neither writes, locks, touches the
    transaction nor calls a function in SIDE_EFFECTS. A "$" (dollar quoting)
    or a backslash (escape strings) is refused anywhere in the text: either
    can make what looks like a string to a simple scan run as code."""
    if "$" in sql or chr(92) in sql:
        raise ValueError(f"query {name!r}: a dollar sign or a backslash")
    try:
        code = _code(sql).strip()
    except ValueError as e:
        raise ValueError(f"query {name!r}: {e}") from None
    if not re.match("(SELECT|WITH)(?![A-Za-z0-9_])", code, re.IGNORECASE):
        raise ValueError(f"query {name!r}: not a SELECT")
    if ";" in code.rstrip(";").rstrip():
        raise ValueError(f"query {name!r}: more than one statement")
    found = FORBIDDEN.search(code)
    if found:
        raise ValueError(f"query {name!r}: {found.group(1).upper()} is not allowed")
    called = SIDE_EFFECTS.search(code)
    if called:
        raise ValueError(f"query {name!r}: calls {called.group(1)}, not allowed")


def queries(sql_file: Path = SQL_FILE) -> dict[str, str]:
    """The SQL file split on its "-- @query <name>" markers, each query
    checked by check_read_only."""
    text = sql_file.read_text(encoding="utf-8")
    parts = re.split(r"^-- @query (.+)$", text, flags=re.MULTILINE)
    found = {
        name.strip(): sql.strip()
        for name, sql in zip(parts[1::2], parts[2::2], strict=True)
    }
    if not found:
        raise ValueError(f"{sql_file.name}: no -- @query sections")
    for name, sql in found.items():
        check_read_only(name, sql)
    return found


def _still_read_only(dbapi, name: str) -> None:
    """Fails if the transaction ended or stopped being read-only during the
    query just run. check_read_only should make that impossible; this is the
    check behind it. It cannot undo anything already committed."""
    status = dbapi.get_transaction_status()
    cursor = dbapi.cursor()
    try:
        cursor.execute("SHOW transaction_read_only")
        read_only = cursor.fetchone()[0]
    finally:
        cursor.close()
    if status != extensions.TRANSACTION_STATUS_INTRANS or read_only != "on":
        raise RuntimeError(
            f"after query {name!r} the transaction is no longer read-only; "
            "anything run after it ended may have been committed"
        )


def as_markdown(df: pd.DataFrame) -> str:
    def cell(value) -> str:
        return (
            ""
            if value is None or (isinstance(value, float) and pd.isna(value))
            else str(value)
        )

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += [
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join(lines)


def run(out: Path, sql_file: Path = SQL_FILE) -> None:
    results = {}
    # One read-only snapshot transaction for every query (db.readonly).
    with read_only_connection() as conn:
        # Each query goes through the same connection's own cursor with no
        # parameters, so it reaches Postgres exactly as written (a "%" in the
        # SQL is an operator, not a placeholder).
        dbapi = conn.connection.dbapi_connection
        if dbapi is None:
            raise RuntimeError("no DBAPI connection to run the queries on")
        cursor = dbapi.cursor()
        try:
            for name, sql in queries(sql_file).items():
                cursor.execute(sql)
                # Checked before the result is read: a query whose last
                # statement writes returns no result to read.
                _still_read_only(dbapi, name)
                if cursor.description is None:
                    raise RuntimeError(f"query {name!r} returned no result set")
                columns = [column[0] for column in cursor.description]
                results[name] = pd.DataFrame(
                    cursor.fetchall(), columns=columns, dtype=object
                )
                print(f"  {name}: {len(results[name])} rows")
        finally:
            cursor.close()

    sections = [f"### {name}\n\n{as_markdown(df)}\n" for name, df in results.items()]
    out.write_text("\n".join(sections), encoding="utf-8")
    print(f"Results written to {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run diagnostic queries read-only")
    parser.add_argument(
        "--live",
        action="store_true",
        help="read the non-local database in .env, as the read-only role",
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="results file (Markdown)"
    )
    parser.add_argument(
        "--sql",
        type=Path,
        default=SQL_FILE,
        help="the query file, split on -- @query markers (default: 1a_diagnostics.sql)",
    )
    args = parser.parse_args(argv)

    # The file is checked before anything connects (check_read_only).
    try:
        queries(args.sql)
    except (OSError, ValueError) as e:
        print(f"Refusing: {e}", file=sys.stderr)
        return 2

    refusal = live_refusal(args.live)
    if refusal:
        print(f"Refusing: {refusal}", file=sys.stderr)
        return 2
    print(
        "Target: "
        + (
            "a local database"
            if is_local_database()
            else "the live database configured in .env, as the read-only role"
        )
        + "; one read-only transaction, rolled back at the end"
    )
    try:
        run(args.out, args.sql)
    except Exception as e:
        print(f"Failed: {type(e).__name__}: {masked(str(e))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
