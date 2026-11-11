"""Read-only access to the configured database.

For the code that reads the shared live database without the pipeline
around it: the diagnostics in scripts/diagnostics/ and the evaluation's
snapshot export. Five safeguards, the same for both:

  - live_refusal: callers refuse a non-local database, before connecting,
    unless told otherwise (their --live flag) and the read-only role is
    configured;
  - a non-local database is read as that role (DB_READONLY_USER and
    DB_READONLY_PASSWORD, created by sql/schema_roles.sql), which can only
    SELECT: never with the pipeline's credentials, which can write. The
    other safeguards depend on the code being right; this one does not;
  - read_only_connection: one REPEATABLE READ, READ ONLY transaction,
    checked with transaction_read_only before anything runs, with a
    statement timeout, rolled back at the end;
  - require_rows: a table the caller reads that always holds rows, found
    empty, stops the read before the caller runs anything. Row-level
    security with no policy for a role hides every row from it and raises
    no error: read as quantflow_readonly, every core table of the live
    database looked empty (2026-11-11), which a report or a snapshot would
    have carried as data;
  - masked: errors are printed with the connection settings replaced.

A local database (a throwaway cluster, the integration tests') is read with
the configured credentials, as before.
"""

import re
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

from quantflow import config
from quantflow.db.connection import engine_as, get_engine

LOCAL_HOSTS = {"localhost", "127.0.0.1"}

NO_ROLE = (
    "the live database is read only as the read-only role: set "
    "DB_READONLY_USER and DB_READONLY_PASSWORD in .env (sql/schema_roles.sql)"
)


# The core tables every daily refresh fills. Found empty by a read, they are
# most likely hidden from its role, not empty: row-level security with no
# SELECT policy for the role (sql/schema_roles.sql, step 7). Not alerts_log,
# which holds a row only once an alert fires, nor the trending tables, empty
# until the tier's first run.
ROWS_EXPECTED = frozenset(
    {
        "raw_prices",
        "technical_indicators",
        "anomalies",
        "forecasts",
        "news_sentiment",
        "model_metrics",
        "backtest_results",
    }
)

_TABLE_NAME = re.compile(r"[a-z_][a-z0-9_]*")


class NoReadOnlyRoleError(RuntimeError):
    """A non-local database, and no read-only role to read it as."""


class EmptyTablesError(RuntimeError):
    """A table the read expects rows in shows its role none."""


def is_local_database() -> bool:
    return config.DB_HOST in LOCAL_HOSTS


def has_read_only_role() -> bool:
    return bool(config.DB_READONLY_USER and config.DB_READONLY_PASSWORD)


def live_refusal(live: bool) -> str | None:
    """Why a run must not connect, or None: a non-local database needs the
    caller's --live and the read-only role."""
    if is_local_database():
        return None
    if not live:
        return ".env points at a non-local database; pass --live."
    if not has_read_only_role():
        return NO_ROLE + "."
    return None


def masked(message: str) -> str:
    """message with the connection settings replaced, for printing errors."""
    for secret in (
        config.DB_PASSWORD,
        config.DB_READONLY_PASSWORD,
        config.DB_USER,
        config.DB_READONLY_USER,
        config.DB_HOST,
        config.DB_NAME,
    ):
        if secret:
            message = message.replace(secret, "<masked>")
    return message


def _engine():
    """The engine to read with: the read-only role's for a non-local
    database, the configured one for a local database."""
    if is_local_database():
        return get_engine()
    if not has_read_only_role():
        raise NoReadOnlyRoleError(NO_ROLE)
    return engine_as(config.DB_READONLY_USER, config.DB_READONLY_PASSWORD)


def require_rows(conn, tables: Iterable[str]) -> None:
    """Raise EmptyTablesError unless each of tables shows conn a row.

    One query, which also asks whether row-level security applies to the
    role on each table, so the error says which it is: rows hidden by a
    missing policy, or a table that is empty.
    """
    names = sorted(set(tables))
    if not names:
        return
    odd = [name for name in names if not _TABLE_NAME.fullmatch(name)]
    if odd:
        raise ValueError(f"not a plain table name: {odd}")
    query = " UNION ALL ".join(
        f"SELECT '{name}', EXISTS (SELECT 1 FROM {name}), row_security_active('{name}')"
        for name in names
    )
    empty = [
        (name, secured)
        for name, seen, secured in conn.exec_driver_sql(query).all()
        if not seen
    ]
    if not empty:
        return
    role = conn.exec_driver_sql("SELECT current_user").scalar_one()
    hidden = [name for name, secured in empty if secured]
    bare = [name for name, secured in empty if not secured]
    reasons = []
    if hidden:
        reasons.append(
            f"{', '.join(hidden)}: no row visible to {role}. Row-level security "
            "applies to the role there, and without a SELECT policy for it every "
            "row is hidden (sql/schema_roles.sql, step 7)"
        )
    if bare:
        reasons.append(
            f"{', '.join(bare)}: empty, with no row-level security in the way: "
            "is this the database .env should point at?"
        )
    raise EmptyTablesError(
        "; ".join(reasons) + ". Stopped before reading on: these tables always "
        "hold rows, and a read without them would report their absence as data."
    )


@contextmanager
def read_only_connection(
    statement_timeout: str = "120s", expect_rows: Iterable[str] = ()
) -> Iterator:
    """A connection inside one read-only snapshot transaction.

    The transaction's first statement makes it REPEATABLE READ, READ ONLY:
    every read sees the same snapshot, and Postgres rejects any write. It is
    rolled back at the end, and the engine disposed. A non-local database is
    read as the read-only role, or not at all (NoReadOnlyRoleError).
    expect_rows: tables the caller reads that always hold rows; one that
    shows none stops the read before the caller's first query
    (require_rows, in the same snapshot).
    """
    engine = _engine()
    try:
        with engine.connect() as conn:
            txn = conn.begin()
            try:
                conn.exec_driver_sql(
                    "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
                )
                read_only = conn.exec_driver_sql(
                    "SHOW transaction_read_only"
                ).scalar_one()
                if read_only != "on":
                    raise RuntimeError(
                        f"transaction_read_only is {read_only!r}; stopping"
                    )
                conn.exec_driver_sql(
                    f"SET LOCAL statement_timeout = '{statement_timeout}'"
                )
                require_rows(conn, expect_rows)
                yield conn
            finally:
                txn.rollback()
    finally:
        engine.dispose()
