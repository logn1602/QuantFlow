"""Read-only access to the configured database.

For the code that reads the shared live database without the pipeline
around it: the diagnostics in scripts/diagnostics/ and the evaluation's
snapshot export. Four safeguards, the same for both:

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
  - masked: errors are printed with the connection settings replaced.

A local database (a throwaway cluster, the integration tests') is read with
the configured credentials, as before.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from quantflow import config
from quantflow.db.connection import engine_as, get_engine

LOCAL_HOSTS = {"localhost", "127.0.0.1"}

NO_ROLE = (
    "the live database is read only as the read-only role: set "
    "DB_READONLY_USER and DB_READONLY_PASSWORD in .env (sql/schema_roles.sql)"
)


class NoReadOnlyRoleError(RuntimeError):
    """A non-local database, and no read-only role to read it as."""


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


@contextmanager
def read_only_connection(statement_timeout: str = "120s") -> Iterator:
    """A connection inside one read-only snapshot transaction.

    The transaction's first statement makes it REPEATABLE READ, READ ONLY:
    every read sees the same snapshot, and Postgres rejects any write. It is
    rolled back at the end, and the engine disposed. A non-local database is
    read as the read-only role, or not at all (NoReadOnlyRoleError).
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
                yield conn
            finally:
                txn.rollback()
    finally:
        engine.dispose()
