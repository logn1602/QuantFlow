"""Read-only access to the configured database.

For the code that reads the shared live database without the pipeline
around it: the diagnostics in scripts/diagnostics/ and the evaluation's
snapshot export. Three safeguards, the same for both:

  - is_local_database: callers refuse a non-local database unless told
    otherwise (their --live flag), before connecting;
  - read_only_connection: one REPEATABLE READ, READ ONLY transaction,
    checked with transaction_read_only before anything runs, with a
    statement timeout, rolled back at the end;
  - masked: errors are printed with the connection settings replaced.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from quantflow import config
from quantflow.db.connection import get_engine

LOCAL_HOSTS = {"localhost", "127.0.0.1"}


def is_local_database() -> bool:
    return config.DB_HOST in LOCAL_HOSTS


def masked(message: str) -> str:
    """message with the connection settings replaced, for printing errors."""
    for secret in (config.DB_PASSWORD, config.DB_USER, config.DB_HOST, config.DB_NAME):
        if secret:
            message = message.replace(secret, "<masked>")
    return message


@contextmanager
def read_only_connection(statement_timeout: str = "120s") -> Iterator:
    """A connection inside one read-only snapshot transaction.

    The transaction's first statement makes it REPEATABLE READ, READ ONLY:
    every read sees the same snapshot, and Postgres rejects any write. It is
    rolled back at the end, and the engine disposed.
    """
    engine = get_engine()
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
