"""
db/connection.py
----------------
Database connection helpers.
Use get_engine() for SQLAlchemy (DataFrames).
Use get_conn() for raw psycopg2 (custom SQL).
"""

import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from quantflow.config import DATABASE_URL, DB_SSLMODE, DB_SSLROOTCERT
from quantflow.utils.logger import get_logger

logger = get_logger(__name__)

_engine = None


def _ssl_args() -> dict:
    """libpq TLS parameters shared by both connection paths."""
    args = {"sslmode": DB_SSLMODE}
    if DB_SSLROOTCERT:
        args["sslrootcert"] = DB_SSLROOTCERT
    return args


def get_engine():
    """Return a singleton SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            connect_args=_ssl_args(),
        )
    return _engine


def get_conn():
    """Return a raw psycopg2 connection. Caller is responsible for closing it.

    Hands the DSN to libpq intact rather than rebuilding it from urlparse
    fields. The old version passed only host/port/dbname/user/password, so any
    query string on DATABASE_URL — '?sslmode=verify-full', say — was dropped
    with no error, leaving this path silently weaker than get_engine().
    """
    return psycopg2.connect(DATABASE_URL, **_ssl_args())


def test_connection() -> bool:
    """Ping the database. Returns True if reachable."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection OK")
        return True
    except OperationalError as e:
        logger.error(f"Database connection FAILED: {e}")
        return False
