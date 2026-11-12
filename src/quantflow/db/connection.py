"""
db/connection.py
----------------
Database connection helpers.
Use get_engine() for SQLAlchemy (DataFrames).
"""

from sqlalchemy import URL, create_engine, text
from sqlalchemy.exc import OperationalError

from quantflow import config
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


def engine_as(user: str, password: str):
    """A new engine for the configured database, logged in as another role:
    db.readonly's read-only one. Not the shared engine; the caller disposes
    of it. The URL is built escaped, so any password works."""
    url = URL.create(
        "postgresql",
        username=user,
        password=password,
        host=config.DB_HOST,
        port=config.DB_PORT,
        database=config.DB_NAME,
    )
    return create_engine(url, pool_pre_ping=True, connect_args=_ssl_args())


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
