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

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import DATABASE_URL
from utils.logger import get_logger

logger = get_logger(__name__)

_engine = None


def get_engine():
    """Return a singleton SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    return _engine


def get_conn():
    """Return a raw psycopg2 connection. Caller is responsible for closing it."""
    from urllib.parse import urlparse
    u = urlparse(DATABASE_URL)
    return psycopg2.connect(
        host=u.hostname,
        port=u.port or 5432,
        dbname=u.path.lstrip("/"),
        user=u.username,
        password=u.password,
    )


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
