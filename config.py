"""
config.py
---------
Central config loader. All modules import from here — never
read os.environ directly in your scripts.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# ── Database ─────────────────────────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "stock_pipeline")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# TLS. Default 'require' rather than libpq's 'prefer': prefer falls back to
# cleartext silently — no error, no log line — if the peer declines TLS, and
# the database is a public endpoint reached from a laptop, a GitHub Actions
# runner and Streamlit Cloud.
#
# 'require' encrypts but does NOT verify the peer's certificate. 'verify-full'
# does, and is the recommended upgrade, but Supabase issues its pooler cert
# from a private CA ("Supabase Intermediate 2021 CA") that is in neither the
# system trust store nor certifi, so it needs DB_SSLROOTCERT pointing at
# Supabase's CA file. See the TLS section in the README.
DB_SSLMODE = os.getenv("DB_SSLMODE", "require")
DB_SSLROOTCERT = os.getenv("DB_SSLROOTCERT", "")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ── API Keys ──────────────────────────────────────────────────────────────────
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# ── Tickers ───────────────────────────────────────────────────────────────────
_raw_tickers = os.getenv("TICKERS", "AAPL,MSFT,GOOGL,NVDA")
TICKERS: list[str] = [t.strip().upper() for t in _raw_tickers.split(",") if t.strip()]

# ── Scheduler ─────────────────────────────────────────────────────────────────
FETCH_INTERVAL_MINUTES = int(os.getenv("FETCH_INTERVAL_MINUTES", 15))

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")


def validate():
    """
    Call once at startup to catch missing config early.

    Returns True only when every variable is set, including the optional API
    keys. A False result is informational — use require_or_exit() for the
    subset the pipeline genuinely cannot run without.
    """
    errors = []
    if not DB_PASSWORD:
        errors.append("DB_PASSWORD is not set in .env")
    if not ALPHA_VANTAGE_API_KEY:
        errors.append("ALPHA_VANTAGE_API_KEY is not set in .env")
    if not NEWS_API_KEY:
        errors.append("NEWS_API_KEY is not set in .env")
    if not TICKERS:
        errors.append("TICKERS list is empty in .env")
    if errors:
        for e in errors:
            print(f"[config] WARNING: {e}")
    return len(errors) == 0


def require_or_exit():
    """
    Hard gate for CLI entry points: exit non-zero if a variable the pipeline
    cannot function without is missing.

    Deliberately narrower than validate(). Only two things are fatal:

      DB_PASSWORD  every stage reads or writes Postgres, so without it nothing
                   can run and the failure is otherwise a confusing
                   authentication error deep in a model job.
      TICKERS      an empty list means every loop body is skipped and the
                   pipeline "succeeds" having done nothing.

    The API keys are NOT fatal. Alpha Vantage is unused by the default
    ingestion path, and sentiment.py already degrades to the RSS feed with a
    warning when NEWS_API_KEY is absent — killing the whole run for that would
    be worse than the missing data.
    """
    # Plain ASCII: these go to stderr on Windows consoles and CI logs alike,
    # where a non-UTF-8 code page turns an em dash into mojibake.
    fatal = []
    if not DB_PASSWORD:
        fatal.append("DB_PASSWORD is not set - cannot reach the database")
    if not TICKERS:
        fatal.append("TICKERS is empty - there is nothing to process")

    if fatal:
        for f in fatal:
            print(f"[config] FATAL: {f}", file=sys.stderr)
        print("[config] Fix your .env (see .env.example) and retry.", file=sys.stderr)
        sys.exit(1)
