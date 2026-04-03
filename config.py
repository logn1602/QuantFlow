"""
config.py
---------
Central config loader. All modules import from here — never
read os.environ directly in your scripts.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ── Database ────────────────────────────────────────────────────────────────
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = int(os.getenv("DB_PORT", 5432))
DB_NAME     = os.getenv("DB_NAME", "stock_pipeline")
DB_USER     = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

DATABASE_URL = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ── API Keys ─────────────────────────────────────────────────────────────────
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")

# ── Tickers ──────────────────────────────────────────────────────────────────
_raw_tickers = os.getenv("TICKERS", "AAPL,MSFT,GOOGL,NVDA")
TICKERS: list[str] = [t.strip().upper() for t in _raw_tickers.split(",") if t.strip()]

# ── Scheduler ────────────────────────────────────────────────────────────────
FETCH_INTERVAL_MINUTES = int(os.getenv("FETCH_INTERVAL_MINUTES", 15))

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR   = os.path.join(os.path.dirname(__file__), "logs")


def validate():
    """Call once at startup to catch missing config early."""
    errors = []
    if not DB_PASSWORD:
        errors.append("DB_PASSWORD is not set in .env")
    if not ALPHA_VANTAGE_API_KEY:
        errors.append("ALPHA_VANTAGE_API_KEY is not set in .env")
    if not TICKERS:
        errors.append("TICKERS list is empty in .env")
    if errors:
        for e in errors:
            print(f"[config] WARNING: {e}")
    return len(errors) == 0
