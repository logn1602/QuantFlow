"""
config.py
---------
Central config loader. All modules import from here — never
read os.environ directly in your scripts.
"""

import os
import re
import sys
from datetime import UTC, date, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Database ─────────────────────────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "localhost")
# `or` rather than a getenv default here and for FETCH_INTERVAL_MINUTES: a
# blank `DB_PORT=` falls back too, instead of failing int('') at import and
# taking every entry point down.
DB_PORT = int(os.getenv("DB_PORT") or 5432)
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

# A role that can only read (sql/schema_roles.sql), on the same database. The
# reads of the live database outside the pipeline, the diagnostics' and the
# evaluation's snapshot export with --live, log in as it and refuse to run
# without it (db.readonly): the pipeline's own credentials can write. On
# Supabase's pooler the user is quantflow_readonly.<project-ref>.
DB_READONLY_USER = os.getenv("DB_READONLY_USER", "")
DB_READONLY_PASSWORD = os.getenv("DB_READONLY_PASSWORD", "")

# ── API Keys ──────────────────────────────────────────────────────────────────
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# ── Tickers ───────────────────────────────────────────────────────────────────
# The fallback when TICKERS is unset: the same eight that .env.example, the
# daily workflow and the README track. tests/unit/test_config.py and
# test_workflows.py fail if they drift apart.
DEFAULT_TICKERS = "AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA,META,JPM"
_raw_tickers = os.getenv("TICKERS", DEFAULT_TICKERS)
TICKERS: list[str] = [t.strip().upper() for t in _raw_tickers.split(",") if t.strip()]

# ── Scheduler ─────────────────────────────────────────────────────────────────
FETCH_INTERVAL_MINUTES = int(os.getenv("FETCH_INTERVAL_MINUTES") or 15)

# ── Split guard ───────────────────────────────────────────────────────────────
# A fraction, not a percentage: 0.15 fails the refresh when stored daily closes
# differ from Yahoo's raw closes by more than 15%. Kept as the raw text and
# parsed by quantflow.ingestion.split_guard.configured_threshold, which applies
# the default for a blank value and rejects anything else that is not a
# fraction between 0 and 1. Parsed here, a typo such as "15%" would fail at
# import and take every entry point down, the dashboard included.
SPLIT_GUARD_THRESHOLD = os.getenv("SPLIT_GUARD_THRESHOLD", "")

# Real one-day moves that the split guard's jump check would otherwise take
# for a split, each with the reason it was judged real. A committed file, not
# an environment variable, so a commit carries its list: the daily workflow
# reads the one it runs on, a local run or the scheduler its working tree,
# with no second environment to edit. Resolved from this file rather than the
# working directory: everywhere the guard runs (uv sync and the daily
# workflow) installs the package editable, so this is the checkout. Read by
# split_jump_acknowledged() when the guard runs, never at import, for the
# same reason as SPLIT_GUARD_THRESHOLD.
SPLIT_JUMP_ACKNOWLEDGED_FILE = (
    Path(__file__).resolve().parents[2] / "config" / "split_jump_acknowledged.txt"
)

# ── Trending tier ─────────────────────────────────────────────────────────────
# docs/trending.md, section 11. Each is kept as the raw text and parsed by
# quantflow.ingestion.trending_rules.settings() when the tier runs, never at
# import, for SPLIT_GUARD_THRESHOLD's reason: a typo must not take the
# dashboard down. A blank value takes the default below.
#
# TRENDING_PINNED's default is also written in .env.example and trending.yml;
# tests/unit/test_trending_config.py fails if the three drift apart.
DEFAULT_TRENDING_PINNED = "SPCX"
DEFAULT_TRENDING_LISTS = "trending_tickers,most_actives,day_gainers,day_losers"
DEFAULT_TRENDING_MAX_ACTIVE = 15
DEFAULT_TRENDING_MAX_NEW_PER_RUN = 5
DEFAULT_TRENDING_INACTIVE_AFTER_DAYS = 5
DEFAULT_TRENDING_MIN_PRICE = 5.0
DEFAULT_TRENDING_PROMOTION_MIN_DAYS = 20
DEFAULT_PROMOTION_MIN_HISTORY_DAYS = 504

TRENDING_PINNED = os.getenv("TRENDING_PINNED", "")
TRENDING_LISTS = os.getenv("TRENDING_LISTS", "")
TRENDING_MAX_ACTIVE = os.getenv("TRENDING_MAX_ACTIVE", "")
TRENDING_MAX_NEW_PER_RUN = os.getenv("TRENDING_MAX_NEW_PER_RUN", "")
TRENDING_INACTIVE_AFTER_DAYS = os.getenv("TRENDING_INACTIVE_AFTER_DAYS", "")
TRENDING_MIN_PRICE = os.getenv("TRENDING_MIN_PRICE", "")
# No default: blank leaves the market-cap floor off.
TRENDING_MIN_MARKET_CAP = os.getenv("TRENDING_MIN_MARKET_CAP", "")
TRENDING_PROMOTION_MIN_DAYS = os.getenv("TRENDING_PROMOTION_MIN_DAYS", "")
PROMOTION_MIN_HISTORY_DAYS = os.getenv("PROMOTION_MIN_HISTORY_DAYS", "")

# Symbols Yahoo reuses: a trending symbol's bars before its date here belong to
# a former holder (docs/trending.md, section 5). Committed, in the format of
# split_jump_acknowledged.txt and read by the same parser, for the same reason.
TRENDING_LISTING_DATES_FILE = (
    Path(__file__).resolve().parents[2] / "config" / "trending_listing_dates.txt"
)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
# Anchored to the working directory, not to this file. Under the old flat
# layout os.path.dirname(__file__) was the repo root; under src/ it would be
# src/quantflow/logs, which silently moves the log file inside the package.
# Every entry point runs from the repo root, so cwd reproduces the previous
# path exactly, and an installed copy still gets somewhere sensible.
LOG_DIR = os.getenv("LOG_DIR") or os.path.join(os.getcwd(), "logs")


# A ticker as stored (raw_prices.ticker is VARCHAR(10)), a colon, an ISO date.
_ACKNOWLEDGED_ENTRY = re.compile(r"([A-Z][A-Z0-9.\-]{0,9}):(\d{4}-\d{2}-\d{2})")


def split_jump_acknowledged(path: Path | None = None) -> dict[tuple[str, date], str]:
    """The acknowledged split-sized moves: (ticker, date) mapped to the reason.

    The date is the trading date whose close moved, compared with the stored
    close before it. Blank lines and lines starting with # are skipped; every
    other line must read TICKER:YYYY-MM-DD followed by a # comment giving the
    reason. Tickers are upcased, as in TICKERS. A date may not be in the
    future: only a move that has happened can be confirmed.

    Lines end at a newline only, as git and a review diff see them. Python's
    splitlines() also ends one at a form feed, U+2028 and the like, so
    '#<form feed>XYZ:...' would read as a comment in review and load as an
    entry. Any control or invisible character other than a tab is refused.

    Raises ValueError naming the file, and the line where there is one, for a
    file that is missing, unreadable or not UTF-8, and for a malformed,
    repeated or future entry: the guard then stops as misconfigured rather
    than run on a list it misread. A missing file is not read as an empty
    list, because the file is committed with the code.
    """
    path = SPLIT_JUMP_ACKNOWLEDGED_FILE if path is None else path
    try:
        # utf-8-sig: Notepad and other Windows editors may save with a BOM.
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise ValueError(
            f"{path} not found; it is committed with the code, and a list "
            f"with no entries is a file with only comments"
        ) from None
    except UnicodeDecodeError as e:
        raise ValueError(
            f"{path} is not UTF-8 text (byte {e.start}); save it as UTF-8"
        ) from None
    except OSError as e:
        raise ValueError(f"{path} cannot be read: {e}") from None

    # UTC's date is never behind New York's, so this refuses only dates that
    # have not begun anywhere the exchange is.
    today = datetime.now(UTC).date()
    found: dict[tuple[str, date], str] = {}
    first_seen: dict[tuple[str, date], int] = {}
    for number, raw in enumerate(text.split("\n"), start=1):
        where = f"{path}, line {number}"
        line = raw.removesuffix("\r")
        if not line.replace("\t", " ").isprintable():
            raise ValueError(
                f"{where}: contains a control or invisible character; the list "
                f"must be plain text"
            )
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entry, hash_mark, reason = line.partition("#")
        reason = reason.strip()
        if not hash_mark or not reason:
            raise ValueError(
                f"{where}: give the reason after a #, as in "
                f"'XYZ:2026-05-01  # fell 34% on earnings'"
            )
        match = _ACKNOWLEDGED_ENTRY.fullmatch(entry.strip().upper())
        if match is None:
            raise ValueError(
                f"{where}: expected TICKER:YYYY-MM-DD before the #, "
                f"got {entry.strip()!r}"
            )
        try:
            day = date.fromisoformat(match.group(2))
        except ValueError:
            raise ValueError(f"{where}: {match.group(2)} is not a date") from None
        if day > today:
            raise ValueError(
                f"{where}: {day} is in the future; only a move that has happened "
                f"can be acknowledged"
            )
        key = (match.group(1), day)
        if key in found:
            raise ValueError(f"{where}: repeats line {first_seen[key]}")
        found[key] = reason
        first_seen[key] = number
    return found


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
