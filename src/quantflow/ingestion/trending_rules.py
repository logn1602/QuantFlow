"""The trending tier's rules, as pure functions (docs/trending.md, section 4).

Everything here decides from its arguments alone: no database, no network, no
clock. The run (pipelines/trending.py) reads the inputs and writes the
results, so each rule can be tested on its own.

Symbols arrive from unofficial endpoints and from the environment, so each one
is validated before it reaches the database, a URL or a log line (4.1).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TypeGuard

from quantflow import config
from quantflow.features.sentiment import _clean_text
from quantflow.utils.logger import get_logger

logger = get_logger(__name__)

# raw_prices.ticker and every trending table's symbol column are VARCHAR(10).
MAX_SYMBOL_CHARS = 10
_SYMBOL = re.compile(r"[A-Z][A-Z0-9.\-]{0,9}", re.ASCII)

# tickers.short_name is VARCHAR(100), tickers.quote_type VARCHAR(20).
MAX_SHORT_NAME_CHARS = 100
MAX_QUOTE_TYPE_CHARS = 20

# The lists the sources can read, in the default priority order (4.3).
KNOWN_LISTS = ("trending_tickers", "most_actives", "day_gainers", "day_losers")

# TRENDING_PINNED is read up to this many symbols, so the owner's list cannot
# crowd out the tier's budget (section 5).
MAX_PINNED = 10


class TrendingConfigError(ValueError):
    """A trending setting that cannot be read; the run stops on it."""


def valid_symbol(symbol: object) -> TypeGuard[str]:
    """True for a symbol the tier may store, put in a URL or log as it is:
    upper-case ASCII letters, digits, "." and "-", starting with a letter, at
    most 10 characters. Lower case is refused, not upcased: Yahoo sends upper
    case, so anything else is not a symbol it meant."""
    return isinstance(symbol, str) and _SYMBOL.fullmatch(symbol) is not None


def shown(value: object, limit: int = 20) -> str:
    """An untrusted value as it may appear in a log line: escaped by repr,
    so no control character or newline reaches the log, and cut short."""
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def clean_text(value: object, limit: int) -> str | None:
    """Feed text as it may be stored: cleaned as headlines are (no NUL, no
    code point UTF-8 cannot encode), with every other control or format
    character removed, runs of whitespace made one space, at most limit
    characters. None for no text at all."""
    if not isinstance(value, str):
        return None
    cleaned = "".join(
        " " if ch.isspace() else ch
        for ch in _clean_text(value, len(value))
        if ch.isspace() or unicodedata.category(ch) not in ("Cc", "Cf")
    )
    cleaned = " ".join(cleaned.split())[:limit].strip()
    return cleaned or None


# ── Settings (section 11) ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrendingSettings:
    pinned: tuple[str, ...]
    lists: tuple[str, ...]
    max_active: int
    max_new_per_run: int
    inactive_after: int
    min_price: float  # 0: no floor
    min_market_cap: float | None  # None: no floor
    promotion_min_days: int
    promotion_min_history_days: int


def _integer(name: str, raw: str, default: int, minimum: int) -> int:
    text = raw.strip()
    if not text:
        return default
    try:
        value = int(text)
    except ValueError:
        raise TrendingConfigError(
            f"{name} must be a whole number, got {shown(raw)}"
        ) from None
    if value < minimum:
        raise TrendingConfigError(f"{name} must be at least {minimum}, got {value}")
    return value


def _amount(name: str, raw: str, default: float | None) -> float | None:
    text = raw.strip()
    if not text:
        return default
    try:
        value = float(text)
    except ValueError:
        raise TrendingConfigError(
            f"{name} must be a number of dollars, got {shown(raw)}"
        ) from None
    if not value >= 0 or value == float("inf"):
        raise TrendingConfigError(f"{name} must be 0 or more, got {shown(raw)}")
    return value


def pinned_symbols(raw: str, core: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """TRENDING_PINNED as symbols: blank takes the default; upcased, as
    TICKERS is; an invalid symbol or a core ticker is logged and left out,
    never fatal; repeats dropped; at most MAX_PINNED kept."""
    text = raw if raw.strip() else config.DEFAULT_TRENDING_PINNED
    kept: list[str] = []
    for entry in text.split(","):
        symbol = entry.strip().upper()
        if not symbol or symbol in kept:
            continue
        if not valid_symbol(symbol):
            logger.warning(f"TRENDING_PINNED: {shown(entry.strip())} is not a symbol")
            continue
        if symbol in core:
            logger.warning(
                f"TRENDING_PINNED: {symbol} is a core ticker, which the trending "
                f"tier never tracks"
            )
            continue
        if len(kept) == MAX_PINNED:
            logger.warning(
                f"TRENDING_PINNED: more than {MAX_PINNED} symbols; {symbol} and any "
                f"after it are left out"
            )
            break
        kept.append(symbol)
    return tuple(kept)


def list_names(raw: str) -> tuple[str, ...]:
    """TRENDING_LISTS as list names, in the order given: blank takes the
    default. An unknown name stops the run: read as a typo and skipped, it
    would leave a list out without anyone noticing."""
    text = raw if raw.strip() else config.DEFAULT_TRENDING_LISTS
    names: list[str] = []
    for entry in text.split(","):
        name = entry.strip().lower()
        if not name or name in names:
            continue
        if name not in KNOWN_LISTS:
            raise TrendingConfigError(
                f"TRENDING_LISTS: unknown list {shown(entry.strip(), 40)}; the "
                f"known ones are {', '.join(KNOWN_LISTS)}"
            )
        names.append(name)
    if not names:
        raise TrendingConfigError("TRENDING_LISTS names no list")
    return tuple(names)


def settings(core: list[str] | tuple[str, ...]) -> TrendingSettings:
    """The trending settings from config, read now rather than at import, so
    a typo stops the trending run alone (TrendingConfigError) and never the
    dashboard's import. core is the core watchlist, passed in so the rules
    never read TICKERS themselves."""
    return TrendingSettings(
        pinned=pinned_symbols(config.TRENDING_PINNED, core),
        lists=list_names(config.TRENDING_LISTS),
        max_active=_integer(
            "TRENDING_MAX_ACTIVE",
            config.TRENDING_MAX_ACTIVE,
            config.DEFAULT_TRENDING_MAX_ACTIVE,
            minimum=0,
        ),
        max_new_per_run=_integer(
            "TRENDING_MAX_NEW_PER_RUN",
            config.TRENDING_MAX_NEW_PER_RUN,
            config.DEFAULT_TRENDING_MAX_NEW_PER_RUN,
            minimum=0,
        ),
        inactive_after=_integer(
            "TRENDING_INACTIVE_AFTER_DAYS",
            config.TRENDING_INACTIVE_AFTER_DAYS,
            config.DEFAULT_TRENDING_INACTIVE_AFTER_DAYS,
            minimum=1,
        ),
        min_price=_amount(
            "TRENDING_MIN_PRICE",
            config.TRENDING_MIN_PRICE,
            config.DEFAULT_TRENDING_MIN_PRICE,
        )
        or 0.0,
        min_market_cap=_amount(
            "TRENDING_MIN_MARKET_CAP", config.TRENDING_MIN_MARKET_CAP, None
        )
        or None,
        promotion_min_days=_integer(
            "TRENDING_PROMOTION_MIN_DAYS",
            config.TRENDING_PROMOTION_MIN_DAYS,
            config.DEFAULT_TRENDING_PROMOTION_MIN_DAYS,
            minimum=1,
        ),
        promotion_min_history_days=_integer(
            "PROMOTION_MIN_HISTORY_DAYS",
            config.PROMOTION_MIN_HISTORY_DAYS,
            config.DEFAULT_PROMOTION_MIN_HISTORY_DAYS,
            minimum=1,
        ),
    )
