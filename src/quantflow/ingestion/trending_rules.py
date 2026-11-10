"""The trending tier's rules, as pure functions (docs/trending.md, section 4).

Everything here decides from its arguments alone: no database, no network, no
clock. The run (pipelines/trending.py) reads the inputs and writes the
results, so each rule can be tested on its own.

Symbols arrive from unofficial endpoints and from the environment, so each one
is validated before it reaches the database, a URL or a log line (4.1).
"""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard
from urllib.parse import quote

import pandas as pd

from quantflow import config
from quantflow.features.sentiment import _clean_text
from quantflow.utils.logger import get_logger
from quantflow.utils.trading_calendar import EXCHANGE_TZ, TRADING_DAY

if TYPE_CHECKING:
    from quantflow.ingestion.trending_sources import ListFetch, TrendingItem

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


# ── News (7.5) ────────────────────────────────────────────────────────────────

# Yahoo's per-symbol RSS feed: the form every core feed in
# features.sentiment.RSS_FEEDS takes, which a test keeps equal, so a trending
# symbol's feed is the same kind of feed.
RSS_FEED_TEMPLATE = (
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
)


def rss_feed(symbol: str) -> str:
    """A symbol's RSS feed URL, built from a validated symbol only and
    URL-encoded (4.1)."""
    if not valid_symbol(symbol):
        raise ValueError(f"not a symbol: {shown(symbol)}")
    return RSS_FEED_TEMPLATE.format(symbol=quote(symbol, safe=""))


# ── Sessions (4.3) ────────────────────────────────────────────────────────────

# The regular session opens at 09:30 New York.
SESSION_OPENS = dt.time(9, 30)


def session_date(at: dt.datetime | pd.Timestamp) -> dt.date:
    """The session a fetch made at `at` describes: the latest New York session
    that had opened by then. Today's from 09:30 New York on a trading day,
    else the trading day before. So a run in the evening, one after midnight
    UTC and one before the next open date to the same session and count once,
    and a run during the session describes that session. A naive `at` is
    read as UTC."""
    ts = pd.Timestamp(at)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    local = ts.tz_convert(EXCHANGE_TZ)
    today = local.tz_localize(None).normalize()
    if TRADING_DAY.is_on_offset(today) and local.time() >= SESSION_OPENS:
        return today.date()
    return (today - TRADING_DAY).date()


# ── Candidates (4.2) ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Candidate:
    """One symbol through rules 1 to 4, its lists merged (rule 5): every list
    it was on, in priority order, its best rank, and the rest from its
    best-ranked item."""

    symbol: str
    lists: tuple[str, ...]
    best_rank: int
    best_list: str
    pct_change: float | None
    volume: int | None
    market_cap: int | None
    price: float | None
    quote_type: str | None
    short_name: str | None
    first_trade_date: dt.date | None


def _priority(lists: Sequence[str]) -> Callable[[str], int]:
    """A list's place in TRENDING_LISTS; a list not named there comes last."""
    order = {name: place for place, name in enumerate(lists)}
    return lambda name: order.get(name, len(order))


def rejection(
    item: TrendingItem, *, core: Iterable[str], settings: TrendingSettings
) -> str | None:
    """Why an item fails rules 1 to 4 of 4.2, or None when it passes. A floor
    needs a value: an item with no price is under any price floor."""
    if not valid_symbol(item.symbol):
        return "not a symbol"
    if item.quote_type != "EQUITY":
        return "not EQUITY"
    if item.symbol in set(core):
        return "a core ticker"
    if settings.min_price > 0 and (
        item.price is None or item.price < settings.min_price
    ):
        return "under the price floor"
    if settings.min_market_cap is not None and (
        item.market_cap is None or item.market_cap < settings.min_market_cap
    ):
        return "under the market-cap floor"
    return None


def _better(item: TrendingItem, held: TrendingItem | None, priority) -> bool:
    """Whether item outranks held: a lower rank, or the same rank on a list
    named earlier in TRENDING_LISTS."""
    return held is None or (item.rank, priority(item.list_name)) < (
        held.rank,
        priority(held.list_name),
    )


def candidates(
    fetches: Iterable[ListFetch], *, core: Iterable[str], settings: TrendingSettings
) -> list[Candidate]:
    """Rules 1 to 5 of 4.2 on every item of the lists that answered, in the
    order the active set takes them (4.3): best rank first; at equal ranks
    the list named first in TRENDING_LISTS, then the higher volume, then the
    symbol, so the order never depends on how the lists arrived. Pinned
    symbols are left out: they are active whatever the rules say."""
    core = set(core)
    priority = _priority(settings.lists)
    best: dict[str, TrendingItem] = {}
    lists: dict[str, set[str]] = {}
    dropped: dict[str, int] = {}
    for fetch in fetches:
        for item in fetch.items:
            if item.symbol in settings.pinned:
                continue
            reason = rejection(item, core=core, settings=settings)
            if reason is not None:
                dropped[reason] = dropped.get(reason, 0) + 1
                continue
            lists.setdefault(item.symbol, set()).add(item.list_name)
            if _better(item, best.get(item.symbol), priority):
                best[item.symbol] = item
    if dropped:
        logger.info(
            "Trending rules dropped "
            + ", ".join(f"{n} {reason}" for reason, n in sorted(dropped.items()))
        )
    merged = [
        Candidate(
            symbol=symbol,
            lists=tuple(sorted(lists[symbol], key=priority)),
            best_rank=item.rank,
            best_list=item.list_name,
            pct_change=item.pct_change,
            volume=item.volume,
            market_cap=item.market_cap,
            price=item.price,
            quote_type=item.quote_type,
            short_name=item.short_name,
            first_trade_date=item.first_trade_date,
        )
        for symbol, item in best.items()
    ]
    return sorted(
        merged,
        key=lambda c: (
            c.best_rank,
            priority(c.best_list),
            c.volume is None,
            -(c.volume or 0),
            c.symbol,
        ),
    )


def pinned_items(
    fetches: Iterable[ListFetch], settings: TrendingSettings
) -> dict[str, TrendingItem]:
    """Each pinned symbol's best-ranked item today, through the rules or not:
    its name, quote type and first trade date, for its tickers row."""
    priority = _priority(settings.lists)
    found: dict[str, TrendingItem] = {}
    for fetch in fetches:
        for item in fetch.items:
            if item.symbol in settings.pinned and _better(
                item, found.get(item.symbol), priority
            ):
                found[item.symbol] = item
    return found


# ── The active set (4.3) ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class Selection:
    """Who is active after a run, and why the rest are not."""

    active: tuple[str, ...]  # the pinned first, then the discovered as chosen
    pinned: tuple[str, ...]
    joined: tuple[str, ...]  # active now and not before
    new: tuple[str, ...]  # joined with no stored history: a first-sight download
    waiting: tuple[str, ...]  # new, over the run's limit: left for a later run
    kept_absent: tuple[str, ...]  # active before, on no list that answered, kept
    deactivated: tuple[str, ...]
    complete: bool  # every list answered: the session counts (T8)


def _missing(value: Any) -> bool:
    return value is None or bool(pd.isna(value))


def absences(last_seen: Any, counted: Sequence[dt.date]) -> int:
    """How many of the counted sessions came after the one a symbol was last
    seen in (last_seen a timestamp, or None for never)."""
    if _missing(last_seen):
        return len(counted)
    seen = session_date(last_seen)
    return sum(1 for day in counted if day > seen)


def _recency(row: Mapping[str, Any]) -> int:
    last_seen = row.get("last_seen")
    return 0 if _missing(last_seen) else pd.Timestamp(last_seen).value


def choose_active(
    found: Sequence[Candidate],
    stored: Mapping[str, Mapping[str, Any]],
    *,
    settings: TrendingSettings,
    complete: bool,
    counted: Sequence[dt.date],
) -> Selection:
    """The active set for this run (4.3), from today's candidates in order
    and the stored tickers rows by symbol.

    Every pinned symbol is active, outside the cap and the new-symbol limit.
    Then, up to the cap, discovered symbols: today's candidates first, then
    symbols active before, on no list today, but absent from fewer than
    TRENDING_INACTIVE_AFTER_DAYS counted sessions, most recently seen first.
    A symbol with no stored history that was not active before is new: at
    most TRENDING_MAX_NEW_PER_RUN of them join, and the rest wait for a later
    run, without taking a place.

    After a fetch in which any list failed (complete False, T8), nobody is
    deactivated: every symbol active before stays, and today's candidates
    join only into room left under the cap. A symbol on the failed list alone
    would otherwise look absent and lose its place to the other lists' names.

    counted is the latest counted sessions, newest first, as
    db.trending.counted_sessions reads them after this run's fetch is stored,
    so this run's session is among them when complete.
    """
    pinned = tuple(settings.pinned)
    before = [
        symbol
        for symbol, row in sorted(stored.items())
        if row.get("active") and symbol not in pinned
    ]
    was_active = set(before)
    listed = [c.symbol for c in found if c.symbol not in pinned]
    on_lists = set(listed)

    chosen: list[str] = []
    new: list[str] = []
    waiting: list[str] = []
    kept_absent: list[str] = []

    def admit(symbol: str) -> None:
        history = (stored.get(symbol) or {}).get("first_bar_date")
        if symbol not in was_active and _missing(history):
            if len(new) >= settings.max_new_per_run:
                waiting.append(symbol)
                return
            new.append(symbol)
        chosen.append(symbol)

    def by_recency(symbols: Iterable[str]) -> list[str]:
        return sorted(symbols, key=lambda s: (-_recency(stored[s]), s))

    if complete:
        for symbol in listed:
            if len(chosen) >= settings.max_active:
                break
            admit(symbol)
        recent = [
            symbol
            for symbol in before
            if symbol not in on_lists
            and absences(stored[symbol].get("last_seen"), counted)
            < settings.inactive_after
        ]
        for symbol in by_recency(recent):
            if len(chosen) >= settings.max_active:
                break
            chosen.append(symbol)
            kept_absent.append(symbol)
    else:
        absent = by_recency(symbol for symbol in before if symbol not in on_lists)
        chosen.extend(symbol for symbol in listed if symbol in was_active)
        chosen.extend(absent)
        kept_absent.extend(absent)
        for symbol in listed:
            if len(chosen) >= settings.max_active:
                break
            if symbol not in was_active:
                admit(symbol)

    taken = set(chosen)
    return Selection(
        active=pinned + tuple(chosen),
        pinned=pinned,
        joined=tuple(s for s in chosen if s not in was_active),
        new=tuple(new),
        waiting=tuple(waiting),
        kept_absent=tuple(kept_absent),
        deactivated=tuple(s for s in before if s not in taken),
        complete=complete,
    )


# ── What the run writes (sections 6 and 7.1) ──────────────────────────────────

# Every tickers column a run writes, as db.trending.TICKER_COLUMNS lists them;
# repeated so the rules need no database module, and kept equal by a test.
TICKER_COLUMNS = (
    "symbol",
    "short_name",
    "quote_type",
    "first_seen",
    "last_seen",
    "active",
    "pinned",
    "first_bar_date",
    "first_trade_date",
    "summary_session",
    "last_close",
    "rsi_14",
    "recent_flags",
    "big_move_date",
)


def ticker_rows(
    selection: Selection,
    found: Sequence[Candidate],
    pinned_found: Mapping[str, TrendingItem],
    stored: Mapping[str, Mapping[str, Any]],
    *,
    now: dt.datetime,
    listing: Mapping[str, dt.date] | None = None,
) -> list[dict[str, Any]]:
    """The tickers rows the selection writes: one for every candidate (seen
    in this run), every pinned symbol and every stored symbol active or
    pinned before, so that one upsert records who is active now. A stored
    row keeps its first_seen, its history and its summary; its name, quote
    type and first trade date follow today's item where there is one.
    last_seen moves to now for a symbol seen or pinned, and stays put for
    the rest, so absence counts from the last sighting.

    first_trade_date is where the symbol's history counts from for
    eligibility (section 8): Yahoo's first trade date, or the symbol's date
    in the listing-date file (listing) when that is later, since Yahoo's
    metadata can be a former holder's. Stored that way, the dashboard needs
    no copy of the file, which the deployed package does not carry."""
    listing = listing or {}
    by_symbol = {c.symbol: c for c in found}
    active = set(selection.active)
    pinned = set(selection.pinned)
    symbols = sorted(
        set(by_symbol)
        | pinned
        | {s for s, row in stored.items() if row.get("active") or row.get("pinned")}
    )
    rows = []
    for symbol in symbols:
        base = dict(stored.get(symbol) or {})
        row = {column: base.get(column) for column in TICKER_COLUMNS}
        row["symbol"] = symbol
        seen = symbol in by_symbol or symbol in pinned
        if _missing(row["first_seen"]):
            row["first_seen"] = now
        if seen or _missing(row["last_seen"]):
            row["last_seen"] = now
        meta: Candidate | TrendingItem | None = by_symbol.get(
            symbol
        ) or pinned_found.get(symbol)
        if meta is not None:
            for column in ("short_name", "quote_type", "first_trade_date"):
                value = getattr(meta, column)
                if value is not None:
                    row[column] = value
        if symbol in listing:
            row["first_trade_date"] = history_start(
                None if _missing(row["first_trade_date"]) else row["first_trade_date"],
                listing[symbol],
            )
        row["active"] = symbol in active
        row["pinned"] = symbol in pinned
        rows.append(row)
    return rows


def snapshot_rows(
    fetches: Iterable[ListFetch], *, snapshot_ts: dt.datetime, session: dt.date
) -> list[dict[str, Any]]:
    """Every item every list returned, as trending_snapshots stores it: after
    symbol validation, before the other rules (section 6)."""
    return [
        {
            "snapshot_ts": snapshot_ts,
            "session_date": session,
            "symbol": item.symbol,
            "source": item.source,
            "list_name": item.list_name,
            "rank": item.rank,
            "pct_change": item.pct_change,
            "volume": item.volume,
            "market_cap": item.market_cap,
        }
        for fetch in fetches
        for item in fetch.items
    ]


def fetch_rows(
    fetches: Iterable[ListFetch], *, snapshot_ts: dt.datetime, session: dt.date
) -> list[dict[str, Any]]:
    """Each list's outcome, as trending_fetches stores it (T8)."""
    return [
        {
            "fetched_at": snapshot_ts,
            "session_date": session,
            "source": fetch.source,
            "list_name": fetch.list_name,
            "succeeded": fetch.succeeded,
            "items": len(fetch.items),
            "error": fetch.error,
        }
        for fetch in fetches
    ]


# ── Reused symbols (section 5) ────────────────────────────────────────────────


def listing_dates(path: Path | None = None) -> dict[str, dt.date]:
    """config/trending_listing_dates.txt as symbol to listing date, read by
    the split guard's parser (config.split_jump_acknowledged), so the same
    rules hold: a reason on every line, no future date, no repeat. It raises
    ValueError on a file it cannot read, and the run stops on it: the file is
    committed with the code. A symbol listed twice keeps its later date, the
    current holder's."""
    entries = config.split_jump_acknowledged(
        config.TRENDING_LISTING_DATES_FILE if path is None else path
    )
    dates: dict[str, dt.date] = {}
    for symbol, day in entries:
        if symbol not in dates or day > dates[symbol]:
            dates[symbol] = day
    return dates


def cut_before_listing(
    bars: pd.DataFrame, listed: dt.date | None
) -> tuple[pd.DataFrame, int]:
    """bars, indexed by each date's 00:00 UTC, without those dated before the
    listing date; and how many were dropped. They belong to the symbol's
    former holder (section 5)."""
    if listed is None or bars.empty:
        return bars, 0
    keep = pd.DatetimeIndex(bars.index).date >= listed
    return bars[keep], int((~keep).sum())


# ── Promotion eligibility (section 8) ─────────────────────────────────────────


def history_start(
    first_trade_date: dt.date | None, listed: dt.date | None
) -> dt.date | None:
    """Where a symbol's history counts from: the later of Yahoo's first trade
    date and its listing date, since Yahoo's metadata can still be a former
    holder's; None when neither is known."""
    known = [day for day in (first_trade_date, listed) if day is not None]
    return max(known) if known else None


def history_days(start: dt.date, as_of: dt.date) -> int:
    """NYSE trading days from start to as_of, both counted."""
    if start > as_of:
        return 0
    return len(pd.date_range(start, as_of, freq=TRADING_DAY))


def promotion_eligible(
    *,
    days_trending: int,
    pinned: bool,
    first_trade_date: dt.date | None,
    listed: dt.date | None,
    as_of: dt.date,
    settings: TrendingSettings,
) -> bool:
    """Shown as promotion-eligible (section 8): on the lists on at least
    TRENDING_PROMOTION_MIN_DAYS sessions, or pinned, and with at least
    PROMOTION_MIN_HISTORY_DAYS trading days of history. Unknown history is
    not enough history."""
    start = history_start(first_trade_date, listed)
    if start is None:
        return False
    attention = pinned or days_trending >= settings.promotion_min_days
    return (
        attention and history_days(start, as_of) >= settings.promotion_min_history_days
    )
