"""The lists the trending tier reads (docs/trending.md, section 3).

Two sources, both Yahoo's and both unofficial endpoints:

  YahooScreenerSource  the predefined screeners most_actives, day_gainers and
                       day_losers, through yfinance's yf.screen (its own
                       session, as every other Yahoo call here).
  YahooTrendingSource  the trending-tickers list, the stocks people look up
                       most, through a plain request: yfinance 1.7 has no call
                       for it.

A source never raises. Each list it reads comes back as a ListFetch holding
its items, or the reason it failed, so the run can store what the other lists
returned, record each list's outcome and still end red (T8). The quote fields
read are checked for type: a renamed or retyped field fails its list, loudly,
rather than turning into missing values that the rules would quietly filter.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd
import requests

from quantflow.ingestion.trending_rules import (
    MAX_QUOTE_TYPE_CHARS,
    MAX_SHORT_NAME_CHARS,
    clean_text,
    shown,
    valid_symbol,
)
from quantflow.utils.logger import get_logger
from quantflow.utils.trading_calendar import EXCHANGE_TZ

logger = get_logger(__name__)

SCREENER_SOURCE = "yahoo_screener"
TRENDING_SOURCE = "yahoo_trending"
SCREENER_LISTS = ("most_actives", "day_gainers", "day_losers")
TRENDING_LIST = "trending_tickers"

# Items asked of each list. Yahoo's cap is 250; the active set takes 15.
LIST_COUNT = 25

TRENDING_URL = "https://query1.finance.yahoo.com/v1/finance/trending/US"
# Names the project, as Yahoo can tell who is calling; no personal contact.
USER_AGENT = "QuantFlow/1.0 (+https://github.com/logn1602/QuantFlow)"
TIMEOUT_SECONDS = 10
# The real answer is about 30 KB; anything far larger is not the list.
MAX_BODY_BYTES = 2_000_000

# The quote fields the sources read, which the trending request asks for.
QUOTE_FIELDS = (
    "symbol",
    "shortName",
    "quoteType",
    "regularMarketPrice",
    "regularMarketChangePercent",
    "regularMarketVolume",
    "marketCap",
    "firstTradeDateMilliseconds",
)

# trending_fetches.error is VARCHAR(200).
MAX_ERROR_CHARS = 200

# trending_snapshots.pct_change is NUMERIC(10, 4); volume and market_cap
# BIGINT. A value outside them could not be stored, and is no real quote.
_MAX_PCT_CHANGE = 1e5
_MAX_BIGINT = 2**63 - 1


@dataclass(frozen=True)
class TrendingItem:
    symbol: str  # validated (section 4.1) before anything else sees it
    source: str  # SCREENER_SOURCE or TRENDING_SOURCE
    list_name: str  # "trending_tickers", "most_actives", ...
    rank: int  # 1-based position in the list as Yahoo sent it
    pct_change: float | None
    volume: int | None
    market_cap: int | None
    price: float | None  # for the minimum price (section 4)
    quote_type: str | None  # "EQUITY", "ETF", ...
    short_name: str | None  # cleaned, at most 100 characters
    first_trade_date: dt.date | None  # Yahoo's, in New York (section 8)
    fetched_at: dt.datetime  # UTC, when its list answered


@dataclass(frozen=True)
class ListFetch:
    source: str
    list_name: str
    items: tuple[TrendingItem, ...]  # valid symbols only (section 4.1)
    error: str | None = None  # None: the list answered, with items or without

    @property
    def succeeded(self) -> bool:
        return self.error is None


class TrendingSource(Protocol):
    name: str

    def fetch(self) -> list[ListFetch]:
        """One ListFetch per list the source reads; never raises."""
        ...


class MalformedListError(ValueError):
    """An answer that is not the list it should be."""


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _number(quote: dict, field: str, position: int) -> float | None:
    """A numeric quote field: None when absent, null or not finite; a
    MalformedListError for any other type (a bool is not a number)."""
    value = quote.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise MalformedListError(
            f"item {position}: {field} is {type(value).__name__}, not a number"
        )
    number = float(value)
    return number if math.isfinite(number) else None


def _count(quote: dict, field: str, position: int) -> int | None:
    """A whole-number quote field (volume, market cap), within BIGINT; a
    negative or oversized one is treated as missing."""
    number = _number(quote, field, position)
    if number is None or not 0 <= number <= _MAX_BIGINT:
        return None
    return int(number)


def _text(quote: dict, field: str, position: int, limit: int) -> str | None:
    value = quote.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise MalformedListError(
            f"item {position}: {field} is {type(value).__name__}, not text"
        )
    return clean_text(value, limit)


def _first_trade_date(quote: dict, position: int) -> dt.date | None:
    """Yahoo's first trading date, from milliseconds since 1970 (negative for
    a listing before it), as the New York date. Through pandas, which takes a
    negative timestamp anywhere: datetime.fromtimestamp refuses one on
    Windows. A value outside pandas' range is treated as missing."""
    millis = _number(quote, "firstTradeDateMilliseconds", position)
    if millis is None:
        return None
    try:
        moment = pd.Timestamp(int(millis), unit="ms", tz="UTC")
    except (OverflowError, ValueError):
        return None
    return moment.tz_convert(EXCHANGE_TZ).date()


def items_from_quotes(
    source: str, list_name: str, quotes: object, fetched_at: dt.datetime
) -> tuple[TrendingItem, ...]:
    """One list's quotes as items, in Yahoo's order.

    An item whose symbol is not valid is dropped and logged (escaped), the
    rest kept; a symbol repeated in the list keeps its first, best, rank.
    Raises MalformedListError when quotes is not a list of objects, when a
    field read has the wrong type, or when the list had items and not one of
    them a valid symbol: what a renamed field would look like.
    """
    if not isinstance(quotes, list):
        raise MalformedListError(f"quotes is {type(quotes).__name__}, not a list")
    items: list[TrendingItem] = []
    rejected: list[str] = []
    seen: set[str] = set()
    for position, quote in enumerate(quotes, start=1):
        if not isinstance(quote, dict):
            raise MalformedListError(f"item {position} is not an object")
        symbol = quote.get("symbol")
        if not valid_symbol(symbol):
            rejected.append(shown(symbol))
            continue
        if symbol in seen:
            continue
        seen.add(symbol)
        pct_change = _number(quote, "regularMarketChangePercent", position)
        if pct_change is not None and abs(pct_change) >= _MAX_PCT_CHANGE:
            pct_change = None
        items.append(
            TrendingItem(
                symbol=symbol,
                source=source,
                list_name=list_name,
                rank=position,
                pct_change=pct_change,
                volume=_count(quote, "regularMarketVolume", position),
                market_cap=_count(quote, "marketCap", position),
                price=_number(quote, "regularMarketPrice", position),
                quote_type=_text(quote, "quoteType", position, MAX_QUOTE_TYPE_CHARS),
                short_name=_text(quote, "shortName", position, MAX_SHORT_NAME_CHARS),
                first_trade_date=_first_trade_date(quote, position),
                fetched_at=fetched_at,
            )
        )
    if rejected:
        logger.warning(
            f"{list_name}: dropped {len(rejected)} items that are not symbols: "
            f"{', '.join(rejected[:5])}{', ...' if len(rejected) > 5 else ''}"
        )
    if quotes and not items:
        raise MalformedListError(f"no valid symbol among {len(quotes)} items")
    return tuple(items)


# A URL's query string, which in yfinance's requests carries its session crumb.
_QUERY_STRING = re.compile(r"(https?://[^\s?#]+)[?#]\S*")


def _reason(error: BaseException) -> str:
    """A failure as trending_fetches stores it: the kind and the message,
    one line, at most MAX_ERROR_CHARS, with any URL's query string cut: it
    is noise in a reason, and in yfinance's requests holds a session token."""
    message = _QUERY_STRING.sub(r"\1?...", f"{type(error).__name__}: {error}")
    return clean_text(message, MAX_ERROR_CHARS) or type(error).__name__


def _failed(source: str, list_name: str, error: BaseException) -> ListFetch:
    reason = _reason(error)
    logger.warning(f"Trending list {list_name} ({source}) failed: {reason}")
    return ListFetch(source=source, list_name=list_name, items=(), error=reason)


class YahooScreenerSource:
    """Yahoo's predefined screeners through yf.screen, one call per list.

    For a predefined name yfinance sends the list's id and the count to
    Yahoo's predefined-screener endpoint, which applies its own definition,
    and returns the body's finance.result[0], raising on an HTTP error.
    """

    name = SCREENER_SOURCE

    def __init__(
        self,
        lists: Iterable[str] = SCREENER_LISTS,
        *,
        screen: Callable[..., Any] | None = None,
        now: Callable[[], dt.datetime] = _utcnow,
    ) -> None:
        unknown = [name for name in lists if name not in SCREENER_LISTS]
        if unknown:
            raise ValueError(f"not a screener list: {', '.join(unknown)}")
        self.lists = tuple(lists)
        self._screen = screen
        self._now = now

    def _call(self, list_name: str) -> Any:
        if self._screen is not None:
            return self._screen(list_name, count=LIST_COUNT)
        import yfinance as yf

        return yf.screen(list_name, count=LIST_COUNT)

    def _fetch_one(self, list_name: str) -> ListFetch:
        try:
            body = self._call(list_name)
            if not isinstance(body, dict):
                raise MalformedListError(f"answer is {type(body).__name__}")
            if "quotes" not in body:
                raise MalformedListError("answer has no quotes")
            items = items_from_quotes(self.name, list_name, body["quotes"], self._now())
        except Exception as error:
            return _failed(self.name, list_name, error)
        logger.info(f"Trending list {list_name}: {len(items)} items")
        return ListFetch(source=self.name, list_name=list_name, items=items)

    def fetch(self) -> list[ListFetch]:
        return [self._fetch_one(list_name) for list_name in self.lists]


class YahooTrendingSource:
    """Yahoo's trending-tickers list for the US, by a plain request.

    useQuotes=true makes every item carry its quote, so one request is
    enough, with no lookup per symbol. The request has a timeout and a
    User-Agent naming the project, and an answer far larger than the list
    is refused.
    """

    name = TRENDING_SOURCE
    lists = (TRENDING_LIST,)

    def __init__(
        self,
        *,
        get: Callable[..., Any] | None = None,
        now: Callable[[], dt.datetime] = _utcnow,
    ) -> None:
        self._get = get
        self._now = now

    def _request(self) -> Any:
        get = self._get if self._get is not None else requests.get
        params: dict[str, str | int] = {
            "count": LIST_COUNT,
            "useQuotes": "true",
            "fields": ",".join(QUOTE_FIELDS),
        }
        return get(
            TRENDING_URL,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=TIMEOUT_SECONDS,
        )

    def _quotes(self) -> object:
        response = self._request()
        response.raise_for_status()
        if len(response.content) > MAX_BODY_BYTES:
            raise MalformedListError(f"answer of {len(response.content)} bytes")
        body = response.json()
        finance = body.get("finance") if isinstance(body, dict) else None
        result = finance.get("result") if isinstance(finance, dict) else None
        if not isinstance(result, list) or not result:
            error = finance.get("error") if isinstance(finance, dict) else None
            raise MalformedListError(f"no result in the answer (error: {error!r})")
        if not isinstance(result[0], dict) or "quotes" not in result[0]:
            raise MalformedListError("the result has no quotes")
        return result[0]["quotes"]

    def fetch(self) -> list[ListFetch]:
        try:
            items = items_from_quotes(
                self.name, TRENDING_LIST, self._quotes(), self._now()
            )
        except Exception as error:
            return [_failed(self.name, TRENDING_LIST, error)]
        logger.info(f"Trending list {TRENDING_LIST}: {len(items)} items")
        return [ListFetch(source=self.name, list_name=TRENDING_LIST, items=items)]


def sources_for(lists: Iterable[str]) -> list[TrendingSource]:
    """The sources that read the given lists (TRENDING_LISTS), each reading
    only those of its lists named."""
    wanted = list(lists)
    sources: list[TrendingSource] = []
    if TRENDING_LIST in wanted:
        sources.append(YahooTrendingSource())
    screener_lists = [name for name in wanted if name in SCREENER_LISTS]
    if screener_lists:
        sources.append(YahooScreenerSource(screener_lists))
    return sources
