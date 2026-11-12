"""tests/unit/test_trending_pipeline.py

One run of the trending tier (docs/trending.md, 7.1 and section 14), offline:
fake lists, a fake Yahoo download, fake FinBERT and RSS, and the fake
database (tests/conftest.py), whose stored rows the next run reads back.

What it checks: a run stores every list's items and outcome, chooses and
records the active set, replaces each active symbol's bars, summarises them,
stores its news in trending_news, and touches no core table; a failed list
(T8) or a failed price replacement ends the run red while the rest still
runs; an empty active set calls no stage; and the stages' run() functions
are never called.
"""

import dataclasses
import datetime as dt
import email.message
import io
import re
import socket
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.exc import IntegrityError, OperationalError

from quantflow.db import trending as db_trending
from quantflow.features import anomalies, indicators, sentiment
from quantflow.ingestion import trending_rules as rules
from quantflow.ingestion.trending_sources import ListFetch, TrendingItem
from quantflow.pipelines import trending

pytestmark = pytest.mark.trending

# The real fetcher, kept before the world fixture fakes it.
REAL_FETCH_RSS = sentiment.fetch_rss

NOW = pd.Timestamp("2026-11-10 23:10", tz="UTC")  # Tuesday 18:10 New York
SESSION = dt.date(2026, 11, 10)
CORE = ("AAPL", "MSFT")
SETTINGS = rules.TrendingSettings(
    pinned=("SPCX",),
    lists=rules.KNOWN_LISTS,
    max_active=3,
    max_new_per_run=5,
    inactive_after=5,
    min_price=5.0,
    min_market_cap=None,
    promotion_min_days=20,
    promotion_min_history_days=504,
)
DAYS = pd.bdate_range("2026-05-11", "2026-11-10")
CORE_TABLES = (
    "raw_prices",
    "news_sentiment",
    "technical_indicators",
    "anomalies",
    "forecasts",
    "model_metrics",
    "backtest_results",
)


def _item(symbol, list_name, rank, price=50.0, quote_type="EQUITY"):
    source = "yahoo_trending" if list_name == "trending_tickers" else "yahoo_screener"
    return TrendingItem(
        symbol=symbol,
        source=source,
        list_name=list_name,
        rank=rank,
        pct_change=2.5,
        volume=1_000_000,
        market_cap=10**10,
        price=price,
        quote_type=quote_type,
        short_name=f"{symbol} Inc",
        first_trade_date=dt.date(2015, 3, 2),
        fetched_at=NOW.to_pydatetime(),
    )


class FakeSource:
    def __init__(self, *fetches):
        self.fetches = list(fetches)
        self.name = "fake"

    def fetch(self):
        return list(self.fetches)


def _answered(list_name, *symbols, start=1):
    source = "yahoo_trending" if list_name == "trending_tickers" else "yahoo_screener"
    items = tuple(_item(s, list_name, start + i) for i, s in enumerate(symbols))
    return ListFetch(source=source, list_name=list_name, items=items)


def _failed(list_name):
    return ListFetch(
        source="yahoo_screener", list_name=list_name, items=(), error="HTTPError: 503"
    )


def _yahoo(days, close=20.0):
    """What yf.download returns for daily bars: naive exchange dates."""
    closes = close + 0.05 * np.arange(len(days)) + np.sin(np.arange(len(days)))
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes + 1,
            "Low": closes - 1,
            "Close": closes,
            "Adj Close": closes,
            "Volume": 1_000_000,
        },
        index=pd.DatetimeIndex(days),
    )


class FakeYahoo:
    """yfinance_fetcher.download for some symbols; others get nothing."""

    def __init__(self, frames):
        self.frames = frames
        self.calls = []

    def __call__(self, symbol, **kwargs):
        self.calls.append((symbol, kwargs))
        frame = self.frames.get(symbol)
        if frame is None:
            return pd.DataFrame()
        if "start" in kwargs:
            frame = frame[frame.index >= pd.Timestamp(kwargs["start"])]
        return frame.copy()


@pytest.fixture
def world(fake_db, monkeypatch):
    """The fake database read back the way db.trending reads Postgres, and
    FinBERT, RSS and the politeness pause faked."""

    def load_tickers(symbols):
        return [
            dict(row)
            for row in fake_db.tables["tickers"]
            if row.get("active") or row.get("pinned") or row["symbol"] in symbols
        ]

    def counted_sessions(limit):
        complete = {}
        for row in fake_db.tables["trending_fetches"]:
            key = (row["fetched_at"], row["session_date"])
            complete[key] = complete.get(key, True) and row["succeeded"]
        days = sorted({day for (_, day), ok in complete.items() if ok}, reverse=True)
        return days[:limit]

    monkeypatch.setattr(db_trending, "load_tickers", load_tickers)
    monkeypatch.setattr(db_trending, "counted_sessions", counted_sessions)

    calls = {"finbert": 0, "rss": []}

    def get_finbert():
        calls["finbert"] += 1
        return object()

    def fetch_rss(symbol, feeds=None):
        calls["rss"].append((symbol, feeds))
        return [
            {
                "headline": f"{symbol} news {i}",
                "url": "https://e.x",
                "published_at": NOW.to_pydatetime(),
                "source": "rss",
            }
            for i in range(2)
        ]

    def analyze_batch(headlines):
        return [
            {
                "sentiment": "positive",
                "score_pos": 0.6,
                "score_neg": 0.1,
                "score_neu": 0.3,
                "compound": 0.5,
            }
            for _ in headlines
        ]

    monkeypatch.setattr(sentiment, "get_finbert", get_finbert)
    monkeypatch.setattr(sentiment, "fetch_rss", fetch_rss)
    monkeypatch.setattr(sentiment, "analyze_batch", analyze_batch)

    # The stages' run() read and write the core's tables, and turn an empty
    # list into the core's tickers: the tier must never call them.
    def never(*args, **kwargs):
        raise AssertionError("a core stage's run() was called")

    for module in (indicators, anomalies, sentiment):
        monkeypatch.setattr(module, "run", never)
    fake_db.calls = calls
    return fake_db


def _run(sources, yahoo, *, now=NOW, listing=None, **changes):
    return trending.run(
        core=CORE,
        settings=dataclasses.replace(SETTINGS, **changes),
        sources=sources,
        listing={} if listing is None else listing,
        now=now,
        get=yahoo,
        pause=0,
    )


def _all_lists(*symbols):
    return [
        FakeSource(
            _answered("trending_tickers", *symbols),
            _answered("most_actives", *symbols[::-1]),
            _answered("day_gainers"),
            _answered("day_losers"),
        )
    ]


def _tickers(db):
    return {row["symbol"]: row for row in db.tables["tickers"]}


def _no_core_writes(db):
    """No row in a core table, and no statement of any kind naming one: an
    UPDATE or a TRUNCATE is as much a write as an INSERT."""
    for table in CORE_TABLES:
        assert db.tables[table] == [], table
        named = [
            sql for sql, _ in db.statements if re.search(rf"\b{table}\b", sql, re.I)
        ]
        assert named == [], table


# ── A full run ────────────────────────────────────────────────────────────────


def test_a_run_stores_the_lists_chooses_and_fills_the_active_set(world):
    yahoo = FakeYahoo({s: _yahoo(DAYS) for s in ("SPCX", "HIMS", "TTWO", "AAON")})
    result = _run(_all_lists("HIMS", "TTWO", "AAON", "ZETA"), yahoo)

    assert result.exit_code == 0 and result.failed_lists == []
    assert result.session == SESSION
    assert result.active == ("SPCX", "HIMS", "ZETA", "TTWO")
    # Every item before the rules, and every list's outcome.
    assert len(world.tables["trending_snapshots"]) == 8
    assert [r["succeeded"] for r in world.tables["trending_fetches"]] == [True] * 4
    # Prices for the active symbols that Yahoo answered; ZETA got nothing.
    assert result.prices_replaced == ["SPCX", "HIMS", "TTWO"]
    assert result.prices_skipped == ["ZETA"]
    stored = {r["ticker"] for r in world.tables["trending_prices"]}
    assert stored == {"SPCX", "HIMS", "TTWO"}
    tickers = _tickers(world)
    assert tickers["HIMS"]["first_bar_date"] == DAYS[0].date()
    assert tickers["HIMS"]["summary_session"] == SESSION
    assert tickers["HIMS"]["rsi_14"] is not None
    assert tickers["SPCX"]["pinned"] and tickers["SPCX"]["active"]
    assert not tickers["AAON"]["active"]  # a candidate past the cap: seen, inactive
    # News: RSS only, from each symbol's own feed, into trending_news.
    assert world.calls["finbert"] == 1
    assert world.calls["rss"][0] == ("SPCX", [rules.rss_feed("SPCX")])
    assert {r["ticker"] for r in world.tables["trending_news"]} == set(result.active)
    _no_core_writes(world)


def test_a_new_symbol_downloads_six_months_and_a_known_one_from_its_first_bar(world):
    yahoo = FakeYahoo({"HIMS": _yahoo(DAYS)})
    _run(_all_lists("HIMS"), yahoo, pinned=())
    assert yahoo.calls[-1] == (
        "HIMS",
        {"period": "6mo", "interval": "1d", "auto_adjust": False},
    )
    first = _tickers(world)["HIMS"]["first_bar_date"]
    assert first == DAYS[0].date()

    _run(_all_lists("HIMS"), yahoo, pinned=(), now=NOW + pd.Timedelta(days=1))
    assert yahoo.calls[-1] == (
        "HIMS",
        {"start": first.isoformat(), "interval": "1d", "auto_adjust": False},
    )


def test_bars_before_a_listing_date_are_dropped_and_history_starts_there(world):
    yahoo = FakeYahoo({"SPCX": _yahoo(DAYS)})
    _run(_all_lists(), yahoo, listing={"SPCX": dt.date(2026, 6, 12)})
    spcx = [r for r in world.tables["trending_prices"] if r["ticker"] == "SPCX"]
    assert min(r["ts"] for r in spcx) == pd.Timestamp("2026-06-12", tz="UTC")
    assert _tickers(world)["SPCX"]["first_bar_date"] == dt.date(2026, 6, 12)


def test_the_stored_first_trade_date_carries_a_later_listing_date(world):
    """Eligibility counts history from the later of Yahoo's first trade date
    and the listing date. The deployed dashboard has no copy of the listing
    file, so the run stores that later date where the tab reads it."""
    yahoo = FakeYahoo({"SPCX": _yahoo(DAYS), "HIMS": _yahoo(DAYS)})
    _run(
        _all_lists("HIMS"),  # HIMS's item: first trade 2015-03-02
        yahoo,
        listing={"SPCX": dt.date(2026, 6, 12), "HIMS": dt.date(2021, 1, 4)},
    )
    tickers = _tickers(world)
    assert tickers["SPCX"]["first_trade_date"] == dt.date(2026, 6, 12)  # no item
    assert tickers["HIMS"]["first_trade_date"] == dt.date(2021, 1, 4)


def _bars_of(db, symbol):
    return [r for r in db.tables["trending_prices"] if r["ticker"] == symbol]


def test_a_download_that_does_not_reach_back_keeps_the_stored_bars(world):
    yahoo = FakeYahoo({"SPCX": _yahoo(DAYS), "HIMS": _yahoo(DAYS)})
    _run(_all_lists("HIMS"), yahoo)
    before = _bars_of(world, "HIMS")

    yahoo.frames["HIMS"] = _yahoo(DAYS[40:])  # Yahoo lost its first 40 days
    result = _run(_all_lists("HIMS"), yahoo, now=NOW + pd.Timedelta(days=1))
    assert result.prices_skipped == ["HIMS"] and result.exit_code == 0
    assert _bars_of(world, "HIMS") == before


def test_a_download_that_ends_early_keeps_the_stored_bars(world):
    """Adversarial F3: a download missing its last days would shorten the
    history at its end, and the summary would go back in time."""
    yahoo = FakeYahoo({"SPCX": _yahoo(DAYS), "HIMS": _yahoo(DAYS)})
    _run(_all_lists("HIMS"), yahoo)
    before = _bars_of(world, "HIMS")

    yahoo.frames["HIMS"] = _yahoo(DAYS[:-5])  # Yahoo lost its last five days
    result = _run(_all_lists("HIMS"), yahoo, now=NOW + pd.Timedelta(days=1))
    assert result.prices_skipped == ["HIMS"] and result.exit_code == 0
    assert _bars_of(world, "HIMS") == before
    assert _tickers(world)["HIMS"]["summary_session"] == SESSION


def test_a_price_too_large_to_store_skips_the_symbol(world):
    frame = _yahoo(DAYS)
    frame.iloc[0, frame.columns.get_loc("High")] = 2e8
    result = _run(_all_lists("HIMS"), FakeYahoo({"HIMS": frame}), pinned=())
    assert result.prices_skipped == ["HIMS"]
    assert world.tables["trending_prices"] == []


# ── Failures (section 14) ─────────────────────────────────────────────────────


def test_a_list_that_offers_nobody_is_read_as_failed(world):
    """Its items all dropped by the rules (a changed quoteType): stored as
    a failed list, so nobody is deactivated, and the run ends red."""
    yahoo = FakeYahoo({s: _yahoo(DAYS) for s in ("SPCX", "A1", "A2", "A3", "B1")})
    _run(_all_lists("A1", "A2", "A3"), yahoo, pinned=())

    changed = ListFetch(
        source="yahoo_screener",
        list_name="most_actives",
        items=tuple(
            _item(s, "most_actives", n, quote_type="STOCK")
            for n, s in enumerate(("A1", "A2", "A3"), 1)
        ),
    )
    sources = [
        FakeSource(
            _answered("trending_tickers", "B1"),
            changed,
            _answered("day_gainers"),
            _answered("day_losers"),
        )
    ]
    result = _run(sources, yahoo, pinned=(), now=NOW + pd.Timedelta(days=1))
    assert result.failed_lists == ["most_actives"] and result.exit_code == 1
    outcome = [
        r for r in world.tables["trending_fetches"] if r["list_name"] == "most_actives"
    ][-1]
    assert not outcome["succeeded"] and outcome["items"] == 0
    assert outcome["error"] == "none of its 3 items passed the rules: 3 not EQUITY"
    assert set(result.active) == {"A1", "A2", "A3"}  # kept: T8


def test_a_failed_list_stores_the_others_runs_everything_and_ends_red(world):
    """T8: the other lists' items and every list's outcome are stored, the
    run goes on, and only then does it exit 1."""
    yahoo = FakeYahoo({s: _yahoo(DAYS) for s in ("SPCX", "HIMS")})
    sources = [
        FakeSource(
            _answered("trending_tickers", "HIMS"),
            _failed("most_actives"),
            _answered("day_gainers"),
            _answered("day_losers"),
        )
    ]
    result = _run(sources, yahoo)

    assert result.failed_lists == ["most_actives"] and result.exit_code == 1
    outcomes = {r["list_name"]: r for r in world.tables["trending_fetches"]}
    assert not outcomes["most_actives"]["succeeded"]
    assert outcomes["most_actives"]["error"] == "HTTPError: 503"
    assert outcomes["trending_tickers"]["items"] == 1
    assert [r["symbol"] for r in world.tables["trending_snapshots"]] == ["HIMS"]
    assert result.prices_replaced == ["SPCX", "HIMS"]
    assert {r["ticker"] for r in world.tables["trending_news"]} == {"SPCX", "HIMS"}


def test_after_a_failed_list_nobody_is_deactivated(world):
    yahoo = FakeYahoo({s: _yahoo(DAYS) for s in ("A1", "A2", "A3", "B1")})
    _run(_all_lists("A1", "A2", "A3"), yahoo, pinned=())
    assert {s for s, r in _tickers(world).items() if r["active"]} == {"A1", "A2", "A3"}

    # The next day the list they were on fails; B1 is on the one that answered.
    later = NOW + pd.Timedelta(days=1)
    partial = [FakeSource(_failed("trending_tickers"), _answered("most_actives", "B1"))]
    result = _run(partial, yahoo, pinned=(), now=later)
    assert result.exit_code == 1
    assert set(result.active) == {"A1", "A2", "A3"}  # full: B1 waits for room
    assert {s for s, r in _tickers(world).items() if r["active"]} == {"A1", "A2", "A3"}


def test_every_list_failing_keeps_the_active_set_and_ends_red(world):
    yahoo = FakeYahoo({s: _yahoo(DAYS) for s in ("SPCX", "HIMS")})
    _run(_all_lists("HIMS"), yahoo)

    outage = [FakeSource(*(_failed(name) for name in rules.KNOWN_LISTS))]
    result = _run(outage, yahoo, now=NOW + pd.Timedelta(days=1))
    assert result.exit_code == 1 and len(result.failed_lists) == 4
    assert result.active == ("SPCX", "HIMS")
    assert result.prices_replaced == ["SPCX", "HIMS"]


def test_a_failed_price_replacement_keeps_the_old_bars_and_ends_red(world, monkeypatch):
    yahoo = FakeYahoo({s: _yahoo(DAYS) for s in ("SPCX", "HIMS")})
    real = db_trending.replace_prices

    def failing(symbol, rows, first_bar_date):
        if symbol == "SPCX":
            raise RuntimeError("connection lost")
        return real(symbol, rows, first_bar_date)

    monkeypatch.setattr(db_trending, "replace_prices", failing)
    result = _run(_all_lists("HIMS"), yahoo)
    assert result.prices_failed == ["SPCX"] and result.exit_code == 1
    assert result.prices_replaced == ["HIMS"]  # the next symbol still ran
    assert result.analysed == ["HIMS"]


def test_one_symbols_failed_download_or_news_is_a_warning_only(world, monkeypatch):
    def broken_download(symbol, **kwargs):
        if symbol == "HIMS":
            raise RuntimeError("Yahoo hiccup")
        return _yahoo(DAYS)

    world_fetch = sentiment.fetch_rss

    def fetch_rss(symbol, feeds=None):
        if symbol == "SPCX":
            raise RuntimeError("feed down")
        return world_fetch(symbol, feeds)

    monkeypatch.setattr(sentiment, "fetch_rss", fetch_rss)
    result = _run(_all_lists("HIMS", "TTWO"), broken_download)
    assert result.prices_skipped == ["HIMS"]
    assert result.sentiment_failed == ["SPCX"]
    assert result.exit_code == 0


def test_no_bars_replaced_for_any_active_symbol_ends_red(world):
    """Adversarial F5: each skip is a warning, but a run in which not one
    active symbol's bars were replaced did nothing a green run promises."""
    result = _run(_all_lists("HIMS"), FakeYahoo({}))
    assert result.prices_skipped == ["SPCX", "HIMS"] and result.prices_failed == []
    assert result.failed_steps == ["prices"] and result.exit_code == 1


def test_analytics_failing_for_one_symbol_warns_and_for_every_one_ends_red(
    world, monkeypatch
):
    real = trending.summarize
    broken = {"HIMS"}

    def summarize(rows, symbol):
        if symbol in broken:
            raise ValueError("bad bars")
        return real(rows, symbol)

    monkeypatch.setattr(trending, "summarize", summarize)
    yahoo = FakeYahoo({s: _yahoo(DAYS) for s in ("SPCX", "HIMS")})
    result = _run(_all_lists("HIMS"), yahoo)
    assert result.analytics_failed == ["HIMS"] and result.analysed == ["SPCX"]
    assert result.exit_code == 0
    assert _tickers(world)["HIMS"]["summary_session"] is None

    broken.add("SPCX")
    result = _run(_all_lists("HIMS"), yahoo, now=NOW + pd.Timedelta(days=1))
    assert result.failed_steps == ["analytics"] and result.exit_code == 1


@pytest.mark.parametrize("status", [503, 429])
def test_the_feed_host_failing_for_every_symbol_ends_red(world, monkeypatch, status):
    """The real failure, through the real fetch_rss and feedparser: the feed
    host answering every request with an error status. feedparser turns it
    into a result with no entries and fetch_rss into no articles, so nothing
    raises and nothing is logged above INFO (the Red Team pass, 2026-11-10):
    a step no symbol got a headline from has failed."""

    def refused(self, request, *args, **kwargs):
        url = getattr(request, "full_url", request)
        raise urllib.error.HTTPError(
            url, status, "refused", email.message.Message(), io.BytesIO(b"")
        )

    monkeypatch.setattr(sentiment, "fetch_rss", REAL_FETCH_RSS)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", refused)
    yahoo = FakeYahoo({s: _yahoo(DAYS) for s in ("SPCX", "HIMS")})
    result = _run(_all_lists("HIMS"), yahoo)
    assert result.sentiment_failed == []  # nothing raised
    assert result.failed_steps == ["sentiment"] and result.exit_code == 1
    assert result.scored == {"SPCX": 0, "HIMS": 0}


def test_one_symbol_without_headlines_is_no_failure(world, monkeypatch):
    world_fetch = sentiment.fetch_rss

    def fetch_rss(symbol, feeds=None):
        return [] if symbol == "HIMS" else world_fetch(symbol, feeds)

    monkeypatch.setattr(sentiment, "fetch_rss", fetch_rss)
    yahoo = FakeYahoo({s: _yahoo(DAYS) for s in ("SPCX", "HIMS")})
    result = _run(_all_lists("HIMS"), yahoo)
    assert result.scored == {"SPCX": 2, "HIMS": 0}
    assert result.failed_steps == [] and result.exit_code == 0


def test_headlines_the_model_scores_none_of_end_red(world, monkeypatch):
    """analyze_batch catches its own errors and leaves a headline it could
    not score as None: for every headline of every symbol, the step failed."""
    monkeypatch.setattr(sentiment, "analyze_batch", lambda heads: [None] * len(heads))
    yahoo = FakeYahoo({s: _yahoo(DAYS) for s in ("SPCX", "HIMS")})
    result = _run(_all_lists("HIMS"), yahoo)
    assert result.scored == {"SPCX": 0, "HIMS": 0}
    assert result.failed_steps == ["sentiment"] and result.exit_code == 1


def test_a_failed_news_write_ends_red_and_the_next_symbol_still_runs(
    world, monkeypatch
):
    """A feed or the model failing for a symbol is a warning; the database
    refusing the write is a fault, as a failed price replacement is."""
    real = sentiment.save_sentiment

    def save_sentiment(articles, symbol, table="news_sentiment"):
        if symbol == "SPCX":
            raise OperationalError("INSERT", {}, Exception("connection lost"))
        return real(articles, symbol, table=table)

    monkeypatch.setattr(sentiment, "save_sentiment", save_sentiment)
    yahoo = FakeYahoo({s: _yahoo(DAYS) for s in ("SPCX", "HIMS")})
    result = _run(_all_lists("HIMS"), yahoo)
    assert result.news_failed == ["SPCX"] and result.exit_code == 1
    assert {r["ticker"] for r in world.tables["trending_news"]} == {"HIMS"}


def test_a_failed_tickers_write_stores_no_fetch_either(world, monkeypatch):
    real = world.execute

    def failing(statement, params=None):
        if str(statement).lstrip().startswith("INSERT INTO tickers"):
            raise IntegrityError("INSERT", {}, Exception("boom"))
        return real(statement, params)

    monkeypatch.setattr(world, "execute", failing)
    with pytest.raises(IntegrityError):
        _run(_all_lists("HIMS"), FakeYahoo({}))
    assert world.tables["trending_fetches"] == []
    assert world.tables["trending_snapshots"] == []


def test_a_symbol_absent_for_five_counted_sessions_goes_on_the_fifth(world):
    """Across runs, as the database keeps them: A1 is on the lists on the
    first session only; it is kept while absent from fewer than
    TRENDING_INACTIVE_AFTER_DAYS (5) counted sessions, and goes on the
    fifth."""
    days = ["2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13", "2026-11-16"]
    yahoo = FakeYahoo({s: _yahoo(DAYS) for s in ("A1", "B1")})
    _run(_all_lists("A1"), yahoo, pinned=())
    for day in days[1:]:
        result = _run(
            _all_lists("B1"),
            yahoo,
            pinned=(),
            now=pd.Timestamp(f"{day} 23:10", tz="UTC"),
        )
        assert "A1" in result.active, day  # absent from 1 to 4 sessions
    result = _run(
        _all_lists("B1"),
        yahoo,
        pinned=(),
        now=pd.Timestamp("2026-11-17 23:10", tz="UTC"),
    )
    assert result.active == ("B1",)
    assert not _tickers(world)["A1"]["active"]


def test_a_pinned_symbol_that_is_not_an_equity_is_warned_about(world, caplog):
    sources = [
        FakeSource(
            ListFetch(
                source="yahoo_screener",
                list_name="most_actives",
                items=(_item("SPCX", "most_actives", 1, quote_type="ETF"),),
            ),
            _answered("trending_tickers", "HIMS"),
        )
    ]
    _run(sources, FakeYahoo({s: _yahoo(DAYS) for s in ("SPCX", "HIMS")}))
    assert any(
        "Pinned SPCX has quote type ETF" in r.getMessage() for r in caplog.records
    )


def test_the_feeds_are_read_with_a_socket_timeout_put_back_after(world, monkeypatch):
    """feedparser has no timeout: it reads through urllib, which waits as long
    as the socket default allows, forever unless one is set. The real
    fetch_rss sets it around each read, for the core and the tier alike."""
    import feedparser

    seen = []

    def parse(url):
        seen.append(socket.getdefaulttimeout())
        return feedparser.FeedParserDict(entries=[])

    monkeypatch.setattr(sentiment, "fetch_rss", REAL_FETCH_RSS)
    monkeypatch.setattr(sentiment.feedparser, "parse", parse)
    before = socket.getdefaulttimeout()
    _run(_all_lists("HIMS"), FakeYahoo({s: _yahoo(DAYS) for s in ("SPCX", "HIMS")}))
    assert seen == [sentiment.RSS_TIMEOUT_SECONDS] * 2
    assert socket.getdefaulttimeout() == before


def test_finbert_unavailable_ends_red_after_prices_and_analytics(world, monkeypatch):
    monkeypatch.setattr(sentiment, "get_finbert", lambda: None)
    result = _run(_all_lists("HIMS"), FakeYahoo({"HIMS": _yahoo(DAYS)}))
    assert result.finbert_failed and result.exit_code == 1
    assert "HIMS" in result.analysed
    assert world.tables["trending_news"] == []


def test_an_empty_active_set_calls_no_stage(world):
    """Each stage reads an empty ticker list as the core's watchlist, so the
    tier must hand none on: no download, no FinBERT, no RSS."""
    yahoo = FakeYahoo({})
    result = _run(_all_lists(), yahoo, pinned=())
    assert result.active == () and result.exit_code == 0
    assert yahoo.calls == []
    assert world.calls["finbert"] == 0 and world.calls["rss"] == []


# ── The summary (7.4) ─────────────────────────────────────────────────────────


def test_the_summary_counts_recent_flags_and_dates_the_latest_big_move():
    closes = np.full(len(DAYS), 20.0) + 0.01 * np.arange(len(DAYS))
    closes[-3] *= 1.6  # a 60% one-day move, three sessions ago
    frame = _yahoo(DAYS)
    frame["Close"] = closes
    rows = trending.price_rows(trending.as_daily_bars(frame), "HIMS")
    summary, jumps = trending.summarize(rows, "HIMS")
    assert summary["summary_session"] == DAYS[-1].date()
    assert summary["last_close"] == pytest.approx(round(closes[-1], 4))
    assert summary["big_move_date"] == DAYS[-2].date()  # the fall back counts too
    assert summary["recent_flags"] >= 1
    assert 0 <= summary["rsi_14"] <= 100
    assert len(jumps) == 2


def test_too_short_a_history_has_no_rsi_but_still_a_summary():
    rows = trending.price_rows(trending.as_daily_bars(_yahoo(DAYS[-12:])), "NEWCO")
    summary, _ = trending.summarize(rows, "NEWCO")
    assert summary["rsi_14"] is None and summary["recent_flags"] == 0
    assert summary["summary_session"] == DAYS[-1].date()


# ── The command ───────────────────────────────────────────────────────────────

SETTING_NAMES = (
    "TRENDING_PINNED",
    "TRENDING_LISTS",
    "TRENDING_MAX_ACTIVE",
    "TRENDING_MAX_NEW_PER_RUN",
    "TRENDING_INACTIVE_AFTER_DAYS",
    "TRENDING_MIN_PRICE",
    "TRENDING_MIN_MARKET_CAP",
    "TRENDING_PROMOTION_MIN_DAYS",
    "PROMOTION_MIN_HISTORY_DAYS",
)


@pytest.fixture
def command(monkeypatch):
    """main() as the workflow runs it, offline: blank trending settings, so
    no test depends on the developer's .env; a password, so the config
    check passes; and every table there."""
    for name in SETTING_NAMES:
        monkeypatch.setattr(trending.config, name, "")
    monkeypatch.setattr(trending.config, "DB_PASSWORD", "x")
    monkeypatch.setattr(trending.config, "TICKERS", ["AAPL"])
    monkeypatch.setattr(db_trending, "missing_tables", lambda: [])
    return monkeypatch


def test_every_trending_setting_is_blanked_by_the_command_fixture():
    from quantflow import config

    named = {
        name
        for name in vars(config)
        if (name.startswith("TRENDING_") or name.startswith("PROMOTION_"))
        and not name.endswith("_FILE")
    }
    assert named == set(SETTING_NAMES)


def test_a_bad_setting_exits_1_before_anything_runs(command, capsys):
    command.setattr(trending.config, "TRENDING_MAX_ACTIVE", "lots")
    command.setattr(trending, "run", lambda **kw: pytest.fail("ran"))
    assert trending.main([]) == 1
    assert "TRENDING_MAX_ACTIVE" in capsys.readouterr().err


def test_missing_tables_stop_the_run_naming_the_migration(command, capsys):
    """Before the owner runs 002_trending.sql: one clear line, not a
    traceback from the first INSERT after every list was fetched."""
    command.setattr(db_trending, "missing_tables", lambda: ["tickers", "trending_news"])
    command.setattr(trending, "run", lambda **kw: pytest.fail("ran"))
    assert trending.main([]) == 1
    err = capsys.readouterr().err
    assert "tickers, trending_news" in err
    assert "sql/migrations/002_trending.sql" in err


def test_the_cap_flag_limits_the_discovered_symbols(command):
    seen = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        return trending.RunResult(session=SESSION)

    command.setattr(trending, "run", fake_run)
    assert trending.main(["--cap", "3"]) == 0
    assert seen["settings"].max_active == 3
    assert seen["core"] == ["AAPL"]


def test_a_negative_cap_is_refused():
    with pytest.raises(SystemExit):
        trending.main(["--cap", "-1"])


def test_a_failed_summary_write_is_red_and_the_news_still_runs(world, monkeypatch):
    """The summaries were written outside any guard, so a failure there (a
    column grant missing from the writer role, say) escaped run(), skipping
    the news step and the run's last line (found in the full-diff review,
    2026-11-12). It is now recorded, the news still runs, and the run ends
    red."""

    def refuse(rows, update):
        raise OperationalError(
            "UPDATE tickers", {}, Exception("permission denied for table tickers")
        )

    monkeypatch.setattr(db_trending, "upsert_tickers", refuse)
    yahoo = FakeYahoo({s: _yahoo(DAYS) for s in ("SPCX", "HIMS")})
    result = _run(_all_lists("HIMS"), yahoo)
    assert result.summary_failed
    assert result.exit_code == 1
    assert {row["ticker"] for row in world.tables["trending_news"]} == {"SPCX", "HIMS"}


def test_a_new_big_move_warns_is_stored_and_never_stops_the_run(world, caplog):
    """The branch no fixture reached: a split-sized jump in the last ten
    sessions is a warning and a badge, never a stop (section 7.3)."""
    moved = _yahoo(DAYS)
    moved.iloc[-3:, :5] *= 1.6
    yahoo = FakeYahoo({"SPCX": _yahoo(DAYS), "HIMS": moved})
    result = _run(_all_lists("HIMS"), yahoo)
    assert result.exit_code == 0
    assert _tickers(world)["HIMS"]["big_move_date"] is not None
    assert any("HIMS: big move on" in r.getMessage() for r in caplog.records)
