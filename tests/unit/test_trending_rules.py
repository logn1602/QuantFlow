"""tests/unit/test_trending_rules.py

The trending tier's rules as pure functions (docs/trending.md, sections 4, 5
and 8): session dates, the filters and dedupe, the active set (the cap, the
new-symbol limit, pinned symbols outside both, the grace for absent symbols,
and T8's rule after a failed list), the rows a run writes, reused symbols and
promotion eligibility.

The core watchlist is passed in, never read from config.TICKERS.
"""

import dataclasses
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from quantflow.db import trending as db_trending
from quantflow.ingestion import trending_rules as rules
from quantflow.ingestion.trending_sources import (
    SCREENER_LISTS,
    ListFetch,
    TrendingItem,
    YahooScreenerSource,
    YahooTrendingSource,
)

pytestmark = pytest.mark.trending

NOW = pd.Timestamp("2026-11-10 23:10", tz="UTC")  # Tuesday, 18:10 New York
CORE = ("AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM")
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "trending"
SETTINGS = rules.TrendingSettings(
    pinned=("SPCX",),
    lists=rules.KNOWN_LISTS,
    max_active=15,
    max_new_per_run=5,
    inactive_after=5,
    min_price=5.0,
    min_market_cap=None,
    promotion_min_days=20,
    promotion_min_history_days=504,
)
# The latest counted sessions, newest first: the run's own session first.
COUNTED = [dt.date(2026, 11, d) for d in (10, 9, 6, 5, 4)]


def _settings(**changes):
    return dataclasses.replace(SETTINGS, **changes)


def _item(symbol, list_name="most_actives", rank=1, **fields):
    values = {
        "pct_change": 1.0,
        "volume": 1_000_000,
        "market_cap": 10**10,
        "price": 50.0,
        "quote_type": "EQUITY",
        "short_name": f"{symbol} Inc",
        "first_trade_date": None,
    } | fields
    source = "yahoo_trending" if list_name == "trending_tickers" else "yahoo_screener"
    return TrendingItem(
        symbol=symbol,
        source=source,
        list_name=list_name,
        rank=rank,
        fetched_at=NOW.to_pydatetime(),
        **values,
    )


def _answered(list_name, *items):
    source = "yahoo_trending" if list_name == "trending_tickers" else "yahoo_screener"
    return ListFetch(source=source, list_name=list_name, items=tuple(items))


def _failed(list_name):
    return ListFetch(
        source="yahoo_screener", list_name=list_name, items=(), error="HTTPError: 503"
    )


def _candidates(symbols, list_name="most_actives"):
    fetch = _answered(
        list_name, *(_item(s, list_name, rank) for rank, s in enumerate(symbols, 1))
    )
    return rules.candidates([fetch], core=CORE, settings=SETTINGS)


def _stored(symbol, *, active=True, pinned=False, history=True, seen=NOW, **more):
    row = dict.fromkeys(rules.TICKER_COLUMNS)
    row.update(
        symbol=symbol,
        active=active,
        pinned=pinned,
        first_seen=NOW - pd.Timedelta(days=60),
        last_seen=seen,
        first_bar_date=dt.date(2026, 5, 11) if history else None,
    )
    return row | more


def _choose(found, stored_rows=(), *, complete=True, counted=COUNTED, **changes):
    stored = {row["symbol"]: row for row in stored_rows}
    return rules.choose_active(
        found,
        stored,
        settings=_settings(**changes),
        complete=complete,
        counted=counted,
        core=CORE,
    )


# ── Sessions ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("at", "session"),
    [
        ("2026-11-10 22:30", "2026-11-10"),  # the evening run: 17:30 New York
        ("2026-11-11 04:12", "2026-11-10"),  # after midnight UTC: 23:12 New York
        ("2026-11-11 13:00", "2026-11-10"),  # before the next open: 08:00
        ("2026-11-11 14:29", "2026-11-10"),  # 09:29: not open yet
        ("2026-11-11 14:30", "2026-11-11"),  # 09:30: the new session
        ("2026-11-11 18:00", "2026-11-11"),  # a manual run during the session
        ("2026-11-14 15:00", "2026-11-13"),  # a Saturday
        ("2026-11-16 12:00", "2026-11-13"),  # Monday before the open
        ("2026-11-26 18:00", "2026-11-25"),  # Thanksgiving, the exchange shut
        ("2026-10-30 13:31", "2026-10-30"),  # 09:31 EDT
        ("2026-11-02 14:29", "2026-10-30"),  # 09:29 EST, the Monday after DST ended
    ],
)
def test_a_fetch_is_dated_by_the_latest_session_that_had_opened(at, session):
    assert rules.session_date(pd.Timestamp(at, tz="UTC")) == dt.date.fromisoformat(
        session
    )


def test_a_naive_time_is_read_as_utc():
    assert rules.session_date(dt.datetime(2026, 11, 11, 4, 12)) == dt.date(2026, 11, 10)


def test_the_evening_and_after_midnight_runs_of_one_session_count_once():
    evening = rules.session_date(pd.Timestamp("2026-11-09 22:30", tz="UTC"))
    late = rules.session_date(pd.Timestamp("2026-11-10 04:12", tz="UTC"))
    assert evening == late == dt.date(2026, 11, 9)


# ── Candidates (4.2) ──────────────────────────────────────────────────────────


def test_the_rules_keep_equities_off_the_core_list_and_over_the_floor():
    fetch = _answered(
        "most_actives",
        _item("KEEP", rank=1),
        _item("ETFX", rank=2, quote_type="ETF"),
        _item("ANTH.PVT", rank=3, quote_type="PRIVATE_COMPANY"),
        _item("NOTYPE", rank=4, quote_type=None),
        _item("AAPL", rank=5),
        _item("CHEAP", rank=6, price=4.99),
        _item("NOPRICE", rank=7, price=None),
        _item("FIVE", rank=8, price=5.0),
    )
    found = rules.candidates([fetch], core=CORE, settings=SETTINGS)
    assert [c.symbol for c in found] == ["KEEP", "FIVE"]


def test_a_zero_price_floor_keeps_cheap_and_priceless_equities():
    fetch = _answered(
        "most_actives",
        _item("CHEAP", rank=1, price=0.5),
        _item("NOPRICE", rank=2, price=None),
    )
    found = rules.candidates([fetch], core=CORE, settings=_settings(min_price=0.0))
    assert [c.symbol for c in found] == ["CHEAP", "NOPRICE"]


def test_the_market_cap_floor_when_set():
    fetch = _answered(
        "most_actives",
        _item("BIG", rank=1, market_cap=3 * 10**9),
        _item("SMALL", rank=2, market_cap=10**9),
        _item("UNKNOWN", rank=3, market_cap=None),
    )
    found = rules.candidates([fetch], core=CORE, settings=_settings(min_market_cap=2e9))
    assert [c.symbol for c in found] == ["BIG"]


def test_a_symbol_on_several_lists_keeps_every_list_and_its_best_rank():
    found = rules.candidates(
        [
            _answered("day_gainers", _item("HIMS", "day_gainers", 1, pct_change=12.0)),
            _answered(
                "most_actives", _item("HIMS", "most_actives", 9, pct_change=11.0)
            ),
            _answered("trending_tickers", _item("HIMS", "trending_tickers", 12)),
        ],
        core=CORE,
        settings=SETTINGS,
    )
    (hims,) = found
    assert hims.lists == ("trending_tickers", "most_actives", "day_gainers")
    assert (hims.best_rank, hims.best_list) == (1, "day_gainers")
    assert hims.pct_change == 12.0  # the best-ranked item's fields


def test_equal_ranks_go_to_the_list_named_first():
    found = rules.candidates(
        [
            _answered("day_losers", _item("LOSE", "day_losers", 1, volume=9 * 10**9)),
            _answered("trending_tickers", _item("LOOK", "trending_tickers", 1)),
            _answered("most_actives", _item("BUSY", "most_actives", 1)),
        ],
        core=CORE,
        settings=SETTINGS,
    )
    assert [c.symbol for c in found] == ["LOOK", "BUSY", "LOSE"]


def test_a_symbol_at_one_rank_on_two_lists_takes_the_earlier_lists_item():
    found = rules.candidates(
        [
            _answered("day_gainers", _item("TWIN", "day_gainers", 2, pct_change=9.0)),
            _answered("most_actives", _item("TWIN", "most_actives", 2, pct_change=1.0)),
        ],
        core=CORE,
        settings=SETTINGS,
    )
    assert found[0].best_list == "most_actives" and found[0].pct_change == 1.0


def test_further_ties_go_to_higher_volume_then_to_the_symbol():
    """Lists outside TRENDING_LISTS share the last place, so only volume and
    the symbol are left to order them, and the order is still fixed."""
    only = _settings(lists=("trending_tickers",))
    found = rules.candidates(
        [
            _answered("most_actives", _item("BBB", "most_actives", 1, volume=5)),
            _answered("day_gainers", _item("CCC", "day_gainers", 1, volume=9)),
            _answered("day_losers", _item("AAA", "day_losers", 1, volume=5)),
            _answered("most_actives", _item("ZZZ", "most_actives", 2, volume=None)),
        ],
        core=CORE,
        settings=only,
    )
    assert [c.symbol for c in found] == ["CCC", "AAA", "BBB", "ZZZ"]


def test_the_order_does_not_depend_on_the_order_the_lists_arrived():
    fetches = [
        _answered("most_actives", _item("A1", rank=1), _item("A2", rank=2)),
        _answered("day_gainers", _item("G1", "day_gainers", 1)),
        _answered("trending_tickers", _item("T1", "trending_tickers", 2)),
    ]
    forward = rules.candidates(fetches, core=CORE, settings=SETTINGS)
    backward = rules.candidates(fetches[::-1], core=CORE, settings=SETTINGS)
    assert forward == backward
    assert [c.symbol for c in forward] == ["A1", "G1", "T1", "A2"]


def test_pinned_symbols_are_not_candidates_but_their_items_are_kept():
    fetches = [
        _answered("most_actives", _item("SPCX", rank=3, quote_type="ETF")),
        _answered("trending_tickers", _item("SPCX", "trending_tickers", 1)),
    ]
    assert rules.candidates(fetches, core=CORE, settings=SETTINGS) == []
    found = rules.pinned_items(fetches, SETTINGS)
    assert found["SPCX"].list_name == "trending_tickers"


def test_a_list_whose_every_item_fails_the_rules_is_read_as_failed():
    """What a changed field value looks like (quoteType "STOCK" for
    "EQUITY"): the list answers, every item is dropped, and everyone on it
    would look absent. Read as failed, T8 keeps them (adversarial F1)."""
    offered_nobody = _answered(
        "most_actives",
        _item("ZZA", rank=1, quote_type="STOCK"),
        _item("ZZB", rank=2, quote_type="STOCK", price=1.0),
    )
    fine = _answered("day_gainers", _item("ZZC", "day_gainers"))
    empty = _answered("day_losers")
    failed = _failed("trending_tickers")
    checked = rules.offering_nobody(
        [offered_nobody, fine, empty, failed], core=CORE, settings=SETTINGS
    )
    assert checked[1:] == [fine, empty, failed]
    assert checked[0].items == () and not checked[0].succeeded
    assert checked[0].error == ("none of its 2 items passed the rules: 2 not EQUITY")


def test_a_list_of_core_or_pinned_symbols_alone_is_no_failure():
    core_only = _answered("most_actives", _item("NVDA"), _item("XX", quote_type="ETF"))
    pinned = _answered("day_gainers", _item("SPCX", "day_gainers", quote_type="ETF"))
    fetches = [core_only, pinned]
    assert rules.offering_nobody(fetches, core=CORE, settings=SETTINGS) == fetches


def test_this_fetchs_session_counts_when_every_list_answered():
    prior = [dt.date(2026, 11, d) for d in (9, 6, 5, 4, 3)]
    today = dt.date(2026, 11, 10)
    assert rules.counted_with(prior, today, complete=True, limit=5) == [
        today,
        *prior[:4],
    ]
    assert rules.counted_with(prior, today, complete=False, limit=5) == prior
    # A second complete fetch of a session already counted counts it once.
    assert rules.counted_with([today, *prior], today, complete=True, limit=3) == [
        today,
        *prior[:2],
    ]


def test_a_failed_list_contributes_nothing():
    found = rules.candidates(
        [_failed("most_actives"), _answered("day_gainers", _item("G1", "day_gainers"))],
        core=CORE,
        settings=SETTINGS,
    )
    assert [c.symbol for c in found] == ["G1"]


def test_the_captured_lists_through_the_rules():
    """The 2026-11-10 capture: nothing filtered stays, and what is left is
    one entry per symbol (the design's estimate was 70 to 80 a day)."""

    def load(name):
        return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))

    screens = {name: load(name) for name in SCREENER_LISTS}
    fetches = YahooScreenerSource(screen=lambda n, count: screens[n]).fetch()

    class Response:
        status_code = 200
        content = b"{}"

        def raise_for_status(self):
            return None

        def json(self):
            return load("trending_tickers")

    fetches += YahooTrendingSource(get=lambda *a, **k: Response()).fetch()
    found = rules.candidates(fetches, core=CORE, settings=SETTINGS)

    symbols = [c.symbol for c in found]
    assert len(symbols) == len(set(symbols))
    assert not {"NVDA", "MSFT", "SPCX", "ANTH.PVT", "^TNX"} & set(symbols)
    assert all(c.quote_type == "EQUITY" and c.price >= 5 for c in found)
    assert 40 <= len(found) <= 100
    ranks = [(c.best_rank, rules.KNOWN_LISTS.index(c.best_list)) for c in found]
    assert ranks == sorted(ranks)


# ── The active set (4.3) ──────────────────────────────────────────────────────


def test_the_cap_takes_the_best_ranked_and_pinned_symbols_sit_outside_it():
    symbols = [f"S{i:02d}" for i in range(20)]
    stored = [_stored(s, active=False) for s in symbols]
    chosen = _choose(_candidates(symbols), stored)
    assert chosen.active == ("SPCX", *symbols[:15])
    assert chosen.new == () and chosen.waiting == ()


def test_at_most_five_new_symbols_join_and_the_rest_wait_without_a_place():
    symbols = [f"N{i}" for i in range(7)] + ["H1", "H2"]
    stored = [_stored("H1", active=False), _stored("H2", active=False)]
    chosen = _choose(_candidates(symbols), stored)
    assert chosen.new == ("N0", "N1", "N2", "N3", "N4")
    assert chosen.waiting == ("N5", "N6")
    assert chosen.active == ("SPCX", "N0", "N1", "N2", "N3", "N4", "H1", "H2")


def test_only_new_symbols_the_cap_had_room_for_are_waiting():
    """Found measuring a full-size run (2026-11-10): with most candidates new,
    as churn makes them, every new symbol past the limit was listed as
    waiting, 64 of 72, ranks far past the cap included, although no limit
    would have let those in. A symbol waits only if a place was left for it."""
    symbols = [f"N{i:02d}" for i in range(30)]
    chosen = _choose(_candidates(symbols), [])
    assert chosen.new == tuple(symbols[:5])
    assert chosen.waiting == tuple(symbols[5:15])  # the cap's 10 other places
    assert chosen.active == ("SPCX", *symbols[:5])


def test_pinned_symbols_are_outside_the_new_symbol_limit_too():
    chosen = _choose(
        _candidates(["D1", "D2"]),
        [_stored("D1", active=False), _stored("D2", active=False)],
        pinned=("SPCX", "XYZ"),
        max_active=2,
        max_new_per_run=0,
    )
    assert chosen.active == ("SPCX", "XYZ", "D1", "D2")
    assert chosen.new == ()


def test_a_symbol_active_before_without_history_is_not_new():
    chosen = _choose(
        _candidates(["OLD"]), [_stored("OLD", history=False)], max_new_per_run=0
    )
    assert chosen.active == ("SPCX", "OLD") and chosen.new == ()


def test_absent_symbols_are_kept_by_recency_while_in_grace_and_room():
    stored = [
        _stored("P1", seen=pd.Timestamp("2026-11-09 23:00", tz="UTC")),
        _stored("P2", seen=pd.Timestamp("2026-10-29 23:00", tz="UTC")),  # 5 absences
        _stored("P3", seen=pd.Timestamp("2026-11-05 23:00", tz="UTC")),  # 3 absences
        _stored("C1", active=False),
        _stored("C2", active=False),
    ]
    found = _candidates(["C1", "C2"])

    roomy = _choose(found, stored, max_active=4)
    assert roomy.active == ("SPCX", "C1", "C2", "P1", "P3")
    assert roomy.kept_absent == ("P1", "P3")
    assert roomy.deactivated == ("P2",)

    tight = _choose(found, stored, max_active=3)
    assert tight.active == ("SPCX", "C1", "C2", "P1")
    assert tight.deactivated == ("P2", "P3")


def test_absence_counts_counted_sessions_after_the_last_sighting():
    assert rules.absences(pd.Timestamp("2026-11-05 23:00", tz="UTC"), COUNTED) == 3
    # Seen after midnight UTC: still the 5th's session.
    assert rules.absences(pd.Timestamp("2026-11-06 03:00", tz="UTC"), COUNTED) == 3
    assert rules.absences(pd.Timestamp("2026-11-10 23:10", tz="UTC"), COUNTED) == 0
    assert rules.absences(None, COUNTED) == 5
    assert rules.absences(pd.NaT, COUNTED) == 5


def test_an_active_symbol_ranked_below_the_cap_is_deactivated():
    symbols = [f"S{i}" for i in range(4)]
    stored = [_stored(s, active=s == "S3") for s in symbols]
    chosen = _choose(_candidates(symbols), stored, max_active=3)
    assert chosen.active == ("SPCX", "S0", "S1", "S2")
    assert chosen.deactivated == ("S3",)
    assert chosen.joined == ("S0", "S1", "S2")


def test_a_cap_of_zero_tracks_only_the_pinned():
    chosen = _choose(_candidates(["A1"]), [_stored("OLD")], max_active=0)
    assert chosen.active == ("SPCX",) and chosen.deactivated == ("OLD",)


def test_after_a_failed_list_nobody_is_deactivated_and_newcomers_take_only_room():
    """T8: P2 is far past its grace, and on a complete session would go. A
    failed list could be the one it was on, so it stays."""
    stored = [
        _stored("P1", seen=pd.Timestamp("2026-11-09 23:00", tz="UTC")),
        _stored("P2", seen=pd.Timestamp("2026-10-01 23:00", tz="UTC")),
        _stored("P3"),
    ]
    found = _candidates(["P3", "C1", "C2", "C3"])
    chosen = _choose(found, stored, complete=False, max_active=4)
    assert chosen.active == ("SPCX", "P3", "P1", "P2", "C1")
    assert chosen.deactivated == ()
    assert chosen.kept_absent == ("P1", "P2")
    assert chosen.new == ("C1",) and chosen.joined == ("C1",)
    assert not chosen.complete


def test_after_a_failed_list_a_full_set_takes_nobody_new():
    stored = [_stored(f"P{i}") for i in range(3)]
    chosen = _choose(_candidates(["C1"]), stored, complete=False, max_active=2)
    # The cap was lowered to 2 with 3 active: nobody goes until a complete
    # session, and nobody joins.
    assert chosen.active == ("SPCX", "P0", "P1", "P2")
    assert chosen.joined == () and chosen.deactivated == ()


def test_an_outage_of_every_list_keeps_everyone_and_adds_nobody():
    stored = [_stored("P1"), _stored("P2", seen=pd.Timestamp("2026-09-01", tz="UTC"))]
    chosen = _choose([], stored, complete=False, counted=[])
    assert chosen.active == ("SPCX", "P1", "P2")
    assert chosen.deactivated == () and chosen.joined == ()


@pytest.mark.parametrize("complete", [True, False])
def test_a_core_ticker_active_before_leaves_the_tier(complete):
    """A symbol added to TICKERS after the tier tracked it: rule 3 keeps it
    off the candidates, and it must not stay active as an absent symbol
    either, not even after a failed list (T8)."""
    stored = [_stored("NVDA"), _stored("P1")]
    chosen = _choose(_candidates(["C1"]), stored, complete=complete)
    assert "NVDA" not in chosen.active and "P1" in chosen.active
    assert chosen.deactivated == ("NVDA",)
    assert "NVDA" not in chosen.kept_absent


def test_the_selection_does_not_depend_on_the_stored_rows_order():
    stored = [_stored(f"P{i}", seen=NOW - pd.Timedelta(days=i)) for i in range(6)]
    found = _candidates(["C1"])
    one = _choose(found, stored, max_active=4)
    other = _choose(found, stored[::-1], max_active=4)
    assert one == other
    assert one.active == ("SPCX", "C1", "P0", "P1", "P2")


# ── The rows a run writes ─────────────────────────────────────────────────────


def test_ticker_rows_record_who_is_active_and_when_each_was_seen():
    earlier = NOW - pd.Timedelta(days=3)
    stored_rows = [
        _stored("SEEN", seen=earlier, short_name="Old Name", rsi_14=40.0),
        _stored("GONE", seen=earlier),
        _stored("SPCX", pinned=True, seen=earlier, short_name="Space Exploration"),
        _stored("WASPIN", pinned=True, seen=earlier),
    ]
    stored = {row["symbol"]: row for row in stored_rows}
    found = rules.candidates(
        [
            _answered(
                "most_actives",
                _item("SEEN", rank=1, short_name="New Name", first_trade_date=None),
                _item("FRESH", rank=2, first_trade_date=dt.date(2021, 3, 1)),
            )
        ],
        core=CORE,
        settings=SETTINGS,
    )
    selection = rules.choose_active(
        found, stored, settings=SETTINGS, complete=True, counted=COUNTED, core=CORE
    )
    rows = {
        row["symbol"]: row
        for row in rules.ticker_rows(selection, found, {}, stored, now=NOW)
    }

    assert set(rows) == {"SEEN", "FRESH", "GONE", "SPCX", "WASPIN"}
    assert all(list(row) == list(rules.TICKER_COLUMNS) for row in rows.values())
    fresh = rows["FRESH"]
    assert fresh["first_seen"] == NOW and fresh["last_seen"] == NOW
    assert fresh["active"] and not fresh["pinned"] and fresh["first_bar_date"] is None
    assert fresh["first_trade_date"] == dt.date(2021, 3, 1)
    seen = rows["SEEN"]
    assert seen["first_seen"] == NOW - pd.Timedelta(days=60)
    assert seen["last_seen"] == NOW and seen["short_name"] == "New Name"
    assert seen["rsi_14"] == 40.0 and seen["first_bar_date"] == dt.date(2026, 5, 11)
    # Active before, absent with grace: kept, but not seen.
    assert rows["GONE"]["active"] and rows["GONE"]["last_seen"] == earlier
    spcx = rows["SPCX"]
    assert spcx["pinned"] and spcx["active"] and spcx["last_seen"] == NOW
    assert spcx["short_name"] == "Space Exploration"  # no item today: kept
    # Pinned no more: unpinned, and judged like any discovered symbol.
    assert rows["WASPIN"]["pinned"] is False


def test_a_pinned_symbols_item_names_its_row_even_through_no_rule():
    selection = rules.choose_active(
        [], {}, settings=SETTINGS, complete=True, counted=COUNTED, core=CORE
    )
    item = _item("SPCX", quote_type="ETF", short_name="Space Exploration Technologies")
    (row,) = rules.ticker_rows(selection, [], {"SPCX": item}, {}, now=NOW)
    assert row["quote_type"] == "ETF"
    assert row["short_name"] == "Space Exploration Technologies"


def test_first_trade_date_is_the_later_of_yahoos_and_the_listing_date():
    selection = rules.choose_active(
        [],
        {},
        core=CORE,
        settings=_settings(pinned=("SPCX", "OLDCO", "NEWCO")),
        complete=True,
        counted=COUNTED,
    )
    items = {
        "SPCX": _item("SPCX", first_trade_date=dt.date(2010, 1, 4)),  # stale metadata
        "OLDCO": _item("OLDCO", first_trade_date=dt.date(2001, 5, 1)),
    }
    listing = {"SPCX": dt.date(2026, 6, 12), "OLDCO": dt.date(1999, 1, 4),
               "NEWCO": dt.date(2025, 3, 3)}  # fmt: skip
    rows = {
        row["symbol"]: row
        for row in rules.ticker_rows(selection, [], items, {}, now=NOW, listing=listing)
    }
    assert rows["SPCX"]["first_trade_date"] == dt.date(2026, 6, 12)
    assert rows["OLDCO"]["first_trade_date"] == dt.date(2001, 5, 1)  # Yahoo's later
    assert rows["NEWCO"]["first_trade_date"] == dt.date(2025, 3, 3)  # no item at all


def test_the_rules_and_the_database_module_name_the_same_tickers_columns():
    assert rules.TICKER_COLUMNS == db_trending.TICKER_COLUMNS


def test_snapshot_and_outcome_rows():
    fetches = [
        _answered("most_actives", _item("A1", rank=1), _item("A2", rank=2)),
        _failed("day_gainers"),
    ]
    session = dt.date(2026, 11, 10)
    snapshots = rules.snapshot_rows(fetches, snapshot_ts=NOW, session=session)
    assert [(r["symbol"], r["rank"], r["list_name"]) for r in snapshots] == [
        ("A1", 1, "most_actives"),
        ("A2", 2, "most_actives"),
    ]
    assert set(snapshots[0]) == set(db_trending.SNAPSHOT_COLUMNS)
    outcomes = rules.fetch_rows(fetches, snapshot_ts=NOW, session=session)
    assert [
        (r["list_name"], r["succeeded"], r["items"], r["error"]) for r in outcomes
    ] == [
        ("most_actives", True, 2, None),
        ("day_gainers", False, 0, "HTTPError: 503"),
    ]
    assert set(outcomes[0]) == set(db_trending.FETCH_COLUMNS)


# ── Reused symbols (section 5) ────────────────────────────────────────────────


def test_listing_dates_go_through_the_shared_parser(tmp_path):
    path = tmp_path / "listing.txt"
    path.write_text(
        "# reused symbols\n"
        "SPCX:2026-06-12  # was a fund's\n"
        "ABC:2020-01-02  # first reuse\n"
        "ABC:2024-05-06  # second reuse\n",
        encoding="utf-8",
    )
    assert rules.listing_dates(path) == {
        "SPCX": dt.date(2026, 6, 12),
        "ABC": dt.date(2024, 5, 6),
    }


def test_a_listing_date_without_a_reason_stops_the_run(tmp_path):
    path = tmp_path / "listing.txt"
    path.write_text("SPCX:2026-06-12\n", encoding="utf-8")
    with pytest.raises(ValueError, match="give the reason"):
        rules.listing_dates(path)


def test_a_listing_date_on_a_closed_day_rolls_to_the_next_session(tmp_path):
    """The download's reach-back check compares its first bar with the
    listing date. A date with no session (a Saturday, Labor Day) has no
    bar, so the check failed on every run and the symbol's bars were never
    replaced; the next session is the first one the listing can have."""
    path = tmp_path / "listing.txt"
    path.write_text(
        "ABC:2026-06-13  # a Saturday\n"
        "XYZ:2026-09-07  # Labor Day\n"
        "SPCX:2026-06-12  # a Friday, kept\n",
        encoding="utf-8",
    )
    assert rules.listing_dates(path) == {
        "ABC": dt.date(2026, 6, 15),
        "XYZ": dt.date(2026, 9, 8),
        "SPCX": dt.date(2026, 6, 12),
    }


def test_the_committed_listing_dates_are_read_by_default():
    assert rules.listing_dates() == {"SPCX": dt.date(2026, 6, 12)}


def test_bars_before_the_listing_date_are_dropped_and_counted():
    days = pd.DatetimeIndex(
        pd.bdate_range("2026-06-08", "2026-06-17"), tz="UTC", name="ts"
    )
    bars = pd.DataFrame({"close": range(len(days))}, index=days)
    kept, dropped = rules.cut_before_listing(bars, dt.date(2026, 6, 12))
    assert dropped == 4
    assert kept.index[0] == pd.Timestamp("2026-06-12", tz="UTC")
    assert rules.cut_before_listing(bars, None) == (bars, 0)


# ── Promotion eligibility (section 8) ─────────────────────────────────────────


def test_history_counts_nyse_trading_days_both_ends_included():
    assert rules.history_days(dt.date(2026, 11, 2), dt.date(2026, 11, 6)) == 5
    # Thanksgiving week: the 26th is a holiday.
    assert rules.history_days(dt.date(2026, 11, 23), dt.date(2026, 11, 27)) == 4
    assert rules.history_days(dt.date(2026, 11, 7), dt.date(2026, 11, 6)) == 0


def test_history_days_agrees_with_the_calendar_everywhere():
    """Counted by numpy's business-day count on the same NYSE calendar: the
    same answer as listing the days, before 1970 and across holidays."""
    pairs = [
        (dt.date(1962, 1, 2), dt.date(1962, 12, 31)),
        (dt.date(1980, 12, 12), dt.date(2026, 11, 10)),
        (dt.date(2026, 11, 26), dt.date(2026, 11, 26)),  # Thanksgiving alone
        (dt.date(2026, 11, 7), dt.date(2026, 11, 9)),  # from a Saturday
        (dt.date(2026, 12, 24), dt.date(2027, 1, 4)),
    ]
    for start, as_of in pairs:
        expected = len(pd.date_range(start, as_of, freq=rules.TRADING_DAY))
        assert rules.history_days(start, as_of) == expected, (start, as_of)


def test_history_days_counts_decades_quickly():
    """Called for every row the tab shows, on every rerun: listing a
    46-year history's days took 0.16 to 0.36 seconds a call."""
    started = time.perf_counter()
    for _ in range(100):
        rules.history_days(dt.date(1980, 12, 12), dt.date(2026, 11, 10))
    assert time.perf_counter() - started < 1.0


def test_the_recent_window_is_the_last_ten_trading_dates():
    assert rules.RECENT_DATES == 10
    assert rules.recent_start(dt.date(2026, 11, 10)) == dt.date(2026, 10, 28)
    # Thanksgiving week: the 26th is no trading date.
    assert rules.recent_start(dt.date(2026, 11, 30)) == dt.date(2026, 11, 16)


def test_the_rules_never_import_the_sentiment_stage():
    """features.sentiment turns every warning off as it is imported, for the
    whole process. The dashboard imports the rules, so it would silence the
    dashboard's warnings too. Checked in a fresh interpreter: this one has
    imported everything already."""
    probe = (
        "import sys, quantflow.ingestion.trending_rules, "
        "quantflow.dashboard.trending_view; "
        "print('quantflow.features.sentiment' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def test_history_starts_at_the_later_of_first_trade_and_listing():
    assert rules.history_start(dt.date(2010, 1, 4), dt.date(2026, 6, 12)) == dt.date(
        2026, 6, 12
    )
    assert rules.history_start(dt.date(2026, 6, 12), None) == dt.date(2026, 6, 12)
    assert rules.history_start(None, None) is None


def _eligible(**kwargs):
    values = {
        "days_trending": 0,
        "pinned": False,
        "first_trade_date": None,
        "as_of": dt.date(2026, 11, 10),
        "settings": SETTINGS,
    } | kwargs
    return rules.promotion_eligible(**values)


def test_spcx_is_not_eligible_until_it_has_two_years_of_history():
    listed = dt.date(2026, 6, 12)
    spcx = {"pinned": True, "first_trade_date": listed}
    assert not _eligible(**spcx)
    first = next(
        day.date()
        for day in pd.date_range("2028-01-03", "2028-12-29", freq="B")
        if rules.history_days(listed, day.date()) >= 504
    )
    before = first - dt.timedelta(days=1)
    assert not _eligible(**spcx, as_of=before)
    assert _eligible(**spcx, as_of=first)
    assert dt.date(2028, 6, 1) <= first <= dt.date(2028, 6, 30)


def test_a_long_listed_name_needs_its_sessions_on_the_lists():
    old = {"first_trade_date": dt.date(1999, 1, 22)}
    assert not _eligible(**old, days_trending=19)
    assert _eligible(**old, days_trending=20)


def test_unknown_history_is_not_enough_history():
    assert not _eligible(pinned=True, days_trending=500)
