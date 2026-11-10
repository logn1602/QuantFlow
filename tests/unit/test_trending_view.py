"""tests/unit/test_trending_view.py

The Trending tab's reads and shaping (dashboard/trending_view.py,
docs/trending.md section 10), offline: the reads with a fake connection and
canned frames, the empty state for tables not there yet, each list's last
success, the table's badges and the chart's frames. The queries themselves
run on Postgres in tests/integration/test_trending_on_postgres.py.
"""

import dataclasses
import datetime as dt
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.exc import ProgrammingError

from quantflow.dashboard import trending_view as view
from quantflow.ingestion import trending_rules as rules

pytestmark = pytest.mark.trending

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
TODAY = dt.date(2026, 11, 10)
QUERIES = (view.TABLE_QUERY, view.LISTS_QUERY, view.BARS_QUERY)


class FakeConnection:
    def __init__(self):
        self.rollbacks = 0

    def rollback(self):
        self.rollbacks += 1


def _missing_table():
    return ProgrammingError(
        "SELECT", {}, SimpleNamespace(pgcode="42P01", args=("no such table",))
    )


# ── Reads ─────────────────────────────────────────────────────────────────────


def test_each_read_goes_to_its_query_with_its_parameters(monkeypatch):
    seen = []

    def read_sql(query, conn, params=None):
        seen.append((query, params))
        return pd.DataFrame({"x": [1]})

    monkeypatch.setattr(view.pd, "read_sql", read_sql)
    conn = FakeConnection()
    view.load_table(conn)
    view.load_list_status(conn)
    view.load_bars(conn, "SPCX")
    assert seen == [
        (view.TABLE_QUERY, {}),
        (view.LISTS_QUERY, {}),
        (view.BARS_QUERY, {"symbol": "SPCX"}),
    ]


def test_a_table_not_there_yet_reads_as_empty_and_rolls_back(monkeypatch):
    """Before the owner runs 002_trending.sql: the tab's empty state, not an
    error, and the failed statement's transaction is ended."""

    def read_sql(query, conn, params=None):
        raise _missing_table()

    monkeypatch.setattr(view.pd, "read_sql", read_sql)
    conn = FakeConnection()
    assert view.load_table(conn).empty
    assert view.load_list_status(conn).empty
    assert view.load_bars(conn, "SPCX").empty
    assert conn.rollbacks == 3


def test_any_other_database_error_is_raised_not_shown_as_no_data(monkeypatch):
    def read_sql(query, conn, params=None):
        raise ProgrammingError("SELECT", {}, SimpleNamespace(pgcode="42501"))

    monkeypatch.setattr(view.pd, "read_sql", read_sql)
    with pytest.raises(ProgrammingError):
        view.load_table(FakeConnection())


def test_the_table_is_one_query_over_the_tiers_own_tables():
    sql = view.TABLE_QUERY.text
    assert ":" not in re.sub(r"::\w+", "", sql)  # no per-row parameter
    for query in QUERIES:
        tables = set(re.findall(r"\b(?:FROM|JOIN)\s+(\w+)", query.text, re.I))
        tables -= {"latest", "listed", "days", "news"}  # the CTEs
        assert tables <= {
            "tickers",
            "trending_snapshots",
            "trending_fetches",
            "trending_prices",
            "trending_news",
        }, tables


def test_no_query_casts_a_timestamp_to_a_date():
    """Decision K: a bare ts::date follows the session's TimeZone."""
    for query in QUERIES:
        assert "::date" not in query.text


# ── The per-list line (T8) ────────────────────────────────────────────────────


def test_each_list_shows_its_last_success_in_trending_lists_order():
    t = pd.Timestamp("2026-11-10 23:10", tz="UTC")
    status = pd.DataFrame(
        {
            "list_name": ["most_actives", "trending_tickers", "day_gainers"],
            "last_success": [t, t - pd.Timedelta(days=1), pd.NaT],
            "last_fetch": [t, t, t],
            "latest_succeeded": [True, False, False],
        }
    )
    assert view.list_lines(status, rules.KNOWN_LISTS) == [
        "trending_tickers: last succeeded 2026-11-09 23:10 UTC; failed since",
        "most_actives: last succeeded 2026-11-10 23:10 UTC",
        "day_gainers: never succeeded (last tried 2026-11-10 23:10 UTC)",
        "day_losers: never fetched",
    ]


def test_no_status_at_all_reads_never_fetched_for_every_list():
    assert view.list_lines(pd.DataFrame(), ("most_actives",)) == [
        "most_actives: never fetched"
    ]


# ── The table ─────────────────────────────────────────────────────────────────


def _raw(**changes):
    row = {
        "symbol": "HIMS",
        "short_name": "Hims & Hers Health, Inc.",
        "pinned": False,
        "first_trade_date": dt.date(2019, 1, 23),
        "summary_session": TODAY,
        "last_close": 18.81512,
        "rsi_14": 28.6123,
        "recent_flags": 4,
        "big_move_date": None,
        "best_rank": 2,
        "lists": ["day_losers", "trending_tickers", "most_actives"],
        "pct_change": -17.981691,
        "volume": 48734986,
        "days_trending": 3,
        "sentiment_7d": -0.41234,
        "articles_7d": 17,
    }
    return row | changes


def test_a_row_reads_as_text_ready_values():
    shown = view.table_rows(pd.DataFrame([_raw()]), settings=SETTINGS, today=TODAY)
    assert list(shown.columns) == view.TABLE_COLUMNS
    row = shown.iloc[0]
    assert row["Symbol"] == "HIMS" and row["Name"] == "Hims & Hers Health, Inc."
    assert row["Lists"] == "trending_tickers, most_actives, day_losers"
    assert row["% change"] == -17.98 and row["Volume"] == 48734986
    assert row["Sentiment (7d)"] == -0.412 and row["Articles (7d)"] == 17
    assert row["RSI (14)"] == 28.6 and row["Last close"] == 18.82
    assert row["Flags (10d)"] == 4 and row["Session"] == TODAY
    assert row["Badges"] == ""


def test_a_pinned_symbol_off_the_lists_has_no_rank_and_the_pinned_badge():
    raw = _raw(
        symbol="SPCX",
        pinned=True,
        best_rank=np.nan,
        lists=None,
        pct_change=np.nan,
        volume=np.nan,
        days_trending=0,
        sentiment_7d=np.nan,
        articles_7d=0,
        recent_flags=np.nan,
        rsi_14=np.nan,
        first_trade_date=dt.date(2026, 6, 12),
    )
    row = view.table_rows(pd.DataFrame([raw]), settings=SETTINGS, today=TODAY).iloc[0]
    assert pd.isna(row["Best rank"]) and row["Lists"] == ""
    assert pd.isna(row["% change"]) and pd.isna(row["Flags (10d)"])
    assert row["Badges"] == "pinned"  # not eligible: under two years of history


def test_promotion_eligible_needs_sessions_and_two_years_of_history():
    old = {"first_trade_date": dt.date(1999, 1, 22)}
    few = view.badges(
        pd.Series(_raw(**old, days_trending=19)), settings=SETTINGS, today=TODAY
    )
    enough = view.badges(
        pd.Series(_raw(**old, days_trending=20)), settings=SETTINGS, today=TODAY
    )
    young = view.badges(
        pd.Series(_raw(first_trade_date=dt.date(2025, 6, 2), days_trending=60)),
        settings=SETTINGS,
        today=TODAY,
    )
    unknown = view.badges(
        pd.Series(_raw(first_trade_date=None, days_trending=60)),
        settings=SETTINGS,
        today=TODAY,
    )
    assert (few, enough, young, unknown) == ("", "promotion-eligible", "", "")


def test_the_big_move_badge_covers_the_last_ten_trading_dates():
    def badge(day):
        return view.badges(
            pd.Series(_raw(big_move_date=day)), settings=SETTINGS, today=TODAY
        )

    # Ten trading dates ending 2026-11-10 start on 2026-10-28.
    assert badge(dt.date(2026, 10, 28)) == "big move 2026-10-28"
    assert badge(dt.date(2026, 10, 27)) == ""
    assert badge(None) == ""


def test_the_badges_use_the_thresholds_set():
    lenient = dataclasses.replace(
        SETTINGS, promotion_min_days=1, promotion_min_history_days=252
    )
    raw = _raw(first_trade_date=dt.date(2025, 6, 2), days_trending=1)
    assert view.badges(pd.Series(raw), settings=lenient, today=TODAY) == (
        "promotion-eligible"
    )


def test_no_rows_is_an_empty_table_with_its_columns():
    shown = view.table_rows(pd.DataFrame(), settings=SETTINGS, today=TODAY)
    assert shown.empty and list(shown.columns) == view.TABLE_COLUMNS


# ── The chart ─────────────────────────────────────────────────────────────────


def _bars(n):
    days = pd.bdate_range(end="2026-11-10", periods=n)
    closes = 20 + np.sin(np.arange(n)) + 0.05 * np.arange(n)
    closes[-5] *= 1.5
    return pd.DataFrame(
        {
            "ts": pd.DatetimeIndex(days, tz="UTC"),
            "open": closes,
            "high": closes + 1,
            "low": closes - 1,
            "close": closes,
            "volume": 1_000_000,
        }
    )


def test_the_chart_frames_take_the_core_charts_shapes():
    prices, indicators, flags = view.chart_frames(_bars(120), "HIMS")
    assert list(prices.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert len(prices) == 120
    for column in ("date", "rsi_14", "macd", "macd_signal", "macd_hist", "bb_upper"):
        assert column in indicators.columns
    assert indicators["date"].dt.tz is None  # dated like the candles
    assert set(indicators["date"]) <= set(prices["date"])
    assert {"date", "close", "flag"} <= set(flags.columns)
    assert pd.Timestamp("2026-11-04") in set(flags["date"])  # the 50% jump


def test_a_short_history_has_candles_and_no_indicators():
    """Under 26 rows there are no indicators. The IQR detector has no
    warm-up, unlike the 30-row z-score, so the jump is still flagged: the
    core's rule, unchanged."""
    prices, indicators, flags = view.chart_frames(_bars(12), "NEWCO")
    assert len(prices) == 12 and indicators.empty
    # The jump and the fall back the next day.
    assert list(flags["date"]) == [
        pd.Timestamp("2026-11-04"),
        pd.Timestamp("2026-11-05"),
    ]


def test_no_bars_is_three_empty_frames():
    prices, indicators, flags = view.chart_frames(pd.DataFrame(), "NONE")
    assert prices.empty and indicators.empty and flags.empty
    assert "date" in prices.columns
