"""tests/unit/test_daily_bars.py

Covers the one-row-per-trading-date rule in quantflow.db.prices.

Seeding stores one daily bar per date, but every refresh after it stored only
15-min bars. The loaders collapsed a date to its first row by timestamp, so on
every date after the seed the "daily close" the models trained and were
scored on was the close of the 09:30 bar. These tests pin the rule that
replaced it: a date with a daily bar uses it, and a date with only 15-min
bars gets a day bar built from them.
"""

import logging

import pandas as pd
import pytest

from quantflow.db import prices
from quantflow.features import engineering

# Two trading dates: one seeded with a daily bar, one with only a full
# 15-min session whose closes climb from 100.0 to 102.5.
SEEDED_DAY = "2026-04-09"
INTRADAY_DAY = "2026-04-13"
SESSION_CLOSES = [100.0 + 0.1 * i for i in range(26)]


@pytest.fixture
def two_days(price_bars):
    return price_bars.frame(
        price_bars.daily(SEEDED_DAY, 98.0, high=99.0, low=97.0, volume=5e6),
        price_bars.session(INTRADAY_DAY, SESSION_CLOSES, volume=1000.0),
    )


def _row(frame, column, day):
    match = frame[pd.to_datetime(frame[column]) == pd.Timestamp(day)]
    assert len(match) == 1, f"expected one row for {day}, got {len(match)}"
    return match.iloc[0]


# ── Model-facing loaders ──────────────────────────────────────────────────────


def test_daily_close_uses_the_last_15min_bar_on_a_date_without_a_daily_bar(
    read_tables, two_days
):
    read_tables["raw_prices"] = two_days
    daily = prices.load_daily_close("AAPL")

    assert _row(daily, "ds", INTRADAY_DAY)["y"] == pytest.approx(102.5)


def test_daily_close_prefers_the_daily_bar_when_a_date_has_one(read_tables, price_bars):
    read_tables["raw_prices"] = price_bars.frame(
        price_bars.daily(SEEDED_DAY, 98.0),
        price_bars.session(SEEDED_DAY, [97.0, 97.5, 97.9]),
    )
    daily = prices.load_daily_close("AAPL")

    assert len(daily) == 1
    assert daily["y"].iloc[0] == pytest.approx(98.0)


def test_feature_prices_get_a_full_day_bar_built_from_15min_bars(read_tables, two_days):
    read_tables["raw_prices"] = two_days
    for table, columns in {
        "technical_indicators": ["date", "rsi_14", "macd", "macd_signal",
                                 "macd_hist", "bb_upper", "bb_middle", "bb_lower"],
        "anomalies": ["date", "zscore"],
        "news_sentiment": ["date", "sentiment_compound", "pos_count",
                           "neg_count", "article_count"],
    }.items():  # fmt: skip
        read_tables[table] = pd.DataFrame(columns=columns)

    day = _row(engineering.load_features("AAPL"), "date", INTRADAY_DAY)

    assert day["open"] == pytest.approx(SESSION_CLOSES[0] - 0.05)
    assert day["high"] == pytest.approx(max(SESSION_CLOSES) + 0.5)
    assert day["low"] == pytest.approx(min(SESSION_CLOSES) - 0.5)
    assert day["close"] == pytest.approx(SESSION_CLOSES[-1])
    assert day["volume"] == pytest.approx(26 * 1000.0)


# ── Dashboard loader ──────────────────────────────────────────────────────────


def test_dashboard_prices_have_one_row_per_trading_date(read_tables, two_days):
    read_tables["raw_prices"] = two_days
    recent = prices.load_recent_ohlcv("AAPL", days=30)

    assert len(recent) == 2
    assert _row(recent, "date", INTRADAY_DAY)["close"] == pytest.approx(102.5)


# ── The warning for dates built from 15-min bars ──────────────────────────────


def test_dates_built_from_15min_bars_are_counted_in_a_warning(
    read_tables, two_days, caplog
):
    read_tables["raw_prices"] = two_days
    with caplog.at_level(logging.WARNING, logger="quantflow.db.prices"):
        prices.load_daily_close("AAPL")

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("AAPL: 1 of 2 trading dates" in m for m in warnings), warnings


def test_no_warning_when_every_date_has_a_daily_bar(read_tables, price_bars, caplog):
    read_tables["raw_prices"] = price_bars.frame(
        price_bars.daily(SEEDED_DAY, 98.0), price_bars.daily("2026-04-10", 99.0)
    )
    with caplog.at_level(logging.WARNING, logger="quantflow.db.prices"):
        prices.load_daily_close("AAPL")
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


# ── Bars that are neither daily nor on the session grid ──────────────────────


def test_an_off_grid_bar_does_not_replace_the_day(price_bars, caplog):
    """Anything off the 15-min grid used to count as a daily bar, so one stray
    row (a 16:00 print, a bar with seconds) became the whole day: its own
    open, high, low and close, and its few shares of volume."""
    session = price_bars.session(INTRADAY_DAY, SESSION_CLOSES, volume=1000.0)
    stray = {
        **session[-1],
        "ts": pd.Timestamp(f"{INTRADAY_DAY} 16:00", tz="America/New_York").tz_convert(
            "UTC"
        ),
        "close": 999.0,
        "volume": 5.0,
    }
    with caplog.at_level(logging.WARNING, logger="quantflow.db.prices"):
        out = prices.daily_bars(price_bars.frame(session, stray), "AAPL")

    day = _row(out, "date", INTRADAY_DAY)
    assert day["close"] == pytest.approx(SESSION_CLOSES[-1])
    assert day["volume"] == pytest.approx(26 * 1000.0)
    assert any("off the session grid" in r.getMessage() for r in caplog.records)


# ── Trailing windows start at a UTC midnight ──────────────────────────────────


def test_a_trailing_window_starts_at_a_utc_midnight(fake_db, monkeypatch):
    """Cut at NOW() minus N days, the first date lost its 00:00 UTC daily bar
    but kept its afternoon 15-min bars, so the dashboard's first candle was
    built from half a session."""
    seen = {}

    def capture(sql, params, ticker):
        seen.update(params)
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    monkeypatch.setattr(prices, "_read", capture)
    monkeypatch.setattr(
        prices, "_now", lambda: pd.Timestamp("2026-11-06 21:37:12", tz="UTC")
    )
    prices.load_recent_ohlcv("AAPL", days=30)

    assert seen["since"] == pd.Timestamp("2026-10-07 00:00", tz="UTC")


# ── The pure rule ─────────────────────────────────────────────────────────────


def test_a_daily_bar_stored_at_new_york_midnight_is_still_a_daily_bar(price_bars):
    """Older yfinance versions returned exchange-local midnights."""
    bar = price_bars.daily(SEEDED_DAY, 98.0)
    bar["ts"] = pd.Timestamp(SEEDED_DAY, tz="America/New_York").tz_convert("UTC")
    out = prices.daily_bars(price_bars.frame(bar), "AAPL")

    assert list(out["date"]) == [pd.Timestamp(SEEDED_DAY)]
    assert out["close"].iloc[0] == pytest.approx(98.0)


def test_winter_and_summer_sessions_are_both_recognised(price_bars):
    """09:30 New York is 13:30 UTC in summer and 14:30 UTC in winter."""
    frame = price_bars.frame(
        price_bars.session("2026-07-14", [10.0, 11.0]),
        price_bars.session("2026-11-03", [20.0, 21.0]),
    )
    out = prices.daily_bars(frame, "AAPL")

    assert list(out["close"]) == [11.0, 21.0]


def test_the_day_bar_volume_is_missing_only_when_every_bar_lacks_it(price_bars):
    bars = price_bars.session(INTRADAY_DAY, [1.0, 2.0])
    for bar in bars:
        bar["volume"] = float("nan")
    out = prices.daily_bars(price_bars.frame(bars), "AAPL")

    assert pd.isna(out["volume"].iloc[0])


def test_no_bars_give_an_empty_frame_with_the_daily_columns():
    empty = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    out = prices.daily_bars(empty, "AAPL")

    assert out.empty
    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume"]
