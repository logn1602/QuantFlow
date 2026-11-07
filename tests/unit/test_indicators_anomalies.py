"""tests/unit/test_indicators_anomalies.py

Covers the series that technical indicators and anomaly flags are computed on.

Both stages used to read every raw_prices row, so after the seed date RSI,
MACD, Bollinger Bands and the 30-row Z-score were computed over a mix of daily
bars and 15-min bars: a "30-day" window covered about one trading day. The
feature matrix then kept whichever indicator row came first on each date.
Both now run on the one-row-per-trading-date series from db.prices.
"""

import numpy as np
import pandas as pd
import pytest
import ta
from sqlalchemy.exc import DataError

from quantflow.features import anomalies, indicators

SEEDED_DAYS = pd.bdate_range("2026-02-02", periods=40)
SESSION_DAYS = ["2026-03-30", "2026-03-31", "2026-04-01"]


@pytest.fixture
def mixed_bars(price_bars):
    """40 seeded daily bars, then three dates with only 15-min sessions.

    The middle session jumps 8% at 11:45, which a bar-level detector flags
    on the jump bar itself.
    """
    rng = np.random.default_rng(11)
    closes = 100 + np.cumsum(rng.normal(0, 1.0, len(SEEDED_DAYS)))
    groups = [
        price_bars.daily(day.date().isoformat(), float(c))
        for day, c in zip(SEEDED_DAYS, closes, strict=True)
    ]
    level = float(closes[-1])
    for i, day in enumerate(SESSION_DAYS):
        session = [level + 0.05 * k for k in range(26)]
        if i == 1:
            session = [c * (1.08 if k >= 9 else 1.0) for k, c in enumerate(session)]
        groups.append(price_bars.session(day, session))
        level = session[-1]
    return price_bars.frame(*groups)


def _expected_daily_closes(bars):
    from quantflow.db.prices import daily_bars

    return daily_bars(bars, "AAPL").set_index("date")["close"]


# ── Indicators ────────────────────────────────────────────────────────────────


def test_indicators_are_computed_on_one_row_per_trading_date(
    read_tables, mixed_bars, monkeypatch
):
    read_tables["raw_prices"] = mixed_bars
    saved = {}
    monkeypatch.setattr(
        indicators, "save_indicators", lambda df, ticker: saved.setdefault("df", df)
    )

    indicators.run(["AAPL"])
    out = saved["df"]

    assert out.index.is_unique
    assert (out.index == out.index.normalize()).all(), "an intraday timestamp"
    closes = _expected_daily_closes(mixed_bars)
    expected_rsi = ta.momentum.RSIIndicator(close=closes, window=14).rsi()
    assert np.allclose(
        out["rsi_14"].to_numpy(),
        expected_rsi.reindex(out.index.tz_localize(None)).to_numpy(),
    )


def test_saving_indicators_replaces_the_tickers_rows(fake_db):
    """Replacing rather than upserting is what removes the stale rows the
    old bar-level runs left at 15-min timestamps."""
    stale = pd.Timestamp("2026-03-30 14:30", tz="UTC")
    fake_db.seed(
        "technical_indicators",
        [{"ticker": "AAPL", "ts": stale}, {"ticker": "MSFT", "ts": stale}],
    )
    frame = _indicator_frame(["2026-03-30", "2026-03-31"])

    written = indicators.save_indicators(frame, "AAPL")

    rows = fake_db.tables["technical_indicators"]
    assert written == 2
    assert sorted(r["ticker"] for r in rows) == ["AAPL", "AAPL", "MSFT"]
    assert stale not in [r["ts"] for r in rows if r["ticker"] == "AAPL"]
    assert len(fake_db.inserts("technical_indicators")) == 1


def test_a_concurrent_replace_skips_rows_instead_of_failing(fake_db):
    """Two runs replacing the same ticker at once (the laptop scheduler and
    the daily workflow): the second one's DELETE cannot see rows the first
    committed after it started, so its plain INSERT would hit the unique key
    and fail the whole refresh. Skipping those rows leaves the first run's
    identical computation in place."""
    indicators.save_indicators(_indicator_frame(["2026-03-31"]), "AAPL")
    sql, _ = fake_db.inserts("technical_indicators")[0]
    assert "ON CONFLICT (ticker, ts) DO NOTHING" in sql


def test_a_failed_indicator_write_keeps_the_previous_rows(fake_db):
    stale = pd.Timestamp("2026-03-30", tz="UTC")
    fake_db.seed("technical_indicators", [{"ticker": "AAPL", "ts": stale}])
    fake_db.fail_next_insert = DataError("INSERT", {}, Exception("numeric overflow"))

    with pytest.raises(DataError):
        indicators.save_indicators(_indicator_frame(["2026-03-31"]), "AAPL")

    assert [r["ts"] for r in fake_db.tables["technical_indicators"]] == [stale]


def _indicator_frame(days):
    index = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in days], name="ts")
    values = {
        c: np.linspace(1, 2, len(days))
        for c in ["rsi_14", "macd", "macd_signal", "macd_hist"]
    }
    values |= {"bb_upper": [3.0] * len(days), "bb_middle": [2.0] * len(days)}
    values |= {"bb_lower": [np.nan] * len(days)}
    return pd.DataFrame(values, index=index)


# ── Anomalies ─────────────────────────────────────────────────────────────────


def test_anomalies_are_flagged_on_trading_dates_not_15min_bars(
    read_tables, mixed_bars, monkeypatch
):
    read_tables["raw_prices"] = mixed_bars
    saved = {}
    monkeypatch.setattr(
        anomalies, "save_anomalies", lambda df: saved.setdefault("df", df) is None
    )

    anomalies.run(["AAPL"])
    flagged = saved["df"]

    assert not flagged.empty
    assert (flagged.index == flagged.index.normalize()).all(), "an intraday flag"
    assert pd.Timestamp(SESSION_DAYS[1], tz="UTC") in flagged.index


def test_saving_anomalies_counts_only_rows_written(fake_db):
    """save_anomalies added one to its count per INSERT sent, so a row that ON
    CONFLICT DO NOTHING skipped was still reported as written. A run with no
    conflicts was counted right, which is the live table today: it has no
    unique key on (ticker, ts) yet. A fresh install has one."""
    ts = pd.Timestamp("2026-04-10", tz="UTC")
    flags = pd.DataFrame(
        {"ticker": "AAPL", "close": 100.0, "zscore": 3.2, "flag": "HIGH"},
        index=pd.DatetimeIndex([ts, ts], name="ts"),
    )
    assert anomalies.save_anomalies(flags) == 1
    assert len(fake_db.tables["anomalies"]) == 1
