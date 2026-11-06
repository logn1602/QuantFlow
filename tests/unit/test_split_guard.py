"""tests/unit/test_split_guard.py

Covers quantflow.ingestion.split_guard.

Yahoo serves prices split-adjusted as of the moment they are fetched, and
stored rows are never rewritten, so a split after the rows were stored would
read as a crash. The guard must fail on a split, stay quiet on dividend drift
and on closes built from 15-min bars, and take its threshold from config.
"""

import numpy as np
import pandas as pd
import pytest

from quantflow.ingestion import split_guard, yfinance_fetcher

DAYS = pd.bdate_range("2026-05-04", periods=10)


def _closes(values):
    return pd.DataFrame({"date": DAYS[: len(values)], "close": values})


# ── find_basis_breaks ─────────────────────────────────────────────────────────


def test_rows_stored_before_a_ten_for_one_split_are_flagged():
    stored = _closes([1000.0] * 5 + [100.0] * 5)  # the split took effect on day 5
    yahoo = _closes([100.0] * 10)  # Yahoo now serves everything split-adjusted

    breaks = split_guard.find_basis_breaks(stored, yahoo, threshold=0.15)

    assert list(breaks["date"]) == list(DAYS[:5])
    assert np.allclose(breaks["ratio"], 10.0)


def test_dividend_drift_within_the_threshold_is_not_flagged():
    """Seeded rows were stored dividend-adjusted, a few percent below raw."""
    stored = _closes([96.0, 97.0, 100.0])
    yahoo = _closes([100.0, 100.0, 100.0])
    assert split_guard.find_basis_breaks(stored, yahoo, threshold=0.15).empty


def test_the_threshold_is_a_parameter():
    stored = _closes([96.0, 100.0])
    yahoo = _closes([100.0, 100.0])
    breaks = split_guard.find_basis_breaks(stored, yahoo, threshold=0.03)
    assert list(breaks["date"]) == [DAYS[0]]


def test_dates_only_one_side_has_are_ignored():
    stored = _closes([100.0, 100.0])
    yahoo = _closes([100.0])
    assert split_guard.find_basis_breaks(stored, yahoo, threshold=0.15).empty


# ── run / main ────────────────────────────────────────────────────────────────


@pytest.fixture
def split_stored(read_tables, price_bars):
    """AAPL stored at the pre-split scale on every date."""
    read_tables["raw_prices"] = price_bars.frame(
        [price_bars.daily(day.date().isoformat(), 400.0) for day in DAYS]
    )


def _yahoo_serving(monkeypatch, close):
    frame = pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Adj Close": close,
            "Volume": 1e6,
        },
        index=DAYS,
    )
    monkeypatch.setattr(
        yfinance_fetcher.yf, "download", lambda ticker, **kwargs: frame.copy()
    )


def test_a_split_fails_the_run_naming_the_ticker(split_stored, monkeypatch):
    _yahoo_serving(monkeypatch, 100.0)  # a 4-for-1 split since the rows were stored
    with pytest.raises(split_guard.PriceBasisError, match="AAPL"):
        split_guard.run(["AAPL"], threshold=0.15)


def test_the_cli_exits_non_zero_on_a_split(split_stored, monkeypatch, capsys):
    _yahoo_serving(monkeypatch, 100.0)
    monkeypatch.setattr(split_guard, "TICKERS", ["AAPL"])
    assert split_guard.main([]) == 1
    assert "AAPL" in capsys.readouterr().err


def test_matching_prices_pass(split_stored, monkeypatch):
    _yahoo_serving(monkeypatch, 400.0)
    split_guard.run(["AAPL"], threshold=0.15)  # must not raise


def test_the_default_threshold_comes_from_config(split_stored, monkeypatch):
    _yahoo_serving(monkeypatch, 330.0)  # stored is about 21% above Yahoo
    monkeypatch.setattr(split_guard, "SPLIT_GUARD_THRESHOLD", 0.25)
    split_guard.run(["AAPL"])  # within 25%: passes

    monkeypatch.setattr(split_guard, "SPLIT_GUARD_THRESHOLD", 0.15)
    with pytest.raises(split_guard.PriceBasisError):
        split_guard.run(["AAPL"])


@pytest.mark.parametrize("bad", [0.0, 1.0, 15.0, -0.1])
def test_a_threshold_outside_zero_to_one_is_rejected(bad):
    """15 meaning "15%" would silently disable the guard."""
    with pytest.raises(ValueError, match="SPLIT_GUARD_THRESHOLD"):
        split_guard.run(["AAPL"], threshold=bad)


def test_no_reference_from_yahoo_is_a_warning_not_a_failure(split_stored, monkeypatch):
    monkeypatch.setattr(
        yfinance_fetcher.yf, "download", lambda ticker, **kwargs: pd.DataFrame()
    )
    split_guard.run(["AAPL"], threshold=0.15)  # must not raise
