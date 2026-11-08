"""tests/unit/test_booster_bands.py

The boosters' bands (Phase 3, I1): the fixed plus or minus 2% band they
publish, and the two candidates the band rule judges. The live forecast and
the evaluation's band variants build each band with the same function, so
the band the evaluation scores is the band the dashboard would show.

No model is trained.
"""

import numpy as np
import pandas as pd
import pytest

from quantflow.evaluation.rolling_origin import OriginData
from quantflow.features.engineering import get_feature_cols
from quantflow.models import boosting, forecasters
from quantflow.utils.trading_calendar import TRADING_DAY, next_trading_days

DATES = pd.date_range("2025-01-02", periods=120, freq=TRADING_DAY)


class Flat:
    """Predicts the same close for every row."""

    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def _closes(n=80, step=0.01):
    """Log returns alternating +step and -step: their standard deviation is
    known exactly."""
    returns = np.where(np.arange(n - 1) % 2 == 0, step, -step)
    return list(100 * np.exp(np.concatenate([[0.0], np.cumsum(returns)])))


def test_the_random_walk_band_widens_with_the_square_root_of_the_horizon():
    closes = _closes()
    sigma = float(np.std(np.diff(np.log(closes[-61:])), ddof=1))
    lower, upper = boosting.random_walk_band(np.full(7, 100.0), closes)
    half = 1.96 * sigma * np.sqrt(np.arange(1, 8))
    np.testing.assert_allclose(lower, 100 * np.exp(-half), rtol=1e-12)
    np.testing.assert_allclose(upper, 100 * np.exp(half), rtol=1e-12)


def test_the_random_walk_band_reads_only_the_last_60_returns():
    closes = _closes()
    calmer = closes[:-61]  # anything before the last 61 closes does not count
    noisy = [c * (1 + 0.3 * (i % 2)) for i, c in enumerate(calmer)] + closes[-61:]
    a = boosting.random_walk_band(np.full(7, 100.0), closes)
    b = boosting.random_walk_band(np.full(7, 100.0), noisy)
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_the_quantile_band_takes_each_horizons_own_errors():
    errors = {
        h: np.log(np.linspace(1 - 0.01 * h, 1 + 0.02 * h, 12)) for h in range(1, 8)
    }
    lower, upper = boosting.quantile_band(np.full(7, 50.0), errors)
    for h in range(1, 8):
        low, high = np.quantile(errors[h], [0.025, 0.975])
        assert lower[h - 1] == pytest.approx(50 * np.exp(low))
        assert upper[h - 1] == pytest.approx(50 * np.exp(high))


@pytest.mark.parametrize("band", ["fixed_2pct", "random_walk"])
def test_the_live_forecast_publishes_its_familys_band(monkeypatch, band):
    monkeypatch.setitem(boosting.BANDS, "xgboost", band)
    closes = _closes()
    X = pd.DataFrame([dict.fromkeys(get_feature_cols(), 0.5)])
    out = boosting.generate_forecast(
        Flat(101.0), X, DATES[-1], closes, "xgboost", "AAPL"
    )
    expected = (
        boosting.fixed_band(out["predicted_close"])
        if band == "fixed_2pct"
        else boosting.random_walk_band(out["predicted_close"], closes)
    )
    np.testing.assert_allclose(out["lower_bound"], expected[0], rtol=1e-12)
    np.testing.assert_allclose(out["upper_bound"], expected[1], rtol=1e-12)


def test_a_band_with_no_live_form_is_refused(monkeypatch):
    monkeypatch.setitem(boosting.BANDS, "xgboost", "error_quantile")
    X = pd.DataFrame([dict.fromkeys(get_feature_cols(), 0.5)])
    with pytest.raises(ValueError, match="no live"):
        boosting.generate_forecast(Flat(101.0), X, DATES[-1], _closes(), "xgboost", "A")


def _data(origin, closes):
    prices = pd.DataFrame({"date": DATES[: len(closes)], "close": closes})
    return OriginData(
        "AAPL", origin, prices, pd.DataFrame(), next_trading_days(origin, 7)
    )


def _base(value):
    return pd.DataFrame(
        {
            "yhat": np.full(7, value),
            "yhat_lower": value * 0.98,
            "yhat_upper": value * 1.02,
        }
    )


def test_the_evaluations_random_walk_band_is_the_live_one():
    closes = _closes()
    origin = DATES[len(closes) - 1]
    out = forecasters.BAND_VARIANTS["xgboost_rwband"](
        _data(origin, closes), {"xgboost": _base(101.0)}, pd.DataFrame()
    )
    lower, upper = boosting.random_walk_band(np.full(7, 101.0), closes)
    np.testing.assert_allclose(out["yhat"], 101.0)
    np.testing.assert_allclose(out["yhat_lower"], lower, rtol=1e-12)
    np.testing.assert_allclose(out["yhat_upper"], upper, rtol=1e-12)


def _earlier(origins, miss):
    """xgboost cells at the earlier origins, each miss[h] too high."""
    rows = []
    for o in origins:
        for h in range(1, 8):
            target = DATES[DATES.get_loc(o) + h]
            rows.append(
                {"origin": o, "h": h, "target_date": target, "model": "xgboost",
                 "yhat": 100.0 * (1 + miss[h]), "actual": 100.0, "status": "ok"}
            )  # fmt: skip
    return pd.DataFrame(rows)


def test_the_quantile_band_comes_from_earlier_errors_known_by_the_origin():
    grid = list(DATES[5:100:5])
    origin = grid[15]
    miss = {h: 0.01 * h for h in range(1, 8)}
    later = _earlier(grid[15:17], dict.fromkeys(range(1, 8), 0.5))  # after: ignored
    cells = pd.concat([_earlier(grid[3:15], miss), later], ignore_index=True)
    out = forecasters.BAND_VARIANTS["xgboost_qband"](
        _data(origin, _closes(DATES.get_loc(origin) + 1)),
        {"xgboost": _base(80.0)},
        cells,
    )
    # Every earlier forecast was miss[h] too high: log(actual / yhat) is
    # -log(1 + miss[h]) at every row, so both quantiles are that value.
    for h in range(1, 8):
        assert out["yhat_lower"].iloc[h - 1] == pytest.approx(80.0 / (1 + miss[h]))
        assert out["yhat_upper"].iloc[h - 1] == pytest.approx(80.0 / (1 + miss[h]))


def test_the_band_variants_are_registered_and_ask_for_history():
    assert set(forecasters.BAND_VARIANTS) == {
        "xgboost_rwband",
        "xgboost_qband",
        "lightgbm_rwband",
        "lightgbm_qband",
    }
    for variant in forecasters.BAND_VARIANTS.values():
        assert variant.needs_history
    with pytest.raises(ValueError, match="unknown band"):
        forecasters.BandVariant("xgboost", "widest")
