"""tests/unit/test_rolling_origin_weights.py

Covers the ensemble weights learned on the task they are used for
(docs/evaluation.md, section 5): the rows the weights at an origin may use,
the thin-window and all-zero failures, and the evaluation's combiner. The
live ensemble calls the same rolling_origin_fit.
"""

import numpy as np
import pandas as pd
import pytest

from quantflow.evaluation import out_of_fold as oof
from quantflow.evaluation.rolling_origin import OriginData
from quantflow.models import ensemble, forecasters
from quantflow.utils.trading_calendar import TRADING_DAY, next_trading_days

DATES = pd.date_range("2025-01-02", periods=120, freq=TRADING_DAY)
MODELS = ensemble.ROLLING_COLUMNS
H = 7


def _cells(origins, offsets=None, failed=()):
    """Base cells at each origin: model m forecasts actual + offsets[m].
    failed holds (origin, model) pairs whose cells failed."""
    offsets = offsets or {
        "arima": 0.0, "prophet": 5.0, "xgboost": 3.0, "lightgbm": 9.0, "naive": 7.0
    }  # fmt: skip
    rows = []
    for o in origins:
        for h in range(1, H + 1):
            target = DATES[DATES.get_loc(o) + h]
            actual = 100.0 + DATES.get_loc(target)
            for m in MODELS:
                bad = (o, m) in failed
                rows.append(
                    {"origin": o, "h": h, "target_date": target, "model": m,
                     "yhat": np.nan if bad else actual + offsets[m],
                     "actual": actual, "status": "failed" if bad else "ok"}
                )  # fmt: skip
    return pd.DataFrame(rows)


GRID = list(DATES[5:100:5])  # origins every 5 trading days


def test_no_row_has_a_target_after_the_origin():
    origin = GRID[-1]
    stack = oof.rolling_origin_stack(_cells(GRID[:-1]), origin, MODELS)
    targets = [
        DATES[DATES.get_loc(o) + h]
        for o, h in zip(stack["origin"], stack["h"], strict=True)
    ]
    assert max(targets) <= origin
    last = stack[stack["origin"] == GRID[-2]]
    assert sorted(last["h"]) == [1, 2, 3, 4, 5]  # o - 5: horizons 6 and 7 left out


def test_only_the_twelve_latest_earlier_origins_are_used():
    origin = GRID[15]
    cells = _cells(GRID)  # includes origins at and after `origin`
    stack = oof.rolling_origin_stack(cells, origin, MODELS)
    assert sorted(set(stack["origin"])) == GRID[3:15]
    assert len(stack) == 11 * H + 5


def test_an_origin_with_a_failed_base_forecast_is_left_out():
    origin = GRID[15]
    cells = _cells(GRID[3:15], failed={(GRID[5], "prophet")})
    stack = oof.rolling_origin_stack(cells, origin, MODELS)
    assert GRID[5] not in set(stack["origin"]) and len(set(stack["origin"])) == 11


def test_fewer_than_eight_complete_origins_fail():
    origin = GRID[15]
    broken = {(o, "xgboost") for o in GRID[3:8]}  # 5 of 12 incomplete: 7 left
    with pytest.raises(oof.ThinWindowError, match="7 of the 12"):
        oof.rolling_origin_stack(_cells(GRID[3:15], failed=broken), origin, MODELS)
    enough = {(o, "xgboost") for o in GRID[3:7]}  # 8 left
    oof.rolling_origin_stack(_cells(GRID[3:15], failed=enough), origin, MODELS)


def test_the_fit_puts_the_weight_on_the_model_that_was_right():
    meta = oof.rolling_origin_fit(
        _cells(GRID[3:15]), GRID[15], MODELS, ensemble.fit_nnls_strict
    )
    assert meta.coef_[MODELS.index("arima")] == pytest.approx(1.0)


def test_an_all_zero_solution_is_an_error_not_equal_weights():
    X = -np.ones((10, 4))  # no non-negative weights move toward y
    y = np.ones(10)
    assert np.allclose(ensemble._fit_nnls(X, y).coef_, 0.25)  # the old fallback
    with pytest.raises(ValueError, match="all zero"):
        ensemble.fit_nnls_strict(X, y)


def _base(value, width=1.0):
    return pd.DataFrame({"yhat": np.full(H, value), "yhat_lower": value - width,
                         "yhat_upper": value + width})  # fmt: skip


def _data(origin):
    prices = pd.DataFrame({"date": DATES[: DATES.get_loc(origin) + 1], "close": 1.0})
    return OriginData(
        "AAPL", origin, prices, pd.DataFrame(), next_trading_days(origin, H)
    )


def test_the_combiner_weights_todays_base_forecasts_by_the_earlier_cells():
    origin = GRID[15]
    base = {"arima": _base(200.0), "prophet": _base(210.0), "xgboost": _base(190.0),
            "lightgbm": _base(230.0), "naive": _base(220.0)}  # fmt: skip
    out = forecasters.ensemble_rolling(_data(origin), base, _cells(GRID[3:15]))
    assert np.allclose(out["yhat"], 200.0)
    assert np.allclose(out["yhat_upper"] - out["yhat_lower"], 2.0)


def test_the_combiner_fails_on_a_thin_window():
    base = {m: _base(1.0) for m in MODELS}
    with pytest.raises(oof.ThinWindowError):
        forecasters.ensemble_rolling(_data(GRID[15]), base, _cells(GRID[10:15]))


def test_the_combiner_asks_the_harness_for_earlier_origins():
    assert getattr(forecasters.ensemble_rolling, "needs_history", False)
    assert not getattr(forecasters.ensemble_stack, "needs_history", False)


def _cells_by_horizon(origins):
    """ARIMA is exact at days 1 to 3, XGBoost at days 4 to 7; the rest miss."""
    rows = []
    for o in origins:
        for h in range(1, H + 1):
            target = DATES[DATES.get_loc(o) + h]
            actual = 100.0 + DATES.get_loc(target)
            right = "arima" if h <= 3 else "xgboost"
            for i, m in enumerate(MODELS):
                miss = 0.0 if m == right else 4.0 + i
                rows.append(
                    {"origin": o, "h": h, "target_date": target, "model": m,
                     "yhat": actual + miss, "actual": actual, "status": "ok"}
                )  # fmt: skip
    return pd.DataFrame(rows)


BASE_TODAY = {"arima": 200.0, "prophet": 210.0, "xgboost": 190.0, "lightgbm": 230.0,
              "naive": 220.0}  # fmt: skip


def test_per_horizon_weights_follow_the_best_model_day_by_day(monkeypatch):
    """E7: each day's weights come from that day's rows alone."""
    monkeypatch.setattr(forecasters, "ROLLING_PER_HORIZON", True)
    base = {m: _base(v) for m, v in BASE_TODAY.items()}
    out = forecasters.ensemble_rolling(
        _data(GRID[15]), base, _cells_by_horizon(GRID[3:15])
    )
    np.testing.assert_allclose(out["yhat"], [200.0] * 3 + [190.0] * 4)
    assert len(out) == H


def test_one_set_of_weights_blends_the_days(monkeypatch):
    monkeypatch.setattr(forecasters, "ROLLING_PER_HORIZON", False)
    base = {m: _base(v) for m, v in BASE_TODAY.items()}
    out = forecasters.ensemble_rolling(
        _data(GRID[15]), base, _cells_by_horizon(GRID[3:15])
    )
    assert out["yhat"].nunique() == 1
    assert 190.0 < out["yhat"].iloc[0] < 200.0
