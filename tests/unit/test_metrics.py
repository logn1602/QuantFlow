"""tests/unit/test_metrics.py

Covers quantflow.evaluation.metrics.

These three formulas produce every number this project reports, including the
figures quoted in the README. They were previously written out four times
across the codebase; consolidating them is only safe if the single remaining
copy is pinned to hand-computed values, which is what this file does.
"""

import numpy as np
import pandas as pd
import pytest

from quantflow.evaluation.metrics import mae, mape, regression_metrics, rmse

# ── Hand-computed values ──────────────────────────────────────────────────────


def test_mape_matches_hand_computed_value():
    # |10/100| = 0.10, |20/200| = 0.10 -> mean 0.10 -> 10%
    assert mape([100.0, 200.0], [110.0, 180.0]) == pytest.approx(10.0)


def test_rmse_matches_hand_computed_value():
    # errors 3 and 4 -> mean square 12.5 -> sqrt = 3.5355...
    assert rmse([10.0, 20.0], [13.0, 16.0]) == pytest.approx(np.sqrt(12.5))


def test_mae_matches_hand_computed_value():
    # errors 3 and 4 -> mean 3.5
    assert mae([10.0, 20.0], [13.0, 16.0]) == pytest.approx(3.5)


def test_all_metrics_are_zero_for_perfect_predictions():
    actual = np.array([50.0, 75.0, 120.0])
    assert mape(actual, actual.copy()) == pytest.approx(0.0)
    assert rmse(actual, actual.copy()) == pytest.approx(0.0)
    assert mae(actual, actual.copy()) == pytest.approx(0.0)


# ── Properties ────────────────────────────────────────────────────────────────


def test_rmse_is_never_below_mae():
    """A standing inequality for any error vector; a sign error in either
    formula tends to break it."""
    rng = np.random.default_rng(11)
    for _ in range(20):
        actual = 100 + rng.normal(0, 10, 40)
        predicted = actual + rng.normal(0, 5, 40)
        assert rmse(actual, predicted) >= mae(actual, predicted) - 1e-9


def test_metrics_are_symmetric_in_magnitude_of_error():
    """Over- and under-predicting by the same amount must score identically."""
    actual = np.array([100.0, 100.0])
    assert mae(actual, np.array([110.0, 90.0])) == pytest.approx(
        mae(actual, np.array([90.0, 110.0]))
    )


def test_metrics_are_scale_dependent_but_mape_is_not():
    """MAPE is the headline metric because it compares across tickers trading
    at different price levels. Scaling both series must not move it."""
    actual = np.array([100.0, 200.0])
    predicted = np.array([110.0, 180.0])
    assert mape(actual * 7, predicted * 7) == pytest.approx(mape(actual, predicted))
    assert mae(actual * 7, predicted * 7) == pytest.approx(mae(actual, predicted) * 7)


# ── Input handling ────────────────────────────────────────────────────────────


def test_accepts_pandas_series_and_numpy_arrays_interchangeably():
    """Call sites pass both: the ARIMA path holds Series, the boosting path
    holds arrays."""
    actual = [100.0, 200.0, 300.0]
    predicted = [110.0, 180.0, 330.0]
    assert mape(pd.Series(actual), pd.Series(predicted)) == pytest.approx(
        mape(np.array(actual), np.array(predicted))
    )


def test_accepts_integer_input_without_integer_division():
    """Integer price series must not truncate the ratio to zero."""
    assert mape(np.array([100, 200]), np.array([110, 180])) == pytest.approx(10.0)


# ── regression_metrics ────────────────────────────────────────────────────────


def test_regression_metrics_returns_all_three_keys():
    result = regression_metrics([100.0, 200.0], [110.0, 180.0])
    assert set(result) == {"rmse", "mae", "mape"}


def test_regression_metrics_rounding_is_fixed():
    """Rounding lives in one place so the database, MLflow and the console
    cannot disagree about precision."""
    result = regression_metrics([100.0, 200.0], [110.0, 180.0])
    assert result["mape"] == round(mape([100.0, 200.0], [110.0, 180.0]), 2)
    assert result["rmse"] == round(rmse([100.0, 200.0], [110.0, 180.0]), 4)
    assert result["mae"] == round(mae([100.0, 200.0], [110.0, 180.0]), 4)


def test_regression_metrics_values_are_plain_floats():
    """numpy scalars serialise badly into psycopg2 and MLflow."""
    for value in regression_metrics([100.0, 200.0], [110.0, 180.0]).values():
        assert isinstance(value, float)


# ── Rolling-origin measures (docs/evaluation.md, section 3) ───────────────────

import math  # noqa: E402

from quantflow.evaluation import metrics as m  # noqa: E402

ORIGIN = [100.0, 100.0, 100.0, 100.0]
ACTUAL = [101.0, 99.0, 100.0, 102.0]  # up, down, flat, up


def test_directional_accuracy_leaves_out_moves_without_a_direction():
    # The flat actual and the flat forecast are left out; the other two agree.
    assert m.directional_accuracy(ORIGIN, ACTUAL, [102.0, 98.0, 101.0, 100.0]) == 100.0
    assert m.directional_accuracy(ORIGIN[:2], ACTUAL[:2], [99.0, 98.0]) == 50.0


def test_naive_has_no_directional_accuracy():
    assert m.directional_accuracy(ORIGIN, ACTUAL, ORIGIN) is None


def test_up_share_is_what_always_up_would_score():
    assert m.up_share(ORIGIN, ACTUAL) == pytest.approx(200 / 3)


@pytest.mark.parametrize(
    ("model", "naive", "expected"),
    [(1.0, 2.0, 0.5), (2.0, 2.0, 0.0), (3.0, 2.0, -0.5), (1.0, 0.0, None),
     (math.nan, 2.0, None)],
)  # fmt: skip
def test_skill_against_naive(model, naive, expected):
    assert m.skill(model, naive) == expected


def test_band_coverage_counts_only_forecasts_with_a_band():
    assert m.band_coverage([10, 12, 15], [9, 9, math.nan], [11, 11, 16]) == 50.0
    assert m.band_coverage([10], [math.nan], [math.nan]) is None


def test_band_width_is_relative_to_the_forecast():
    assert m.band_width([100, 200], [98, 190], [102, 210]) == pytest.approx(7.0)


def test_newey_west_adds_the_neighbour_correlation():
    alternating = [1.0, -1.0, 1.0, -1.0]
    assert m.newey_west_variance_of_mean(alternating, lag=0) == pytest.approx(0.25)
    # gamma1 = -0.75, weight 1/2: (1 + 2 * 0.5 * -0.75) / 4
    assert m.newey_west_variance_of_mean(alternating, lag=1) == pytest.approx(0.0625)


def test_the_mean_difference_test_on_a_hand_worked_series():
    result = m.mean_difference_test([-1, -2, -3, -1, -2, -3], lag=0)
    # mean -2; centred values give gamma0 = 4/6, so se = sqrt(4/36) = 1/3.
    assert result["mean"] == pytest.approx(-2.0)
    assert result["se"] == pytest.approx(1 / 3)
    assert result["t"] == pytest.approx(-6.0)
    assert result["p"] < 0.01


def test_the_mean_difference_test_edge_cases():
    assert m.mean_difference_test([0.1, 0.2]) is None  # fewer than 3 origins
    assert m.mean_difference_test([0.0] * 5)["p"] == 1.0  # identical errors
    assert m.mean_difference_test([0.5] * 5)["p"] == 0.0  # always worse, no spread


def test_holm_adjusts_in_order_and_never_decreases():
    adjusted = m.holm({"arima": 0.01, "boost": 0.04, "prophet": 0.03})
    assert adjusted == pytest.approx({"arima": 0.03, "prophet": 0.06, "boost": 0.06})
