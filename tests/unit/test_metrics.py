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
