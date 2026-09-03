"""Scoring metrics.

One definition each for MAPE, RMSE and MAE. These were previously implemented
twice — compute_metrics in the ARIMA/Prophet module and a private _mape in the
ensemble — with the same formulas written out separately, which is how two
copies drift.

MAPE is the headline metric everywhere in this project because it is unit-free
and therefore comparable across tickers trading at very different prices. It is
undefined when an actual is zero; that cannot occur here, since every actual is
a traded close price.
"""

import numpy as np


def _as_array(values) -> np.ndarray:
    """Accept a pandas Series or a numpy array interchangeably."""
    return np.asarray(values, dtype=float)


def mape(actual, predicted) -> float:
    """Mean absolute percentage error, as a percentage."""
    a, p = _as_array(actual), _as_array(predicted)
    return float(np.mean(np.abs((a - p) / a)) * 100)


def rmse(actual, predicted) -> float:
    """Root mean squared error, in price units."""
    a, p = _as_array(actual), _as_array(predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mae(actual, predicted) -> float:
    """Mean absolute error, in price units."""
    a, p = _as_array(actual), _as_array(predicted)
    return float(np.mean(np.abs(a - p)))


def regression_metrics(actual, predicted) -> dict:
    """All three metrics, rounded for storage and logging.

    Rounding happens here rather than at each call site so the numbers written
    to model_metrics, logged to MLflow, and printed to the console cannot
    disagree about precision.
    """
    return {
        "rmse": round(rmse(actual, predicted), 4),
        "mae": round(mae(actual, predicted), 4),
        "mape": round(mape(actual, predicted), 2),
    }
