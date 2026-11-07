"""Scoring metrics.

One definition each for MAPE, RMSE and MAE, and for the measures the
rolling-origin evaluation adds (docs/evaluation.md, section 3): directional
accuracy, skill against the naive forecast, prediction-band coverage and
width, and the primary test of a model against naive. These were previously implemented
twice — compute_metrics in the ARIMA/Prophet module and a private _mape in the
ensemble — with the same formulas written out separately, which is how two
copies drift.

MAPE is the headline metric everywhere in this project because it is unit-free
and therefore comparable across tickers trading at very different prices. It is
undefined when an actual is zero; that cannot occur here, since every actual is
a traded close price.
"""

import math

import numpy as np
from scipy import stats


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


# ── Rolling-origin measures ───────────────────────────────────────────────────


def directional_accuracy(origin_close, actual, predicted) -> float | None:
    """The share, as a percentage, of forecasts that move the same way from
    the origin's close as the actual close did.

    A forecast or an actual equal to the origin's close has no direction and
    is left out, so the naive forecast has none: None when nothing is left.
    """
    o, a, p = _as_array(origin_close), _as_array(actual), _as_array(predicted)
    actual_move, predicted_move = a - o, p - o
    keep = (
        np.isfinite(actual_move)
        & np.isfinite(predicted_move)
        & (actual_move != 0)
        & (predicted_move != 0)
    )
    if not keep.any():
        return None
    agree = np.sign(actual_move[keep]) == np.sign(predicted_move[keep])
    return float(np.mean(agree) * 100)


def up_share(origin_close, actual) -> float | None:
    """The share of actual moves that were up, as a percentage: what a
    forecast that always says "up" scores on directional accuracy."""
    move = _as_array(actual) - _as_array(origin_close)
    move = move[np.isfinite(move) & (move != 0)]
    return float(np.mean(move > 0) * 100) if len(move) else None


def skill(model_error: float, naive_error: float) -> float | None:
    """1 - model_error / naive_error: positive when the model beats naive,
    0 when it matches, None when naive made no error to compare with."""
    if not (math.isfinite(model_error) and math.isfinite(naive_error)):
        return None
    if naive_error == 0:
        return None
    return 1 - model_error / naive_error


def band_coverage(actual, lower, upper) -> float | None:
    """The share of actual closes inside the forecast's band, as a
    percentage, over the forecasts that have a band."""
    a, lo, hi = _as_array(actual), _as_array(lower), _as_array(upper)
    keep = np.isfinite(a) & np.isfinite(lo) & np.isfinite(hi)
    if not keep.any():
        return None
    return float(np.mean((a[keep] >= lo[keep]) & (a[keep] <= hi[keep])) * 100)


def band_width(predicted, lower, upper) -> float | None:
    """The band's mean width as a percentage of the forecast itself."""
    p, lo, hi = _as_array(predicted), _as_array(lower), _as_array(upper)
    keep = np.isfinite(p) & np.isfinite(lo) & np.isfinite(hi) & (p != 0)
    if not keep.any():
        return None
    return float(np.mean((hi[keep] - lo[keep]) / np.abs(p[keep])) * 100)


def newey_west_variance_of_mean(values, lag: int) -> float:
    """The variance of the mean of a series whose neighbours are correlated
    up to `lag` steps apart (Bartlett weights).

    Consecutive origins five trading days apart share one or two target days
    at horizons 6 and 7, so their errors are not independent; lag 1 covers
    that.
    """
    x = _as_array(values)
    n = len(x)
    centred = x - x.mean()
    total = float(np.dot(centred, centred)) / n
    for k in range(1, lag + 1):
        weight = 1 - k / (lag + 1)
        total += 2 * weight * float(np.dot(centred[k:], centred[:-k])) / n
    return max(total, 0.0) / n


def mean_difference_test(differences, lag: int = 1) -> dict | None:
    """Is the mean of the per-origin differences zero?

    differences are, per origin, a model's error minus naive's, already
    averaged over tickers and horizons: negative means the model did better.
    Two-sided Student t with n - 1 degrees of freedom on a Newey-West
    standard error. None with fewer than 3 origins. Identical errors give a
    zero standard error: p is 1 when the mean is 0, and 0 otherwise.
    """
    d = _as_array(differences)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return None
    mean = float(d.mean())
    se = math.sqrt(newey_west_variance_of_mean(d, lag))
    if se == 0:
        return {"n": n, "mean": mean, "se": 0.0, "t": math.nan,
                "p": 1.0 if mean == 0 else 0.0}  # fmt: skip
    t = mean / se
    p = float(2 * stats.t.sf(abs(t), df=n - 1))
    return {"n": n, "mean": mean, "se": se, "t": t, "p": p}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values, so testing several models against naive
    keeps the chance of any false "beats naive" at the stated level."""
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return adjusted
