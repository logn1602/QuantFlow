"""Expanding-window out-of-fold evaluation.

This is the module that makes the ensemble's reported numbers honest, and it is
the one worth reading first.

A stacking meta-learner scored on the same rows it was fitted on cannot lose:
it has already seen every actual it is being graded against. An earlier version
of this project did exactly that and reported the ensemble as the best model on
all eight tickers. With the leak removed it wins on one. The README says so.

The loop here fits the meta-learner on days [0, i) and predicts day i, for each
i from `warmup` onward. No prediction ever sees its own actual, or any later
one. Fitting the first meta-learner consumes `warmup` days, so the evaluation
window is shorter than the holdout — 20 days rather than 30 — and base models
must be re-scored on that same window for the comparison to be like-for-like.

The loop takes the fitting function as an argument rather than importing one.
That keeps evaluation independent of any particular meta-learner, and avoids a
circular import: quantflow.models.ensemble imports this module, not the reverse.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from quantflow.utils.logger import get_logger

logger = get_logger(__name__)

# Days consumed fitting the first meta-learner before scoring can begin.
DEFAULT_WARMUP_DAYS = 10


def expanding_window_predictions(
    stacked_df: pd.DataFrame,
    feature_cols: list[str],
    fit_fn: Callable,
    warmup: int = DEFAULT_WARMUP_DAYS,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold predictions over an expanding training window.

    For each day i in [warmup, len(stacked_df)):
        meta = fit_fn(X[:i], y[:i])   # strictly earlier days only
        pred = meta.predict(X[i])

    Returns (predictions, actuals), both of length len(stacked_df) - warmup.
    Returns two empty arrays when the frame is too short to leave any
    out-of-fold day, which is a real case for a thinly traded ticker.
    """
    X = stacked_df[feature_cols].values
    y = stacked_df["actual"].values

    if len(stacked_df) <= warmup:
        logger.warning(
            f"  Stack has {len(stacked_df)} rows, warmup is {warmup} — "
            f"no out-of-fold days available"
        )
        return np.array([]), np.array([])

    preds = np.empty(len(stacked_df) - warmup, dtype=float)
    for j, i in enumerate(range(warmup, len(stacked_df))):
        meta = fit_fn(X[:i], y[:i])
        preds[j] = float(meta.predict(X[i : i + 1])[0])

    return preds, y[warmup:]
