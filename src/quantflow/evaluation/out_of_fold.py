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

The rolling-origin form below (docs/evaluation.md, section 5) fits the
weights on the task they are used for: the base models' 7-day forecasts at
the grid origins before an origin, keeping only rows whose target is known
by it. The evaluation and the live ensemble call the same function.
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


# ── The rolling-origin form ───────────────────────────────────────────────────

# Grid origins before an origin whose forecasts the weights are fitted on, and
# how many of them must have every base model's forecast.
WINDOW_ORIGINS = 12
MIN_COMPLETE_ORIGINS = 8


class ThinWindowError(RuntimeError):
    """Too few complete earlier origins to fit the weights on."""


def rolling_origin_stack(
    cells: pd.DataFrame,
    origin,
    models: list[str],
    window: int = WINDOW_ORIGINS,
    min_complete: int = MIN_COMPLETE_ORIGINS,
) -> pd.DataFrame:
    """The rows the weights at `origin` may be fitted on.

    cells are rolling-origin cells (origin, h, target_date, model, yhat,
    actual, status). From the `window` latest origins before `origin`, the
    complete ones (every model in `models` has a scorable forecast for every
    horizon there) give one row per (origin, h) whose target date is on or
    before `origin`: origin, h, one column per model, actual. Origin o - 5's
    horizons 6 and 7 fall after o, so they are left out.

    Raises ThinWindowError if fewer than `min_complete` origins are
    complete: the ensemble fails rather than guessing.
    """
    origin = pd.Timestamp(origin)
    dated = cells.assign(
        origin=pd.to_datetime(cells["origin"]),
        target_date=pd.to_datetime(cells["target_date"]),
    )
    base = dated[dated["model"].isin(models) & (dated["origin"] < origin)]
    recent = sorted(base["origin"].unique())[-window:]
    base = base[base["origin"].isin(recent)]
    scorable = base["status"].eq("ok") & np.isfinite(
        pd.to_numeric(base["yhat"], errors="coerce").to_numpy(dtype=float)
    )
    per_model = (
        base.assign(scorable=scorable)
        .groupby(["origin", "model"])["scorable"]
        .all()
        .unstack("model")
        .reindex(columns=models)
    )
    complete = per_model.index[per_model.fillna(False).astype(bool).all(axis=1)]
    if len(complete) < min_complete:
        raise ThinWindowError(
            f"{len(complete)} of the {len(recent)} origins before "
            f"{origin.date()} have every base forecast; {min_complete} needed"
        )
    rows = base[base["origin"].isin(complete) & (base["target_date"] <= origin)]
    wide = rows.pivot_table(
        index=["origin", "h"], columns="model", values="yhat", aggfunc="first"
    )[models]
    actual = rows.groupby(["origin", "h"])["actual"].first()
    return wide.assign(actual=actual).reset_index()


def rolling_origin_fit(
    cells: pd.DataFrame,
    origin,
    models: list[str],
    fit_fn: Callable,
    window: int = WINDOW_ORIGINS,
    min_complete: int = MIN_COMPLETE_ORIGINS,
):
    """The meta-learner for `origin`: fit_fn(X, y) on rolling_origin_stack's
    rows, X the base forecasts in `models` order, y the actuals."""
    stack = rolling_origin_stack(cells, origin, models, window, min_complete)
    return fit_fn(
        stack[models].to_numpy(dtype=float), stack["actual"].to_numpy(dtype=float)
    )
