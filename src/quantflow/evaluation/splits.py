"""The train/holdout boundary.

Every model in this project is evaluated the same way: fit on everything except
the last HOLDOUT_DAYS observations, score on those. This module is the only
place that split is defined.

It matters that it is a single definition. The split was previously written out
at six sites across three modules, each with its own HOLDOUT_DAYS constant, and
a single one drifting out of step would have produced metrics that looked
comparable but were not.

The split is positional on a frame the caller has already sorted by date, so
train is strictly earlier than holdout. It is deliberately not a random split:
shuffling a price series lets a model interpolate between observations it will
be scored on, which inflates every error metric and invalidates the comparison
against a buy-and-hold benchmark.
"""

import pandas as pd

# Trading days held out for evaluation. Roughly six calendar weeks.
HOLDOUT_DAYS = 30


def train_holdout_split(
    df: pd.DataFrame,
    holdout_days: int = HOLDOUT_DAYS,
    copy: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a date-ordered frame into (train, holdout).

    The caller is responsible for having sorted by date; this function cannot
    verify it cheaply for every frame shape in use. Every caller loads through
    quantflow.db.prices, which sorts ascending in SQL.

    copy=True returns independent frames, needed where the caller then mutates
    the result (the gradient boosters fill NaNs in place).
    """
    train = df.iloc[:-holdout_days]
    holdout = df.iloc[-holdout_days:]
    return (train.copy(), holdout.copy()) if copy else (train, holdout)


def has_enough_history(df: pd.DataFrame, minimum_train_days: int = 30) -> bool:
    """True when a frame has a holdout plus a usable training period."""
    return len(df) >= HOLDOUT_DAYS + minimum_train_days
