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

import math
from dataclasses import dataclass

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


# ── Rolling origins (docs/evaluation.md, section 2) ──────────────────────────
#
# The second scheme: forecasts from many origins, each model refitted on the
# history up to its origin. It lives here too, so both ways of splitting time
# have one definition.

# Days ahead each origin forecasts: the published 7-day forecast. Equal to
# quantflow.models.FORECAST_DAYS, which evaluation cannot import; a test
# keeps them together.
HORIZON = 7
ORIGIN_STEP = 5  # trading days between origins
TEST_ORIGINS = 25
VALIDATION_ORIGINS = 42
WARMUP_ORIGINS = 12  # earlier origins that only fit the ensemble's weights
# The least history any origin may train on. The earliest origin of the
# grid on the 2026-11-06 snapshot has 246 rows.
MIN_ORIGIN_HISTORY = 120


@dataclass(frozen=True)
class OriginGrid:
    """Origin positions in a ticker's trading dates, oldest first.

    test and validation are scored; each warmup list only fits the
    ensemble's first weights for its window. Every validation origin's last
    target falls before the first test target, so no choice made on the
    validation dates sees a test-date close.
    """

    test: tuple[int, ...]
    test_warmup: tuple[int, ...]
    validation: tuple[int, ...]
    validation_warmup: tuple[int, ...]

    def window(self, name: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """(scored, warmup) origins of the "test" or "validation" window."""
        if name == "test":
            return self.test, self.test_warmup
        if name == "validation":
            return self.validation, self.validation_warmup
        raise ValueError(f"unknown window {name!r}; expected test or validation")


def origin_grid(
    n_dates: int,
    horizon: int = HORIZON,
    step: int = ORIGIN_STEP,
    n_test: int = TEST_ORIGINS,
    n_validation: int = VALIDATION_ORIGINS,
    n_warmup: int = WARMUP_ORIGINS,
) -> OriginGrid:
    """The grid for a series of n_dates trading dates.

    The last test origin is the last date whose horizon-day target is in
    the data. Validation origins sit on the same step, starting at the first
    one whose last target precedes the first test target. Raises ValueError
    if any origin would have fewer than MIN_ORIGIN_HISTORY rows to train on.
    """
    last = n_dates - 1 - horizon
    test = [last - step * i for i in range(n_test)][::-1]
    test_warmup = [test[0] - step * k for k in range(1, n_warmup + 1)][::-1]
    # A validation origin v forecasts up to v + horizon, which must come
    # before the first test target, test[0] + 1.
    first_gap = math.ceil(horizon / step)
    validation = [
        test[0] - step * k for k in range(first_gap, first_gap + n_validation)
    ][::-1]
    validation_warmup = [validation[0] - step * k for k in range(1, n_warmup + 1)][::-1]
    earliest = min(validation_warmup[0], test_warmup[0])
    if earliest + 1 < MIN_ORIGIN_HISTORY:
        raise ValueError(
            f"{n_dates} trading dates are too few for the grid: its earliest "
            f"origin would train on {earliest + 1} rows, under "
            f"{MIN_ORIGIN_HISTORY}"
        )
    return OriginGrid(
        tuple(test), tuple(test_warmup), tuple(validation), tuple(validation_warmup)
    )
