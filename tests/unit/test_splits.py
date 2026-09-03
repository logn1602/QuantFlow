"""tests/unit/test_splits.py

Covers quantflow.evaluation.splits.

This is a guardrail suite. The split is what separates data a model learned
from data it is scored on, and every number in the README depends on it holding.
A split that quietly stopped being time-ordered would not fail loudly anywhere
else — the models would still train, still report a MAPE, and the MAPE would
just be wrong. These tests exist so that failure is loud.
"""

import pandas as pd
import pytest

from quantflow.evaluation.splits import (
    HOLDOUT_DAYS,
    has_enough_history,
    train_holdout_split,
)


@pytest.fixture
def series():
    """A 100-row date-ordered frame; `day` is strictly increasing."""
    return pd.DataFrame(
        {
            "ds": pd.bdate_range("2026-01-01", periods=100),
            "day": range(100),
        }
    )


# ── The ordering guarantee ────────────────────────────────────────────────────


def test_train_is_strictly_earlier_than_holdout(series):
    """The property the whole evaluation rests on: no training row may be
    dated at or after any holdout row."""
    train, holdout = train_holdout_split(series)
    assert train["ds"].max() < holdout["ds"].min()


def test_split_is_positional_not_random(series):
    """A shuffled split would interleave the two sides. Every training index
    must be below every holdout index."""
    train, holdout = train_holdout_split(series)
    assert list(train["day"]) == list(range(100 - HOLDOUT_DAYS))
    assert list(holdout["day"]) == list(range(100 - HOLDOUT_DAYS, 100))


def test_split_is_deterministic(series):
    """Two calls must agree. A seeded shuffle would also be deterministic, so
    this is necessary but not sufficient — read it with the test above."""
    first = train_holdout_split(series)
    second = train_holdout_split(series)
    assert first[0].equals(second[0])
    assert first[1].equals(second[1])


def test_no_row_appears_in_both_sides(series):
    train, holdout = train_holdout_split(series)
    assert set(train["day"]).isdisjoint(set(holdout["day"]))


def test_split_covers_every_row(series):
    train, holdout = train_holdout_split(series)
    assert len(train) + len(holdout) == len(series)


# ── Sizing ────────────────────────────────────────────────────────────────────


def test_holdout_length_is_holdout_days(series):
    _, holdout = train_holdout_split(series)
    assert len(holdout) == HOLDOUT_DAYS


def test_custom_holdout_length_is_honoured(series):
    train, holdout = train_holdout_split(series, holdout_days=10)
    assert len(holdout) == 10
    assert len(train) == 90


# ── copy semantics ────────────────────────────────────────────────────────────


def test_copy_false_returns_views_sharing_the_source(series):
    train, _ = train_holdout_split(series, copy=False)
    assert train._is_view or train.index.equals(series.index[: 100 - HOLDOUT_DAYS])


def test_copy_true_isolates_the_caller_from_the_source(series):
    """The gradient boosters fill NaNs in place on their split, so mutation
    must not reach back into the caller's frame."""
    train, _ = train_holdout_split(series, copy=True)
    train.loc[train.index[0], "day"] = -999
    assert series.loc[0, "day"] == 0


# ── has_enough_history ────────────────────────────────────────────────────────


def test_rejects_a_frame_with_no_room_for_training(series):
    """A frame exactly as long as the holdout leaves nothing to train on."""
    assert has_enough_history(series.iloc[:HOLDOUT_DAYS]) is False


def test_rejects_a_frame_shorter_than_the_holdout(series):
    assert has_enough_history(series.iloc[:5]) is False


def test_accepts_a_frame_with_holdout_plus_training_room(series):
    assert has_enough_history(series) is True


def test_boundary_is_inclusive(series):
    """Exactly HOLDOUT_DAYS + minimum_train_days is enough, one row less is not."""
    minimum = 30
    assert has_enough_history(series.iloc[: HOLDOUT_DAYS + minimum], minimum) is True
    assert (
        has_enough_history(series.iloc[: HOLDOUT_DAYS + minimum - 1], minimum) is False
    )
