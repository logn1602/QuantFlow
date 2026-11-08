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


# ── Rolling origins ───────────────────────────────────────────────────────────

from itertools import pairwise  # noqa: E402

from quantflow.evaluation import splits  # noqa: E402

SNAPSHOT_DATES = 648  # trading dates per ticker, 2024-04-10 to 2026-11-06


def test_the_horizon_is_the_published_forecast_length():
    from quantflow.models import FORECAST_DAYS

    assert splits.HORIZON == FORECAST_DAYS


def test_the_grid_on_the_snapshot_matches_the_design():
    from quantflow.utils.trading_calendar import TRADING_DAY

    days = pd.date_range("2024-04-10", "2026-11-06", freq=TRADING_DAY)
    days = days[days != pd.Timestamp("2025-01-09")]  # the one-off closure
    assert len(days) == SNAPSHOT_DATES
    grid = splits.origin_grid(len(days))

    def span(positions):
        return days[positions[0]].date().isoformat(), days[
            positions[-1]
        ].date().isoformat()

    assert len(grid.test) == 25 and span(grid.test) == ("2026-05-07", "2026-10-28")
    assert span(grid.test_warmup) == ("2026-02-10", "2026-04-30")
    assert len(grid.validation) == 42
    assert span(grid.validation) == ("2025-06-30", "2026-04-23")
    assert span(grid.validation_warmup) == ("2025-04-02", "2025-06-23")
    assert len(grid.tuning) == 24
    assert span(grid.tuning) == ("2024-10-01", "2025-03-19")


def test_every_window_steps_evenly_and_ends_where_the_data_does():
    grid = splits.origin_grid(SNAPSHOT_DATES)
    for positions in (
        grid.test,
        grid.test_warmup,
        grid.validation,
        grid.validation_warmup,
    ):
        assert all(b - a == splits.ORIGIN_STEP for a, b in pairwise(positions))
    assert grid.test[-1] + splits.HORIZON == SNAPSHOT_DATES - 1
    assert grid.test_warmup[-1] + splits.ORIGIN_STEP == grid.test[0]
    assert grid.validation_warmup[-1] + splits.ORIGIN_STEP == grid.validation[0]


def test_no_validation_target_reaches_the_test_window():
    grid = splits.origin_grid(SNAPSHOT_DATES)
    first_test_target = grid.test[0] + 1
    assert grid.validation[-1] + splits.HORIZON < first_test_target
    assert (
        grid.validation[-1] + splits.ORIGIN_STEP + splits.HORIZON >= first_test_target
    )


def test_tuning_ends_before_the_validation_warmup_and_has_enough_history():
    """Booster settings are tuned only on closes that every validation and
    test origin trains on (Phase 3, E3)."""
    grid = splits.origin_grid(SNAPSHOT_DATES)
    assert all(b - a == splits.ORIGIN_STEP for a, b in pairwise(grid.tuning))
    assert grid.tuning[-1] + splits.HORIZON < grid.validation_warmup[0]
    assert grid.tuning[0] + 1 >= splits.MIN_ORIGIN_HISTORY
    assert grid.tuning[0] - splits.ORIGIN_STEP + 1 < splits.MIN_ORIGIN_HISTORY


def test_a_window_is_chosen_by_name():
    grid = splits.origin_grid(SNAPSHOT_DATES)
    assert grid.window("test") == (grid.test, grid.test_warmup)
    assert grid.window("validation") == (grid.validation, grid.validation_warmup)
    assert grid.window("tuning") == (grid.tuning, ())
    with pytest.raises(ValueError, match="unknown window"):
        grid.window("holdout")


def test_too_little_history_is_refused():
    with pytest.raises(ValueError, match="too few"):
        splits.origin_grid(400)
