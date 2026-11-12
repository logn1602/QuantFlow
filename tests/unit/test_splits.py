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


def test_the_warmup_origins_cover_the_rolling_weights_window():
    """The rolling ensemble fits its weights on the WINDOW_ORIGINS origins
    before each scored one, and WARMUP_ORIGINS exist for exactly that: two
    literals for one requirement. Raised alone, the window would give the
    first scored origins thinner windows than the rest (found in the
    full-diff review, 2026-11-12)."""
    from quantflow.evaluation.out_of_fold import WINDOW_ORIGINS

    assert splits.WARMUP_ORIGINS >= WINDOW_ORIGINS


# ── The registered windows, by date (decision D21) ───────────────────────────


def _calendar(end, start="2024-04-10", closed=("2025-01-09",)):
    """The project calendar's trading days less the given closures: the
    2026-11-06 snapshot's dates when end is 2026-11-06."""
    from quantflow.utils.trading_calendar import TRADING_DAY

    days = pd.date_range(start, end, freq=TRADING_DAY)
    return days[~days.isin(pd.DatetimeIndex(closed))]


def _span(dates, positions):
    first, last = dates[positions[0]], dates[positions[-1]]
    return first.date().isoformat(), last.date().isoformat()


def test_the_registered_grid_on_the_snapshot_is_the_grid_drawn_there():
    dates = _calendar("2026-11-06")
    assert len(dates) == SNAPSHOT_DATES
    assert splits.registered_grid(dates) == splits.origin_grid(SNAPSHOT_DATES)


@pytest.mark.parametrize("end", ["2026-11-18", "2026-11-20", "2027-05-28"])
def test_a_newer_snapshot_scores_the_same_dates(end):
    """Counted back from its last date, data to 2026-11-18 put the last test
    origin on 2026-11-09, the forward test's first origin, and data to
    2026-11-20 put test dates in the validation window (full-diff review,
    2026-11-12)."""
    dates = _calendar(end)
    grid = splits.registered_grid(dates)
    assert _span(dates, grid.validation) == ("2025-06-30", "2026-04-23")
    assert _span(dates, grid.validation_warmup) == ("2025-04-02", "2025-06-23")
    assert _span(dates, grid.test) == ("2026-05-07", "2026-10-28")
    assert _span(dates, grid.test_warmup) == ("2026-02-10", "2026-04-30")
    assert _span(dates, grid.tuning) == ("2024-10-01", "2025-03-19")
    assert grid.forward == grid.forward_warmup == ()


def test_a_snapshot_short_of_the_last_test_target_is_refused():
    with pytest.raises(ValueError, match="before the last test origin"):
        splits.registered_grid(_calendar("2026-11-05"))
    with pytest.raises(ValueError, match="do not include 2026-10-28"):
        splits.registered_grid(_calendar("2026-10-27"))


def test_history_that_differs_inside_the_windows_is_refused_not_shifted():
    dates = _calendar("2026-11-20")
    lost = dates.delete(dates.get_loc(pd.Timestamp("2025-09-02")))
    with pytest.raises(ValueError, match="validation window would run 2025-06-27"):
        splits.registered_grid(lost)


def test_the_forward_window_is_registered_and_empty_until_its_last_target():
    """D16: 42 origins every 5 trading days from 2026-11-09, the last on
    2027-09-03 with its day-7 target 2027-09-15, scored once that close is
    in, so before then the window is empty and nothing can look early."""
    assert splits.last_forward_target() == pd.Timestamp("2027-09-15")
    dates = _calendar("2027-09-15")
    grid = splits.registered_grid(dates)
    assert len(grid.forward) == splits.FORWARD_ORIGINS == 42
    assert _span(dates, grid.forward) == ("2026-11-09", "2027-09-03")
    assert dates[grid.forward[-1] + splits.HORIZON] == pd.Timestamp("2027-09-15")
    assert all(b - a == splits.ORIGIN_STEP for a, b in pairwise(grid.forward))
    assert len(grid.forward_warmup) == splits.WARMUP_ORIGINS
    assert grid.forward_warmup[-1] + splits.ORIGIN_STEP == grid.forward[0]
    assert grid.window("forward") == (grid.forward, grid.forward_warmup)
    assert splits.registered_grid(_calendar("2027-09-14")).forward == ()


def test_a_closed_day_is_not_a_trading_day_so_the_last_target_moves_a_day():
    """D22: the window keeps its 42 origins over the days the exchange
    opened, so a closure moves its end one trading day later."""
    closed = pd.Timestamp("2027-03-10")
    assert splits.last_forward_target([closed]) == pd.Timestamp("2027-09-16")
    dates = _calendar("2027-09-16", closed=("2025-01-09", "2027-03-10"))
    grid = splits.registered_grid(dates)
    assert len(grid.forward) == 42
    assert dates[grid.forward[-1] + splits.HORIZON] == pd.Timestamp("2027-09-16")
    assert splits.registered_grid(dates[:-1]).forward == ()
