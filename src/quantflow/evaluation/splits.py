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
from collections.abc import Iterable
from dataclasses import dataclass, replace

import pandas as pd

from quantflow.utils.trading_calendar import next_trading_days

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
    validation dates sees a test-date close. tuning (Phase 3's booster
    settings) lies wholly before the validation warmup: its targets are
    training data for every validation and test origin. forward (D16) is
    the forward test's window, empty until the data holds its last target,
    since it is scored once (registered_grid).
    """

    test: tuple[int, ...]
    test_warmup: tuple[int, ...]
    validation: tuple[int, ...]
    validation_warmup: tuple[int, ...]
    tuning: tuple[int, ...] = ()
    forward: tuple[int, ...] = ()
    forward_warmup: tuple[int, ...] = ()

    def window(self, name: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """(scored, warmup) origins of the "test", "validation", "tuning" or
        "forward" window; tuning has no warmup."""
        if name == "test":
            return self.test, self.test_warmup
        if name == "validation":
            return self.validation, self.validation_warmup
        if name == "tuning":
            return self.tuning, ()
        if name == "forward":
            return self.forward, self.forward_warmup
        raise ValueError(
            f"unknown window {name!r}; expected test, validation, tuning or forward"
        )


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
    # Tuning origins, on the same step back from the validation warmup:
    # each one's last target comes before the first warmup origin, and the
    # earliest still has MIN_ORIGIN_HISTORY rows to train on.
    tuning_gap = math.ceil((horizon + 1) / step)
    tuning = []
    k = tuning_gap
    while validation_warmup[0] - step * k + 1 >= MIN_ORIGIN_HISTORY:
        tuning.append(validation_warmup[0] - step * k)
        k += 1
    return OriginGrid(
        tuple(test),
        tuple(test_warmup),
        tuple(validation),
        tuple(validation_warmup),
        tuple(tuning[::-1]),
    )


# ── The registered windows, by date (decision D21) ───────────────────────────
#
# origin_grid counts back from a series' last date. That is how the windows
# were drawn, on the snapshot of 2026-11-06, but counted that way on a newer
# snapshot they would move: with data to 2026-11-18 the last test origin
# would be 2026-11-09, the forward test's first origin, and from 2026-11-20
# the validation window would hold test dates, all without a trace. So every
# command takes its grid from registered_grid, which anchors it at the last
# test origin's date and checks each window against the dates below.

# The snapshot the windows were drawn on (docs/evaluation.md, section 2).
REGISTERED_AS_OF = pd.Timestamp("2026-11-06")
# Each window's first and last origin (D1, D1a). The tuning window is left
# to the data's first date, as it always was: it is every origin before the
# validation warmup with MIN_ORIGIN_HISTORY rows behind it.
REGISTERED_WINDOWS = {
    "validation": (pd.Timestamp("2025-06-30"), pd.Timestamp("2026-04-23")),
    "validation_warmup": (pd.Timestamp("2025-04-02"), pd.Timestamp("2025-06-23")),
    "test": (pd.Timestamp("2026-05-07"), pd.Timestamp("2026-10-28")),
    "test_warmup": (pd.Timestamp("2026-02-10"), pd.Timestamp("2026-04-30")),
}
# The forward test (D16): FORWARD_ORIGINS origins every ORIGIN_STEP trading
# days from the first trading date after REGISTERED_AS_OF, scored once, when
# the last one's day-HORIZON close is in.
FIRST_FORWARD_ORIGIN = pd.Timestamp("2026-11-09")
FORWARD_ORIGINS = 42


def last_forward_target(closed: Iterable = ()) -> pd.Timestamp:
    """The date of the forward window's last target: 2027-09-15 by the
    calendar, one trading day later for each listed closure (closed, D22).
    The forward test is scored once this close is in, and not before."""
    days = (FORWARD_ORIGINS - 1) * ORIGIN_STEP + HORIZON + 1
    return next_trading_days(REGISTERED_AS_OF, days, closed=closed)[-1]


def registered_grid(dates) -> OriginGrid:
    """The registered grid on a snapshot's trading dates: each window on its
    registered dates, whatever date the snapshot ends on, and the forward
    window (D16) once the data holds its last target (empty before, so no
    command can score part of it).

    Raises ValueError when the dates do not reach the last test origin's
    day-HORIZON target, or when a window's first or last origin is not its
    registered date, as when a snapshot's history inside the windows
    differs from the 2026-11-06 snapshot's: the windows are fixed by date,
    and a grid that cannot match them is refused rather than shifted.
    """
    dates = pd.DatetimeIndex(dates)
    last_test = REGISTERED_WINDOWS["test"][1]
    if last_test not in dates:
        raise ValueError(
            f"the snapshot's dates do not include {last_test.date()}, the last "
            f"test origin; the windows are fixed by date"
        )
    position = dates.get_loc(last_test)
    if position + HORIZON >= len(dates):
        raise ValueError(
            f"the snapshot ends {dates[-1].date()}, before the last test "
            f"origin's day-{HORIZON} target"
        )
    grid = origin_grid(position + HORIZON + 1)
    for name, registered in REGISTERED_WINDOWS.items():
        positions = getattr(grid, name)
        found = (dates[positions[0]], dates[positions[-1]])
        if found != registered:
            raise ValueError(
                f"on this snapshot the {name} window would run "
                f"{found[0].date()} to {found[1].date()}, not "
                f"{registered[0].date()} to {registered[1].date()} as "
                f"registered: its trading dates differ from the 2026-11-06 "
                f"snapshot's inside the windows"
            )
    return replace(grid, **_forward(dates))


def _forward(dates: pd.DatetimeIndex) -> dict[str, tuple[int, ...]]:
    """The forward window's origins and warmup in dates, both empty while the
    data does not hold the last origin's day-HORIZON close. Counted over the
    dates in the data, which are the days the exchange opened: a closed day
    (D22) is not one of them."""
    if FIRST_FORWARD_ORIGIN not in dates:
        return {"forward": (), "forward_warmup": ()}
    first = dates.get_loc(FIRST_FORWARD_ORIGIN)
    forward = tuple(first + ORIGIN_STEP * k for k in range(FORWARD_ORIGINS))
    if forward[-1] + HORIZON >= len(dates):
        return {"forward": (), "forward_warmup": ()}
    warmup = tuple(first - ORIGIN_STEP * k for k in range(WARMUP_ORIGINS, 0, -1))
    return {"forward": forward, "forward_warmup": warmup}
