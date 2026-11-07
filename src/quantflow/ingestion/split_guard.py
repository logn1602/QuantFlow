"""
ingestion/split_guard.py
------------------------
Fail the refresh when stored prices show a stock split they were never
adjusted for.

Yahoo serves every bar split-adjusted as of the moment it is fetched, and
rows here are never rewritten. So after a split, every row stored before it
sits on the old share basis (a 10-for-1 split leaves them ten times too high)
and the models would read the split as a 90% crash. Two checks look for that.

The basis check compares each ticker's stored daily closes with Yahoo's
current raw closes for the same dates and fails loudly when they disagree by
more than SPLIT_GUARD_THRESHOLD on a run of consecutive dates.

  Why a threshold rather than equality: stored closes legitimately differ
  from Yahoo's raw closes a little. The seeded history was stored
  dividend-adjusted as of the seed date (about 2% a year at most on these
  tickers), and dates built from 15-min bars differ from the official close
  by a few hundredths of a percent. A split moves prices by far more.

  Why a run of dates: a split shifts every row stored before it, which is a
  long unbroken run. One or two isolated mismatches are a bad print on one
  side, and failing the refresh on them would stop every stage until someone
  hand-edited a row. They are logged as warnings instead.

The jump check fails on any stored close at or below FALL_LIMIT, or at or
above RISE_LIMIT, times the stored close before it: a move the size of a
split. It needs nothing from Yahoo, so it still catches a split on a day
Yahoo's reference download fails, when the basis check cannot run and the
scheduler would otherwise reuse the previous trading day's pass.

A real move of that size is listed, with its reason, in
config/split_jump_acknowledged.txt (read by config.split_jump_acknowledged).
The listing silences the move only when the basis check ran for that ticker
in the same run and Yahoo's raw closes agree with the stored ones on the move
date and the date before. An acknowledgement confirms a market move: a move
Yahoo does not show was made by a stored row, and a move nothing could check
is not confirmed, so both still fail.

Neither check keeps state: once the affected rows are rescaled, the guard
passes again by itself. Rescaling is a scoped UPDATE on the live database,
run by hand, by the ratio in Yahoo's split events, which the command lists
when it fails.

Usage:
    python -m quantflow.ingestion.split_guard
"""

import argparse
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from fractions import Fraction

import pandas as pd

from quantflow import config
from quantflow.config import TICKERS
from quantflow.db.prices import load_daily_bars
from quantflow.ingestion.yfinance_fetcher import download, split_events
from quantflow.utils.logger import get_logger
from quantflow.utils.trading_calendar import exchange_dates

logger = get_logger(__name__)

# A fraction, not a percentage. Any stock split moves prices by far more
# (3-for-2, the smallest common ratio, is 50%); dividend drift on these
# tickers stays under about 2% a year.
DEFAULT_THRESHOLD = 0.15

# Consecutive compared dates that must disagree before the guard fails.
MIN_BREAK_RUN = 3

# A split day's own move compounds with the split, so the jump check's limits
# sit this far inside the smallest common split's price ratio. Wide on
# purpose: a missed split costs more than a false alarm.
JUMP_TOLERANCE = 0.10

# A stored close at or below FALL_LIMIT times the stored close before it, or
# at or above RISE_LIMIT times it, is a split-sized move. 3-for-2 and its
# reverse, 2-for-3, the smallest common splits, move the price to 2/3 and 3/2
# of the close before; JUMP_TOLERANCE moves each limit 10% toward no move at
# all. Everything beyond is flagged whatever its ratio: 7-for-1, 50-for-1,
# 1-for-8, a 2-for-1 on a day the stock also fell 20%. The price: any real
# one-day fall of 26.7% or more, or rise of 35% or more, fails the guard until
# it is acknowledged.
FALL_LIMIT = 2 / 3 * (1 + JUMP_TOLERANCE)  # 0.733
RISE_LIMIT = 3 / 2 * (1 - JUMP_TOLERANCE)  # 1.35

# Splits common enough to name, as new shares per old, with their reverses.
# The jump check flags by size; these only name the nearest split.
COMMON_SPLITS = (3 / 2, 2, 3, 4, 5, 7, 8, 10, 15, 20, 25, 50)

# How many dates a failure message lists for one ticker before it sums up.
LISTED_DATES = 5


class PriceBasisError(RuntimeError):
    """Stored prices show a split they were not adjusted for, or a
    split-sized move nothing has confirmed as real."""

    def __init__(self, message: str, since: Mapping[str, pd.Timestamp] | None = None):
        super().__init__(message)
        # Each failing ticker's first stored date, from which the command
        # lists Yahoo's split events. run() fetches none: the scheduler waits
        # on it before it can pause.
        self.since = dict(since or {})


class GuardError(RuntimeError):
    """At least one ticker could not be checked because of an error."""


class GuardMisconfigured(ValueError):
    """A split guard setting cannot be read: the threshold or the
    acknowledged list. Not an incomplete check: neither check can run until
    it is fixed, so the scheduler pauses as for a split."""


def configured_threshold(raw: str | None = None) -> float:
    """SPLIT_GUARD_THRESHOLD as a fraction between 0 and 1.

    Parsed here rather than when config is imported, so a typo such as "15%"
    fails the guard with a message naming the variable instead of taking every
    entry point down at import. Unset or blank means DEFAULT_THRESHOLD.
    """
    raw = config.SPLIT_GUARD_THRESHOLD if raw is None else raw
    if not raw.strip():
        return DEFAULT_THRESHOLD
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(
            f"SPLIT_GUARD_THRESHOLD must be a fraction such as 0.15, got {raw!r}"
        ) from None
    return _checked(value)


def _checked(threshold: float) -> float:
    if not 0 < threshold < 1:
        raise ValueError(
            f"SPLIT_GUARD_THRESHOLD must be a fraction between 0 and 1, "
            f"got {threshold}; use 0.15 for 15%"
        )
    return threshold


def compared_closes(
    stored: pd.DataFrame, reference: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """Stored and reference closes on every date both sides have.

    Both frames need date and close columns. Returns date, stored, yahoo,
    ratio (stored / yahoo), off (they differ by more than threshold) and run:
    how many consecutive compared dates share the row's off value. Reference
    closes that are missing or not positive are left out; a zero from Yahoo
    is a bad print, not a price.
    """
    reference = reference[reference["close"] > 0]
    merged = (
        stored[["date", "close"]]
        .merge(reference[["date", "close"]], on="date", suffixes=("_stored", "_yahoo"))
        .sort_values("date")
        .reset_index(drop=True)
    )
    ratio = merged["close_stored"] / merged["close_yahoo"]
    off = (ratio - 1).abs() > threshold
    run_id = (off != off.shift()).cumsum()
    run = off.groupby(run_id).transform("size")
    return merged.assign(ratio=ratio, off=off, run=run).rename(
        columns={"close_stored": "stored", "close_yahoo": "yahoo"}
    )


def mismatched_closes(
    stored: pd.DataFrame, reference: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """Dates where stored and reference closes differ by more than threshold.

    compared_closes' rows with off set: date, stored, yahoo, ratio and run,
    the length of the run of consecutive mismatched dates each belongs to.
    """
    compared = compared_closes(stored, reference, threshold)
    return compared[compared["off"]].drop(columns="off").reset_index(drop=True)


def find_basis_breaks(
    stored: pd.DataFrame,
    reference: pd.DataFrame,
    threshold: float,
    min_run: int = MIN_BREAK_RUN,
) -> pd.DataFrame:
    """The mismatched closes that belong to a run of at least min_run dates."""
    found = mismatched_closes(stored, reference, threshold)
    return found[found["run"] >= min_run].reset_index(drop=True)


def _split_name(shares: float) -> str:
    """A split given as new shares per old, as yfinance gives it, by name:
    10.0 is "10-for-1 split", 0.5 "1-for-2 reverse split"."""
    ratio = Fraction(shares).limit_denominator(1000)
    kind = "reverse split" if ratio < 1 else "split"
    return f"{ratio.numerator}-for-{ratio.denominator} {kind}"


def _nearest_split(price_ratio: float) -> str:
    """The common split whose price ratio is nearest, in log terms, to a
    close / previous close ratio, by name.

    Only a hint: a split day's own move can put it nearer another ratio.
    TSLA's 5-for-1 split of 2020-08-31 came on a day the stock rose 12.6%,
    0.2251 of the close before, which is nearer 4-for-1.
    """
    shares = min(
        (*COMMON_SPLITS, *(1 / split for split in COMMON_SPLITS)),
        key=lambda split: abs(math.log(price_ratio * split)),
    )
    return _split_name(shares)


def split_sized_jumps(stored: pd.DataFrame) -> pd.DataFrame:
    """Dates whose close is a split-sized move from the stored close before it.

    stored needs date and close columns, one row per trading date. Closes that
    are missing or not positive are left out, so a move is measured from the
    last usable close and a bad print the day before a split cannot hide it.
    Returns date, previous_date, previous_close, close, ratio (close /
    previous close) and nearest, the nearest common split by name (a hint).

    A one-day spike that reverses the next day is flagged twice rather than
    excused as a bad print, as the basis check excuses isolated mismatches:
    one row stored on the new basis among rows on the old one, such as a gap
    filled after a split, looks exactly the same.
    """
    frame = (
        stored.loc[stored["close"] > 0, ["date", "close"]]
        .sort_values("date")
        .reset_index(drop=True)
    )
    frame["previous_date"] = frame["date"].shift()
    frame["previous_close"] = frame["close"].shift()
    frame["ratio"] = frame["close"] / frame["previous_close"]
    found = frame[(frame["ratio"] <= FALL_LIMIT) | (frame["ratio"] >= RISE_LIMIT)]
    found = found.assign(nearest=[_nearest_split(ratio) for ratio in found["ratio"]])
    return found[
        ["date", "previous_date", "previous_close", "close", "ratio", "nearest"]
    ].reset_index(drop=True)


def yahoo_daily_closes(ticker: str, start: str) -> pd.DataFrame:
    """Yahoo's raw daily closes from start, as date and close columns."""
    data = download(ticker, start=start, interval="1d", auto_adjust=False)
    if data.empty:
        return pd.DataFrame(columns=["date", "close"])
    return pd.DataFrame(
        {
            "date": exchange_dates(data.index),
            "close": data["Close"].astype(float).to_numpy(),
        }
    )


def _dates(found: pd.DataFrame) -> str:
    days = [d.date().isoformat() for d in found["date"]]
    if len(days) <= LISTED_DATES:
        return ", ".join(days)
    return f"{days[0]} to {days[-1]}"


def _moves(jumps: pd.DataFrame) -> str:
    """Each flagged move of one ticker, for the failure message."""
    moves = [
        f"{day.date()}: {ratio:.4f}x, nearest {nearest}"
        + (", acknowledged but not confirmed by Yahoo" if reason is not None else "")
        for day, ratio, nearest, reason in zip(
            jumps["date"],
            jumps["ratio"],
            jumps["nearest"],
            jumps["acknowledged"],
            strict=True,
        )
    ]
    if len(moves) > LISTED_DATES:
        moves = [*moves[:LISTED_DATES], f"and {len(moves) - LISTED_DATES} more"]
    return "; ".join(moves)


def _split_events_text(ticker: str, since: pd.Timestamp) -> str:
    """Yahoo's split events for ticker since its first stored date, as one
    line of the command's failure output.

    They only help explain the failure, so nothing here may raise in its
    place. Yahoo can be late to list a split, so the line says what Yahoo
    lists, never that there was no split.
    """
    head = f"Yahoo's split events for {ticker}"
    try:
        events = split_events(ticker, since.date().isoformat())
        if events is None:
            return f"{head} could not be fetched"
        listed = [
            f"{_split_name(float(ratio))} on {day.date()}"
            for day, ratio in zip(exchange_dates(events.index), events, strict=True)
            if day >= since and ratio > 0
        ]
    except Exception as e:
        logger.warning(f"{ticker}: could not read Yahoo's split events: {e}")
        return f"{head} could not be read"
    return f"{head} since {since.date()}: " + (", ".join(listed) or "none listed")


@dataclass
class GuardResult:
    """What one pass over the tickers found."""

    # Each ticker with a basis break, and its mismatched closes.
    breaks: dict[str, pd.DataFrame] = field(default_factory=dict)
    # Each ticker with split-sized moves nothing silenced, and the moves.
    jumps: dict[str, pd.DataFrame] = field(default_factory=dict)
    # Each checked ticker's first stored date, where its split events start.
    first_dates: dict[str, pd.Timestamp] = field(default_factory=dict)
    # Tickers Yahoo returned no reference prices for, so not compared with it.
    unchecked: list[str] = field(default_factory=list)
    # Tickers whose check raised, and the error.
    errors: dict[str, Exception] = field(default_factory=dict)


def _report_unmatched(
    ticker: str, moves: pd.DataFrame, acknowledged: Mapping[tuple[str, date], str]
) -> None:
    """Warn about each acknowledged date of ticker that has no split-sized
    move: a typo in the date silences nothing, and the real move goes on
    failing with no other clue why."""
    flagged = {day.date() for day in moves["date"]}
    for listed, day in sorted(acknowledged):
        if listed == ticker and day not in flagged:
            logger.warning(
                f"{ticker}: the acknowledged move on {day} matches no split-sized "
                f"move in the stored closes; check its date in "
                f"{config.SPLIT_JUMP_ACKNOWLEDGED_FILE}"
            )


def _compare_with_yahoo(
    ticker: str, stored: pd.DataFrame, threshold: float
) -> pd.DataFrame | None:
    """compared_closes against Yahoo's raw closes, after logging what the
    comparison found, or None when Yahoo returned no reference prices."""
    reference = yahoo_daily_closes(ticker, stored["date"].min().date().isoformat())
    if reference.empty:
        logger.warning(f"{ticker}: no reference prices from Yahoo; not compared")
        return None

    compared = compared_closes(stored, reference, threshold)
    isolated = compared[compared["off"] & (compared["run"] < MIN_BREAK_RUN)]
    if not isolated.empty:
        logger.warning(
            f"{ticker}: stored close differs from Yahoo's by more than the "
            f"threshold on {len(isolated)} isolated dates ({_dates(isolated)}); "
            f"a bad print on one side rather than a split, so the comparison "
            f"does not fail on it"
        )
    if not (compared["off"] & (compared["run"] >= MIN_BREAK_RUN)).any():
        logger.info(f"{ticker}: stored prices agree with Yahoo")
    return compared


def _unconfirmed(ticker: str, moves: pd.DataFrame, agreed: set) -> pd.DataFrame:
    """moves less the acknowledged ones Yahoo confirms: its raw closes agree
    with the stored ones on the move date and the date before. Each
    acknowledged move is logged either way."""
    keep = []
    for day, previous, nearest, reason in zip(
        moves["date"],
        moves["previous_date"],
        moves["nearest"],
        moves["acknowledged"],
        strict=True,
    ):
        if reason is None:
            keep.append(True)
        elif day in agreed and previous in agreed:
            logger.info(
                f"{ticker}: the move on {day.date()} (nearest {nearest}) is "
                f"acknowledged and confirmed by Yahoo: {reason}"
            )
            keep.append(False)
        else:
            logger.warning(
                f"{ticker}: the move on {day.date()} is acknowledged ({reason}), "
                f"but Yahoo's raw closes do not confirm the stored ones on "
                f"{previous.date()} and {day.date()}, so it still fails"
            )
            keep.append(True)
    return moves[pd.Series(keep, index=moves.index, dtype=bool)].reset_index(drop=True)


def check(
    tickers: list[str] | None = None,
    threshold: float | None = None,
    acknowledged: Mapping[tuple[str, date], str] | None = None,
) -> GuardResult:
    """Check every ticker, whatever happens to the others.

    A ticker whose check raises (the database or Yahoo misbehaving) is logged
    and recorded, and the loop carries on: an error on one ticker used to
    escape the loop and discard a break already found for another.

    Split-sized moves are recorded before the comparison with Yahoo,
    acknowledged or not, and an acknowledged one is dropped only once that
    comparison confirms it. So a missing reference, or a comparison that
    raises, leaves every move standing.

    Raises GuardMisconfigured when the threshold or the acknowledged list
    cannot be read.
    """
    try:
        threshold = configured_threshold() if threshold is None else _checked(threshold)
        if acknowledged is None:
            acknowledged = config.split_jump_acknowledged()
    except ValueError as e:
        raise GuardMisconfigured(str(e)) from e

    result = GuardResult()
    for ticker in tickers or TICKERS:
        try:
            stored = load_daily_bars(ticker)
            if stored.empty:
                continue
            result.first_dates[ticker] = stored["date"].min()
            moves = split_sized_jumps(stored)
            moves["acknowledged"] = [
                acknowledged.get((ticker, day.date())) for day in moves["date"]
            ]
            _report_unmatched(ticker, moves, acknowledged)
            if not moves.empty:
                result.jumps[ticker] = moves
            compared = _compare_with_yahoo(ticker, stored, threshold)
        except Exception as e:
            logger.exception(f"{ticker}: could not be checked: {type(e).__name__}: {e}")
            result.errors[ticker] = e
            continue
        if compared is None:
            result.unchecked.append(ticker)
            continue

        breaks = compared[compared["off"] & (compared["run"] >= MIN_BREAK_RUN)]
        if not breaks.empty:
            result.breaks[ticker] = breaks.drop(columns="off").reset_index(drop=True)
        agreed = set(compared.loc[(compared["ratio"] - 1).abs() <= threshold, "date"])
        remaining = _unconfirmed(ticker, moves, agreed)
        if remaining.empty:
            result.jumps.pop(ticker, None)
        else:
            result.jumps[ticker] = remaining
    return result


def run(
    tickers: list[str] | None = None,
    threshold: float | None = None,
    acknowledged: Mapping[tuple[str, date], str] | None = None,
) -> list[str]:
    """Check every ticker, then raise if any shows a split or could not be checked.

    PriceBasisError, after logging what each check found, when any ticker has
    a basis break or a split-sized move nothing silenced, even if another
    ticker's check failed. Otherwise GuardError when any check raised.
    Returns the tickers Yahoo returned no reference prices for.

    Asks Yahoo for nothing beyond the checks themselves: the split events
    that explain a failure are fetched by the command (main), so the
    scheduler, which waits on this, can pause at once.
    """
    result = check(tickers, threshold, acknowledged)
    failed = ", ".join(sorted(result.errors))

    for ticker, found in result.breaks.items():
        logger.error(
            f"{ticker}: {len(found)} stored daily closes differ from Yahoo's by "
            f"more than the threshold, {found['date'].min().date()} to "
            f"{found['date'].max().date()}, median stored/Yahoo ratio "
            f"{found['ratio'].median():.4f}. A stock split after these rows were "
            f"stored is the usual cause: rescale them before the next run."
        )
    for ticker, jumps in result.jumps.items():
        moves = ", ".join(
            f"{day.date()} closed at {close:.2f} after {previous:.2f} on "
            f"{previous_day.date()}, {ratio:.4f} times it (nearest common split: "
            f"{nearest})"
            for day, close, previous, previous_day, ratio, nearest in zip(
                jumps["date"],
                jumps["close"],
                jumps["previous_close"],
                jumps["previous_date"],
                jumps["ratio"],
                jumps["nearest"],
                strict=True,
            )
        )
        logger.error(
            f"{ticker}: stored close moved by a split-sized amount in one day: "
            f"{moves}. If the rows before a move are on the old share basis, "
            f"rescale them by the ratio in Yahoo's split events, not by the "
            f"nearest common split named here; if the move was real, list its "
            f"date and the reason in {config.SPLIT_JUMP_ACKNOWLEDGED_FILE}."
        )

    failing = sorted(set(result.breaks) | set(result.jumps))
    if failing:
        parts = []
        if result.breaks:
            parts.append(
                "stored prices disagree with Yahoo for "
                + ", ".join(sorted(result.breaks))
            )
        if result.jumps:
            parts.append(
                "split-sized one-day moves in the stored closes for "
                + ", ".join(
                    f"{ticker} ({_moves(result.jumps[ticker])})"
                    for ticker in sorted(result.jumps)
                )
            )
        if result.unchecked:
            parts.append(
                "not compared with Yahoo (no reference prices): "
                + ", ".join(result.unchecked)
            )
        if failed:
            parts.append(f"could not check {failed}")
        raise PriceBasisError(
            "; ".join(parts),
            since={ticker: result.first_dates[ticker] for ticker in failing},
        )
    if result.errors:
        first = next(iter(result.errors.values()))
        raise GuardError(f"could not check {failed}") from first
    return result.unchecked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail if stored prices show a stock split they were not "
        "adjusted for"
    )
    parser.add_argument("--ticker", help="one ticker instead of config.TICKERS")
    args = parser.parse_args(argv)

    # Both read before any price, so a bad setting fails fast and by name.
    try:
        threshold = configured_threshold()
        acknowledged = config.split_jump_acknowledged()
    except ValueError as e:
        print(f"SPLIT GUARD MISCONFIGURED: {e}", file=sys.stderr)
        return 1
    try:
        unchecked = run(
            [args.ticker.upper()] if args.ticker else None, threshold, acknowledged
        )
    except PriceBasisError as e:
        print(f"SPLIT GUARD FAILED: {e}", file=sys.stderr)
        for ticker, since in e.since.items():
            print(f"  {_split_events_text(ticker, since)}", file=sys.stderr)
        return 1
    except GuardError as e:
        print(f"SPLIT GUARD INCOMPLETE: {e} (see the log)", file=sys.stderr)
        return 1
    # A ticker that could not be checked is not a pass: the workflow and
    # make all would carry on as if its prices were verified. The scheduler
    # calls run() instead; the price-basis gate in pipelines/scheduler.py
    # pauses its price jobs unless its last full pass is from the current or
    # previous trading day.
    if unchecked:
        print(
            "SPLIT GUARD INCOMPLETE: no reference prices from Yahoo for "
            + ", ".join(unchecked)
            + "; not compared with Yahoo",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
