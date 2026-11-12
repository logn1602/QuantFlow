"""Unscheduled closures: days the exchange's published schedule lists as
trading days on which it did not open (decision D22, docs/evaluation.md).

The project calendar (utils.trading_calendar) knows the schedule only. A
closure it cannot know leaves a gap in the data where the calendar expects a
session, and the harness refuses an origin whose forecast days cross one
(rolling_origin.CalendarMismatchError), so without a rule one such day would
make the forward test unrunnable as registered. The rule: a closed day is not
a trading day. Listed in config/unscheduled_closures.txt once it has
happened, with its source, it is counted past; a gap nobody listed is still
refused, so a missing bar is never taken for a closure.

Only days from the forward window's first origin on are taken: the windows
before it are scored as registered, 2025-01-09 included (the tuning window
leaves out the origins whose forecast days cross it).
"""

import re
from datetime import UTC, date, datetime
from functools import cache
from pathlib import Path

import pandas as pd

from quantflow.evaluation.splits import FIRST_FORWARD_ORIGIN
from quantflow.utils.trading_calendar import TRADING_DAY

CLOSURES_FILE = (
    Path(__file__).resolve().parents[3] / "config" / "unscheduled_closures.txt"
)

_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


def read(
    path: Path | None = None, today: date | None = None
) -> tuple[pd.Timestamp, ...]:
    """The listed closures, oldest first, as naive midnights.

    Blank lines and lines starting with # are skipped; every other line must
    read YYYY-MM-DD followed by a # comment giving the source. A day must be
    one the schedule lists as a trading day (a weekend or a holiday is not a
    closure), on or after FIRST_FORWARD_ORIGIN, and not after today: only a
    closure that has happened can be listed. As with the split guard's list,
    lines end at a newline only, and a control or invisible character other
    than a tab is refused, so nothing reads differently in review.

    Raises ValueError naming the file, and the line where there is one, for
    a file that is missing, unreadable or not UTF-8, and for a malformed,
    repeated or refused entry: the harness then stops rather than score on a
    list it misread. A missing file is not an empty list, since the file is
    committed with the code.
    """
    path = CLOSURES_FILE if path is None else path
    # UTC's date is never behind New York's, so this refuses only days that
    # have not begun anywhere the exchange is.
    today = datetime.now(UTC).date() if today is None else today
    try:
        # utf-8-sig: Notepad and other Windows editors may save with a BOM.
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise ValueError(
            f"{path} not found; it is committed with the code, and a list with "
            f"no entries is a file with only comments"
        ) from None
    except UnicodeDecodeError as e:
        raise ValueError(
            f"{path} is not UTF-8 text (byte {e.start}); save it as UTF-8"
        ) from None
    except OSError as e:
        raise ValueError(f"{path} cannot be read: {e}") from None

    first_forward = FIRST_FORWARD_ORIGIN.date()
    found: dict[date, int] = {}
    for number, raw in enumerate(text.split("\n"), start=1):
        where = f"{path}, line {number}"
        line = raw.removesuffix("\r")
        if not line.replace("\t", " ").isprintable():
            raise ValueError(
                f"{where}: contains a control or invisible character; the list "
                f"must be plain text"
            )
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entry, hash_mark, source = line.partition("#")
        if not hash_mark or not source.strip():
            raise ValueError(
                f"{where}: give the source after a #, as in "
                f"'2027-03-10  # the exchange's notice of the closure'"
            )
        entry = entry.strip()
        if not _DAY.fullmatch(entry):
            raise ValueError(
                f"{where}: expected YYYY-MM-DD before the #, got {entry!r}"
            )
        try:
            day = date.fromisoformat(entry)
        except ValueError:
            raise ValueError(f"{where}: {entry} is not a date") from None
        if day > today:
            raise ValueError(
                f"{where}: {day} is in the future; only a closure that has "
                f"happened can be listed"
            )
        if day < first_forward:
            raise ValueError(
                f"{where}: {day} is before {first_forward}, the forward window's "
                f"first origin; the earlier windows are scored as registered"
            )
        if not TRADING_DAY.is_on_offset(pd.Timestamp(day)):
            raise ValueError(
                f"{where}: {day} is not a trading day on the published schedule "
                f"(a weekend or a holiday), so it is not an unscheduled closure"
            )
        if day in found:
            raise ValueError(f"{where}: repeats line {found[day]}")
        found[day] = number
    return tuple(pd.Timestamp(day) for day in sorted(found))


@cache
def listed() -> tuple[pd.Timestamp, ...]:
    """The committed list, read once per process."""
    return read()
