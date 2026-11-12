"""tests/unit/test_closures.py

The list of unscheduled closures (decision D22): days the NYSE's schedule
listed as trading days on which it did not open. Listed, the harness counts
past them; anything it cannot read stops the harness, as the split guard's
list does.
"""

import datetime as dt

import pandas as pd
import pytest

from quantflow.evaluation import closures

LATER = dt.date(2027, 12, 31)


def _read(tmp_path, text, today=LATER, encoding="utf-8"):
    path = tmp_path / "unscheduled_closures.txt"
    path.write_bytes(text.encode(encoding))
    return closures.read(path, today=today)


def test_the_committed_list_reads_and_lists_no_closure_yet():
    assert closures.read() == ()


def test_a_closure_that_happened_is_read_with_its_source(tmp_path):
    text = "# header\n\n2027-03-10  # the exchange's notice\n2026-12-01\t# another\n"
    found = _read(tmp_path, text, today=dt.date(2027, 3, 10))
    assert found == (pd.Timestamp("2026-12-01"), pd.Timestamp("2027-03-10"))


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("2027-03-10", "give the source"),
        ("2027-03-10  #   ", "give the source"),
        ("10/03/2027  # notice", "expected YYYY-MM-DD"),
        ("2027-02-30  # notice", "is not a date"),
        ("2027-03-13  # a Saturday", "not a trading day"),
        ("2027-07-05  # Independence Day, observed", "not a trading day"),
        ("2025-01-09  # a national day of mourning", "before 2026-11-09"),
    ],
)
def test_an_entry_it_cannot_take_is_refused_with_its_line(tmp_path, line, message):
    with pytest.raises(ValueError, match=message) as raised:
        _read(tmp_path, f"# header\n{line}\n")
    assert "line 2" in str(raised.value)


def test_a_closure_still_to_come_is_refused(tmp_path):
    with pytest.raises(ValueError, match="in the future"):
        _read(tmp_path, "2027-03-10  # notice\n", today=dt.date(2027, 3, 9))


def test_a_repeated_day_is_refused(tmp_path):
    with pytest.raises(ValueError, match="repeats line 1"):
        _read(tmp_path, "2027-03-10  # notice\n2027-03-10  # again\n")


def test_a_hidden_character_is_refused(tmp_path):
    """A form feed would end the line for splitlines() but not for git."""
    with pytest.raises(ValueError, match="control or invisible"):
        _read(tmp_path, "#\x0c2027-03-10  # notice\n")


def test_a_missing_or_unreadable_file_is_not_an_empty_list(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        closures.read(tmp_path / "missing.txt")
    with pytest.raises(ValueError, match="not UTF-8"):
        _read(tmp_path, "2027-03-10  # caf\xe9\n", encoding="latin-1")


def test_a_bom_from_a_windows_editor_is_read(tmp_path):
    found = _read(tmp_path, "2027-03-10  # notice\n", encoding="utf-8-sig")
    assert found == (pd.Timestamp("2027-03-10"),)


def test_the_harness_counts_past_a_listed_closure_and_refuses_an_unlisted_gap(
    monkeypatch,
):
    """2027-03-05's seven forecast days cross 2027-03-10; with that day
    missing from the data, the origin is refused until it is listed."""
    from quantflow.evaluation import rolling_origin as ro
    from quantflow.utils.trading_calendar import TRADING_DAY

    days = pd.date_range("2026-11-09", "2027-04-30", freq=TRADING_DAY)
    dates = days[days != pd.Timestamp("2027-03-10")]
    origin = dates.get_loc(pd.Timestamp("2027-03-05"))
    monkeypatch.setattr(closures, "listed", lambda: ())
    assert ro.calendar_mismatched(dates, [origin]) == [origin]
    monkeypatch.setattr(closures, "listed", lambda: (pd.Timestamp("2027-03-10"),))
    assert ro.calendar_mismatched(dates, [origin]) == []
