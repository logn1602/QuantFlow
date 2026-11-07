"""tests/unit/test_trading_calendar.py

Covers the NYSE calendar that dates every live forecast.

Forecast dates used to step through weekdays (pd.bdate_range, freq="B"), so a
7-day forecast made the week of Thanksgiving promised a close for a day the
market is shut. The holiday lists below are the exchange's published
full-day closures.
"""

import pandas as pd
import pytest

from quantflow.utils.trading_calendar import (
    NYSEHolidayCalendar,
    last_completed_session,
    next_trading_days,
    previous_trading_day,
)

NYSE_2025 = [
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
]  # fmt: skip
NYSE_2026 = [
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
]  # fmt: skip
NYSE_2027 = [
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
]  # fmt: skip


@pytest.mark.parametrize(
    "year, expected", [(2025, NYSE_2025), (2026, NYSE_2026), (2027, NYSE_2027)]
)
def test_holidays_match_the_published_nyse_schedule(year, expected):
    holidays = NYSEHolidayCalendar().holidays(f"{year}-01-01", f"{year}-12-31")
    assert [d.date().isoformat() for d in holidays] == expected


def test_a_saturday_new_years_day_is_not_observed_on_the_friday_before():
    """The exchange stays open on December 31 when January 1 is a Saturday,
    unlike every other Saturday holiday."""
    holidays = NYSEHolidayCalendar().holidays("2021-12-01", "2022-01-31")
    days = [d.date().isoformat() for d in holidays]
    assert "2021-12-31" not in days
    assert "2021-12-24" in days  # Christmas on a Saturday is observed Friday


def test_next_trading_days_skip_thanksgiving():
    got = next_trading_days(pd.Timestamp("2026-11-24"), 7)
    assert [d.date().isoformat() for d in got] == [
        "2026-11-25", "2026-11-27", "2026-11-30", "2026-12-01",
        "2026-12-02", "2026-12-03", "2026-12-04",
    ]  # fmt: skip


def test_next_trading_days_start_strictly_after_the_given_day():
    got = next_trading_days(pd.Timestamp("2026-07-02"), 2)  # July 3 is a holiday
    assert [d.date().isoformat() for d in got] == ["2026-07-06", "2026-07-07"]


def test_next_trading_days_accept_dates_and_tz_aware_timestamps():
    import datetime as dt

    from_date = next_trading_days(dt.date(2026, 11, 6), 1)
    from_aware = next_trading_days(pd.Timestamp("2026-11-06 20:00", tz="UTC"), 1)
    assert list(from_date) == list(from_aware) == [pd.Timestamp("2026-11-09")]


@pytest.mark.parametrize(
    "day",
    [
        "2025-10-13",  # Columbus Day
        "2025-11-11",  # Veterans Day
        "2026-10-12",
        "2026-11-11",
    ],
)
def test_federal_holidays_the_exchange_keeps_open_are_trading_days(day):
    """The calendar follows the NYSE, not the federal list: Good Friday and
    Juneteenth are closed (in the lists above), Columbus and Veterans Day open."""
    before = pd.Timestamp(day) - pd.Timedelta(days=1)
    assert next_trading_days(before, 1)[0] == pd.Timestamp(day)


def test_one_off_closures_are_not_covered():
    """The exchange also closes for days no rule predicts, such as the national
    day of mourning on 2025-01-09. The calendar cannot know those in advance,
    so a forecast made just before one still lists that day."""
    holidays = NYSEHolidayCalendar().holidays("2025-01-01", "2025-01-31")
    assert pd.Timestamp("2025-01-09") not in holidays


@pytest.mark.parametrize(
    "day, expected",
    [
        ("2026-11-10", "2026-11-09"),  # a Tuesday: the Monday before
        ("2026-11-09", "2026-11-06"),  # a Monday: the Friday before
        ("2026-11-07", "2026-11-06"),  # a Saturday
        ("2026-11-27", "2026-11-25"),  # the day after Thanksgiving
        ("2026-09-08", "2026-09-04"),  # the day after Labor Day
        ("2026-01-02", "2025-12-31"),  # the day after New Year's Day
    ],
)
def test_the_previous_trading_day_skips_weekends_and_holidays(day, expected):
    assert previous_trading_day(pd.Timestamp(day)) == pd.Timestamp(expected)


def test_the_previous_trading_day_accepts_a_tz_aware_timestamp():
    """Only the calendar date counts, as for next_trading_days."""
    got = previous_trading_day(pd.Timestamp("2026-11-09 20:00", tz="UTC"))
    assert got == pd.Timestamp("2026-11-06")


@pytest.mark.parametrize(
    "now, expected",
    [
        ("2026-11-06 18:00", "2026-11-06"),  # Friday, after the bar is final
        ("2026-11-06 16:30", "2026-11-05"),  # Friday, closed but not yet final
        ("2026-11-07 12:00", "2026-11-06"),  # Saturday
        ("2026-11-09 09:00", "2026-11-06"),  # Monday, before the open
        ("2026-11-26 20:00", "2026-11-25"),  # Thanksgiving evening
    ],
)
def test_the_last_completed_session_is_the_latest_final_daily_bar(now, expected):
    """Today counts once DAILY_BAR_FINAL_AFTER (17:00 New York) has passed."""
    at = pd.Timestamp(now, tz="America/New_York")
    assert last_completed_session(at) == pd.Timestamp(expected)


def test_a_naive_time_for_the_last_completed_session_is_read_as_utc():
    # 23:30 UTC is 18:30 in New York in November: Friday's bar is final.
    assert last_completed_session(pd.Timestamp("2026-11-06 23:30")) == pd.Timestamp(
        "2026-11-06"
    )
