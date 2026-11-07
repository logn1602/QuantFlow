"""US equity market time.

The one place that knows which time zone the exchange keeps, when a
session's daily bar or a 15-min bar can be treated as final, and which days
it trades.
"""

import datetime as dt

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)
from pandas.tseries.offsets import CustomBusinessDay

EXCHANGE_TZ = "America/New_York"

# The regular session closes at 16:00 New York and Yahoo's daily bar picks up
# the closing-auction price shortly after. An hour of margin before treating
# today's daily bar as final: stored any earlier, it is a snapshot of a session
# still trading, and with ON CONFLICT DO NOTHING it would never be corrected.
DAILY_BAR_FINAL_AFTER = dt.time(17, 0)

# A 15-min bar covers the quarter hour starting at its timestamp. Yahoo also
# serves the bar still trading, holding the latest price as its close and the
# volume so far, so a bar counts as final two minutes after its quarter hour
# ends, measured from the bar's own timestamp.
INTRADAY_BAR_FINAL_AFTER = dt.timedelta(minutes=15 + 2)


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """The NYSE's scheduled full-day holidays.

    Built from pandas' holiday rules rather than a calendar package so it
    adds no dependency. It covers the published schedule only: unscheduled
    closures (the 2025-01-09 national day of mourning, weather, emergencies)
    cannot be known in advance, so a forecast made just before one still
    lists that day. Early-close days are trading days and are not listed.
    """

    rules = [  # noqa: RUF012 (the pandas API is a mutable class attribute)
        # A Saturday New Year's Day is not moved to the Friday before: the
        # exchange stays open on December 31. A Sunday one moves to Monday.
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday(
            "Juneteenth",
            month=6,
            day=19,
            start_date="2022-01-01",
            observance=nearest_workday,
        ),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


TRADING_DAY = CustomBusinessDay(calendar=NYSEHolidayCalendar())


def exchange_dates(index) -> pd.DatetimeIndex:
    """The exchange calendar date of each timestamp, as naive midnights.

    yfinance returns daily bars as naive exchange-date midnights. An aware
    index (older versions, or a future change) is converted to New York time
    first, so every consumer agrees on which trading day a bar belongs to.
    """
    dates = pd.DatetimeIndex(index)
    if dates.tz is not None:
        dates = dates.tz_convert(EXCHANGE_TZ).tz_localize(None)
    return dates.normalize()


def next_trading_days(after, n: int) -> pd.DatetimeIndex:
    """The n NYSE trading days strictly after `after`, as naive midnights.

    after may be a date, a naive timestamp, or a tz-aware one; only its
    calendar date is used.
    """
    day = pd.Timestamp(after)
    if day.tzinfo is not None:
        day = day.tz_localize(None)
    start = day.normalize() + pd.Timedelta(days=1)
    return pd.date_range(start=start, periods=n, freq=TRADING_DAY)
