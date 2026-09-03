"""tests/unit/test_forecast_writer.py

Covers the date coercion in quantflow.db.forecasts.

This function is the reason three duplicate save_forecasts implementations
could be merged into one. Each carried its own coercion — one handled
pd.Timestamp and numpy datetime64, one assumed .date() always existed, one
guarded against strings — and the retained version is the union. If it ever
narrows, one of the three original call sites starts writing a wrong or
unparseable forecast_date, and Postgres will accept plausible-looking garbage
rather than failing loudly.
"""

import datetime as dt

import numpy as np
import pandas as pd

from quantflow.db.forecasts import _as_date

EXPECTED = dt.date(2026, 3, 14)


# ── Every type the three original call sites could pass ───────────────────────


def test_pandas_timestamp_becomes_a_date():
    """ARIMA and Prophet build forecast dates with pd.bdate_range."""
    assert _as_date(pd.Timestamp("2026-03-14 16:30:00")) == EXPECTED


def test_numpy_datetime64_becomes_a_date():
    assert _as_date(pd.Timestamp(np.datetime64("2026-03-14"))) == EXPECTED


def test_python_datetime_becomes_a_date():
    assert _as_date(dt.datetime(2026, 3, 14, 16, 30)) == EXPECTED


def test_a_plain_date_passes_through_unchanged():
    """datetime.date has no .date() method. The implementation that assumed it
    did would raise here."""
    assert _as_date(EXPECTED) == EXPECTED


def test_a_string_is_left_for_postgres_to_parse():
    """Matches the previous ensemble behaviour, which explicitly excluded
    strings from coercion."""
    assert _as_date("2026-03-14") == "2026-03-14"


# ── Properties ────────────────────────────────────────────────────────────────


def test_time_of_day_is_discarded_not_rounded():
    """A late-evening timestamp must land on its own calendar day, not the
    next one."""
    assert _as_date(pd.Timestamp("2026-03-14 23:59:59")) == EXPECTED


def test_coercion_is_idempotent():
    once = _as_date(pd.Timestamp("2026-03-14 16:30:00"))
    assert _as_date(once) == once


def test_every_input_form_agrees():
    """The whole point of the union: all of these describe the same day and
    must produce the same stored value."""
    forms = [
        pd.Timestamp("2026-03-14"),
        pd.Timestamp("2026-03-14 09:15:00"),
        dt.datetime(2026, 3, 14, 9, 15),
        EXPECTED,
    ]
    assert {_as_date(f) for f in forms} == {EXPECTED}
