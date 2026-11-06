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
import pytest
from sqlalchemy.exc import DataError

from quantflow.db.forecasts import _as_date, save_forecasts

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


# ── save_forecasts ────────────────────────────────────────────────────────────
#
# It used to INSERT row by row and catch each failure, inside one transaction.
# On Postgres the first failure aborts the transaction, so the "skipped" row
# took every later row with it and the commit rolled the batch back.


def _forecast(n=7, model="arima"):
    days = pd.bdate_range("2026-11-09", periods=n)
    return pd.DataFrame(
        {
            "ticker": "AAPL",
            "model": model,
            "ds": days,
            "predicted_close": np.linspace(100, 101, n),
            "lower_bound": np.linspace(98, 99, n),
            "upper_bound": np.linspace(102, 103, n),
        }
    )


def test_a_forecast_is_written_as_one_batched_insert(fake_db):
    assert save_forecasts(_forecast()) == 7
    assert len(fake_db.inserts("forecasts")) == 1


def test_a_forecast_write_error_raises_instead_of_skipping_the_row(fake_db):
    fake_db.fail_next_insert = DataError("INSERT", {}, Exception("numeric overflow"))
    with pytest.raises(DataError):
        save_forecasts(_forecast())
    assert fake_db.tables["forecasts"] == []


def test_stored_forecast_rows_keep_their_coercion_and_rounding(fake_db):
    save_forecasts(_forecast(n=1))
    (row,) = fake_db.tables["forecasts"]
    assert row["forecast_date"] == dt.date(2026, 11, 9)
    assert row["predicted_close"] == 100.0
    assert isinstance(row["run_at"], dt.datetime)
