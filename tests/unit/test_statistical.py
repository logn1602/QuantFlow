"""tests/unit/test_statistical.py

Covers the dates ARIMA and Prophet predict and are scored on.

Prophet's holdout predictions came from make_future_dataframe(freq="B"),
which steps through every weekday. After a market holiday inside the 30-day
window, each prediction was compared with the next trading day's close. The
live forecasts had the same weekday stepping, so they promised closes on
holidays.

No model is trained here: Prophet and ARIMA are replaced by fakes. The fake
Prophet is an oracle that predicts each date's true close, so a perfect model
scores zero error only if every prediction is compared with its own date.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest

from quantflow.models import ensemble, statistical
from quantflow.utils.trading_calendar import NYSEHolidayCalendar

TRADING_DAY = pd.offsets.CustomBusinessDay(calendar=NYSEHolidayCalendar())


def _series(start, end):
    days = pd.date_range(start, end, freq=TRADING_DAY)
    return pd.DataFrame({"ds": days, "y": 100.0 + np.arange(len(days), dtype=float)})


# The 30-day holdout of this series starts 2026-07-02 and has the 2026-07-03
# market holiday inside it.
HOLIDAY_IN_HOLDOUT = _series("2026-04-01", "2026-08-13")
# Ends on the Friday before Thanksgiving week.
BEFORE_THANKSGIVING = _series("2026-07-01", "2026-11-20")
NEXT_SEVEN_TRADING_DAYS = [
    "2026-11-23", "2026-11-24", "2026-11-25", "2026-11-27",
    "2026-11-30", "2026-12-01", "2026-12-02",
]  # fmt: skip


class FakeProphet:
    """Records its settings; predicts each date's true close when known."""

    truth: dict = {}  # noqa: RUF012 (set per test)
    instances: list = []  # noqa: RUF012

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.history = None
        FakeProphet.instances.append(self)

    def fit(self, df):
        self.history = df.copy()
        return self

    def make_future_dataframe(self, periods, freq="D", include_history=True):
        """Prophet's own implementation, so pre-fix code behaves as it did."""
        last = self.history["ds"].max()
        dates = pd.date_range(start=last, periods=periods + 1, freq=freq)
        dates = dates[dates > last][:periods]
        if include_history:
            dates = np.concatenate((np.array(self.history["ds"]), dates))
        return pd.DataFrame({"ds": dates})

    def predict(self, df):
        ds = pd.to_datetime(df["ds"])
        yhat = np.array([self.truth.get(d, 0.0) for d in ds])
        return pd.DataFrame(
            {"ds": ds, "yhat": yhat, "yhat_lower": yhat - 1, "yhat_upper": yhat + 1}
        )


@pytest.fixture
def fake_prophet(monkeypatch):
    FakeProphet.instances = []
    module = types.ModuleType("prophet")
    module.Prophet = FakeProphet
    monkeypatch.setitem(sys.modules, "prophet", module)
    monkeypatch.setattr(statistical, "_log_mlflow", lambda *args, **kwargs: None)
    return FakeProphet


def _oracle(prophet, frame):
    prophet.truth = dict(zip(frame["ds"], frame["y"], strict=True))


# ── Prophet holdout ───────────────────────────────────────────────────────────


def test_prophet_holdout_predictions_are_scored_against_their_own_dates(
    fake_prophet,
):
    _oracle(fake_prophet, HOLIDAY_IN_HOLDOUT)
    _, metrics = statistical.run_prophet(HOLIDAY_IN_HOLDOUT, "AAPL")
    assert metrics["mape"] == 0.0


def test_the_ensembles_prophet_column_lines_up_with_the_holdout(fake_prophet):
    _oracle(fake_prophet, HOLIDAY_IN_HOLDOUT)
    preds = ensemble._prophet_holdout(HOLIDAY_IN_HOLDOUT)
    assert np.allclose(preds, HOLIDAY_IN_HOLDOUT["y"].to_numpy()[-30:])


# ── Yearly seasonality needs two years ────────────────────────────────────────
#
# yearly_seasonality=True was forced on for every ticker. With under two years
# of history Prophet fits a yearly curve to a single season, which it would
# never do on its own: its 'auto' rule needs 730 days.

ONE_YEAR = _series("2025-10-01", "2026-10-01")
# Over two years even after the 30 holdout days are taken off the end.
TWO_YEARS = _series("2024-06-03", "2026-10-01")
# Just over two years in total, under two once the holdout is removed.
NEAR_TWO_YEARS = _series("2024-09-03", "2026-10-01")


def _yearly_flags(prophet):
    return [instance.kwargs["yearly_seasonality"] for instance in prophet.instances]


def test_prophet_skips_yearly_seasonality_with_under_two_years(fake_prophet):
    statistical.run_prophet(ONE_YEAR, "AAPL")
    assert _yearly_flags(fake_prophet) == [False, False]  # holdout fit, live fit


def test_prophet_keeps_yearly_seasonality_with_two_years(fake_prophet):
    statistical.run_prophet(TWO_YEARS, "AAPL")
    assert _yearly_flags(fake_prophet) == [True, True]


def test_near_two_years_only_the_live_fit_gets_yearly_seasonality(fake_prophet):
    """Deliberate: each fit is judged on the history it actually sees, and
    the holdout fit sees 30 fewer trading days than the live one."""
    statistical.run_prophet(NEAR_TWO_YEARS, "AAPL")
    assert _yearly_flags(fake_prophet) == [False, True]


def test_the_ensembles_prophet_fit_follows_the_same_rule(fake_prophet):
    ensemble._prophet_holdout(ONE_YEAR)
    assert _yearly_flags(fake_prophet) == [False]


# ── Live forecast dates ───────────────────────────────────────────────────────


def test_the_live_prophet_forecast_skips_market_holidays(fake_prophet):
    forecast, _ = statistical.run_prophet(BEFORE_THANKSGIVING, "AAPL")
    assert [d.isoformat() for d in forecast["ds"]] == NEXT_SEVEN_TRADING_DAYS


class _FakeForecast:
    def __init__(self, mean):
        self.predicted_mean = mean

    def conf_int(self, alpha=0.05):
        return np.column_stack([self.predicted_mean - 1, self.predicted_mean + 1])


class FakeARIMA:
    """Forecasts the last value flat; no fitting."""

    def __init__(self, endog, order):
        self.last = float(np.asarray(endog)[-1])

    def fit(self):
        return self

    def forecast(self, steps):
        return np.full(steps, self.last)

    def get_forecast(self, steps):
        return _FakeForecast(np.full(steps, self.last))


def test_the_live_arima_forecast_skips_market_holidays(monkeypatch):
    import statsmodels.tsa.arima.model as arima_module

    monkeypatch.setattr(arima_module, "ARIMA", FakeARIMA)
    monkeypatch.setattr(statistical, "_log_mlflow", lambda *args, **kwargs: None)

    forecast, _ = statistical.run_arima(BEFORE_THANKSGIVING, "AAPL")
    assert [d.isoformat() for d in forecast["ds"]] == NEXT_SEVEN_TRADING_DAYS
