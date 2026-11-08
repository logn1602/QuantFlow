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
from quantflow.utils.trading_calendar import TRADING_DAY


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

    def predict(self, df):
        ds = pd.to_datetime(df["ds"])
        yhat = np.array([self.truth.get(d, 0.0) for d in ds])
        return pd.DataFrame(
            {"ds": ds, "yhat": yhat, "yhat_lower": yhat - 1, "yhat_upper": yhat + 1}
        )


@pytest.fixture
def fake_prophet(monkeypatch):
    FakeProphet.instances = []
    FakeProphet.truth = {}  # class state: one test's oracle must not reach the next
    module = types.ModuleType("prophet")
    module.Prophet = FakeProphet
    monkeypatch.setitem(sys.modules, "prophet", module)
    monkeypatch.setattr(statistical, "_log_mlflow", lambda *args, **kwargs: None)
    return FakeProphet


def _oracle(prophet, frame):
    prophet.truth = dict(zip(frame["ds"], frame["y"], strict=True))


class _FakeForecast:
    def __init__(self, mean):
        self.predicted_mean = mean

    def conf_int(self, alpha=0.05):
        return np.column_stack([self.predicted_mean - 1, self.predicted_mean + 1])


class FakeARIMA:
    """Forecasts the last value flat; no fitting. Records each order."""

    orders: list = []  # noqa: RUF012 (reset per test by the fixture)

    def __init__(self, endog, order):
        self.last = float(np.asarray(endog)[-1])
        FakeARIMA.orders.append(tuple(order))

    def fit(self):
        return self

    def forecast(self, steps):
        return np.full(steps, self.last)

    def get_forecast(self, steps):
        return _FakeForecast(np.full(steps, self.last))


@pytest.fixture
def fake_arima(monkeypatch):
    import statsmodels.tsa.arima.model as arima_module

    FakeARIMA.orders = []
    monkeypatch.setattr(arima_module, "ARIMA", FakeARIMA)
    return FakeARIMA


@pytest.fixture
def no_boosters(monkeypatch):
    """The stack's booster columns, without training any booster."""
    monkeypatch.setattr(
        ensemble,
        "_xgb_lgb_holdout",
        lambda *args, **kwargs: {"xgboost": np.zeros(30), "lightgbm": np.zeros(30)},
    )


# ── Prophet holdout ───────────────────────────────────────────────────────────


def test_prophet_holdout_predictions_are_scored_against_their_own_dates(
    fake_prophet,
):
    _oracle(fake_prophet, HOLIDAY_IN_HOLDOUT)
    _, metrics = statistical.run_prophet(HOLIDAY_IN_HOLDOUT, "AAPL")
    assert metrics["mape"] == 0.0


def test_the_ensembles_prophet_column_lines_up_with_the_holdout(
    fake_prophet, fake_arima, no_boosters
):
    _oracle(fake_prophet, HOLIDAY_IN_HOLDOUT)
    stack = ensemble.build_holdout_stack(HOLIDAY_IN_HOLDOUT, pd.DataFrame(), "AAPL")
    assert np.allclose(stack["prophet"], HOLIDAY_IN_HOLDOUT["y"].to_numpy()[-30:])
    assert np.allclose(stack["actual"], HOLIDAY_IN_HOLDOUT["y"].to_numpy()[-30:])


# ── One ARIMA definition ──────────────────────────────────────────────────────
#
# The ensemble's stack fitted its own ARIMA with the order written out as
# (5, 1, 0), so changing ARIMA_ORDER would have changed the scored and live
# ARIMA but not the column the meta-learner is weighted on.


def test_the_ensembles_stack_fits_the_configured_arima_order(
    monkeypatch, fake_prophet, fake_arima, no_boosters
):
    monkeypatch.setattr(statistical, "ARIMA_ORDER", (2, 1, 0))
    monkeypatch.setattr(ensemble, "load_prices", lambda ticker: HOLIDAY_IN_HOLDOUT)
    monkeypatch.setattr(
        ensemble, "load_features", lambda ticker, news=None: pd.DataFrame()
    )
    ensemble.collect_holdout_stacks("AAPL")
    assert fake_arima.orders == [(2, 1, 0)]


def test_the_scored_and_live_arima_fits_use_the_configured_order(
    monkeypatch, fake_arima
):
    monkeypatch.setattr(statistical, "ARIMA_ORDER", (2, 1, 0))
    monkeypatch.setattr(statistical, "_log_mlflow", lambda *args, **kwargs: None)
    statistical.run_arima(HOLIDAY_IN_HOLDOUT, "AAPL")
    assert fake_arima.orders == [(2, 1, 0), (2, 1, 0)]  # holdout fit, live fit


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


def test_the_ensembles_prophet_fit_follows_the_same_rule(
    fake_prophet, fake_arima, no_boosters
):
    ensemble.build_holdout_stack(ONE_YEAR, pd.DataFrame(), "AAPL")
    assert _yearly_flags(fake_prophet) == [False]


# ── Live forecast dates ───────────────────────────────────────────────────────


def test_the_live_prophet_forecast_skips_market_holidays(fake_prophet):
    forecast, _ = statistical.run_prophet(BEFORE_THANKSGIVING, "AAPL")
    assert [d.isoformat() for d in forecast["ds"]] == NEXT_SEVEN_TRADING_DAYS


def test_the_live_arima_forecast_skips_market_holidays(monkeypatch, fake_arima):
    monkeypatch.setattr(statistical, "_log_mlflow", lambda *args, **kwargs: None)

    forecast, _ = statistical.run_arima(BEFORE_THANKSGIVING, "AAPL")
    assert [d.isoformat() for d in forecast["ds"]] == NEXT_SEVEN_TRADING_DAYS


class SamplingProphet(FakeProphet):
    """Draws its band from numpy's global random state, as Prophet's
    uncertainty sampling does (np.random.normal, np.random.laplace)."""

    def predict(self, df):
        out = super().predict(df)
        noise = np.random.normal(0, 1, len(out))
        return out.assign(
            yhat_lower=out["yhat"] - 2 + noise, yhat_upper=out["yhat"] + 2 + noise
        )


def test_the_same_history_gives_the_same_prophet_band(monkeypatch):
    """Prophet simulates its interval from numpy's global random state, so
    the same fit gave a different band on every run, and the evaluation's
    worker processes could not reproduce a sequential run. The band is
    drawn from a fixed seed, and the caller's random state is left as it
    was."""
    module = types.ModuleType("prophet")
    module.Prophet = SamplingProphet
    monkeypatch.setitem(sys.modules, "prophet", module)
    dates = pd.to_datetime(NEXT_SEVEN_TRADING_DAYS)

    np.random.seed(1)
    expected = np.random.random()  # the caller's next draw, with no fit between

    np.random.seed(1)
    first = statistical.prophet_forecast(BEFORE_THANKSGIVING, dates)
    after = np.random.random()
    np.random.seed(2)
    second = statistical.prophet_forecast(BEFORE_THANKSGIVING, dates)

    pd.testing.assert_frame_equal(first, second)
    assert after == expected
