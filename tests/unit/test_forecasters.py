"""tests/unit/test_forecasters.py

Covers the model adapters the evaluation calls. The central check: on the
same history, the booster adapter forecasts exactly what the live run
(boosting.run) saves. No model is trained; the regressors, ARIMA and
Prophet are fakes.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest

from quantflow.evaluation.rolling_origin import OriginData
from quantflow.features.engineering import build_features, get_feature_cols
from quantflow.models import boosting, ensemble, forecasters
from quantflow.utils.trading_calendar import TRADING_DAY, next_trading_days

DATES = pd.date_range("2025-10-01", periods=120, freq=TRADING_DAY)


def _prices(seed=3):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, len(DATES))))
    return pd.DataFrame(
        {"date": DATES, "open": close * 0.999, "high": close * 1.01,
         "low": close * 0.99, "close": close, "volume": 1e6 + np.arange(len(DATES))}
    )  # fmt: skip


def _data(prices):
    origin = prices["date"].iloc[-1]
    return OriginData(
        "AAPL", origin, prices, pd.DataFrame(), next_trading_days(origin, 7)
    )


class FakeRegressor:
    """Deterministic and instant: the training mean, moved by yesterday's
    close relative to its training mean, in the target's own scale (a price
    or a log return, boosting.TARGETS)."""

    def fit(self, X, y, **kwargs):
        lag = X["close_lag_1"]
        self.mean_y = float(np.mean(y))
        self.mean_lag = float(lag.mean())
        self.slope = 0.9 * float(np.std(y)) / float(np.std(lag))
        self.feature_importances_ = np.full(X.shape[1], 1.0 / X.shape[1])
        return self

    def predict(self, X):
        return self.mean_y + self.slope * (
            X["close_lag_1"].to_numpy(dtype=float) - self.mean_lag
        )


@pytest.fixture
def fake_regressors(monkeypatch):
    monkeypatch.setattr(
        boosting, "_new_regressor", lambda name, params=None: FakeRegressor()
    )


def test_the_booster_adapter_forecasts_what_the_live_run_saves(
    monkeypatch, fake_regressors
):
    prices = _prices()
    saved = {}
    monkeypatch.setattr(boosting, "load_news", lambda: None)
    monkeypatch.setattr(
        boosting,
        "load_features",
        lambda t, news: build_features(prices, pd.DataFrame(), t),
    )
    monkeypatch.setattr(
        boosting,
        "save_forecasts",
        lambda df: saved.setdefault(df["model"].iloc[0], df) is None,
    )
    monkeypatch.setattr(boosting, "log_mlflow", lambda *a, **k: None)
    monkeypatch.setattr(boosting, "save_model_metrics", lambda *a, **k: None)
    boosting.run(["AAPL"])

    for name in ("xgboost", "lightgbm"):
        adapted = forecasters.booster(name)(_data(prices))
        live = saved[name]
        assert np.allclose(adapted["yhat"], live["predicted_close"])
        assert np.allclose(adapted["yhat_lower"], live["lower_bound"])
        assert [pd.Timestamp(d) for d in live["ds"]] == list(_data(prices).dates)


class RecordingRegressor(FakeRegressor):
    def predict(self, X):
        self.last = X.copy()
        return super().predict(X)


def test_the_booster_adapter_passes_missing_values_on(monkeypatch):
    model = RecordingRegressor()
    monkeypatch.setattr(boosting, "_new_regressor", lambda name, params=None: model)
    forecasters.booster("xgboost")(_data(_prices()))
    # No articles: every news feature is missing, not 0.
    assert model.last[["has_news", "sentiment_compound"]].isna().all().all()


def test_the_booster_adapter_reads_only_the_history_it_is_given(fake_regressors):
    prices = _prices()
    full = forecasters.booster("xgboost")(_data(prices))
    changed = prices.copy()
    changed.loc[changed.index[-1], "close"] *= (
        1.05  # the origin's own close moves the forecast
    )
    assert not np.allclose(
        forecasters.booster("xgboost")(_data(changed))["yhat"], full["yhat"]
    )


class FakeForecast:
    def __init__(self, mean):
        self.predicted_mean = mean

    def conf_int(self, alpha=0.05):
        return np.column_stack([self.predicted_mean - 2, self.predicted_mean + 2])


class FakeARIMA:
    aic = 0.0  # every order alike, when the order is chosen by AIC

    def __init__(self, endog, order):
        self.last = float(np.asarray(endog)[-1])

    def fit(self):
        return self

    def get_forecast(self, steps):
        return FakeForecast(np.full(steps, self.last))


def test_the_arima_adapter_forecasts_from_the_closes_up_to_the_origin(monkeypatch):
    import statsmodels.tsa.arima.model as arima_module

    monkeypatch.setattr(arima_module, "ARIMA", FakeARIMA)
    prices = _prices()
    out = forecasters.arima(_data(prices))
    assert len(out) == 7 and np.allclose(out["yhat"], prices["close"].iloc[-1])
    assert np.allclose(out["yhat_upper"] - out["yhat_lower"], 4.0)
    assert out["arima_order"].nunique() == 1  # the order travels with the cells


class FakeProphet:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, df):
        return self

    def predict(self, df):
        ds = pd.to_datetime(df["ds"])
        return pd.DataFrame(
            {"ds": ds, "yhat": 1.0, "yhat_lower": 0.5, "yhat_upper": 1.5}
        )


def test_the_prophet_adapter_predicts_the_dates_asked_and_reports_its_yearly_flag(
    monkeypatch,
):
    module = types.ModuleType("prophet")
    module.Prophet = FakeProphet
    monkeypatch.setitem(sys.modules, "prophet", module)
    out = forecasters.prophet(_data(_prices()))
    assert out["yhat"].tolist() == [1.0] * 7
    assert out["yearly_seasonality"].eq(False).all()  # 120 days: under two years


def _base(value, width=1.0):
    return pd.DataFrame({"yhat": np.full(7, value), "yhat_lower": value - width,
                         "yhat_upper": value + width})  # fmt: skip


def test_the_ensemble_adapter_weights_the_base_forecasts_by_the_holdout_stack(
    monkeypatch,
):
    actual = np.linspace(100, 110, 30)
    stack = pd.DataFrame({"actual": actual, "arima": actual, "prophet": actual + 5,
                          "xgboost": actual + 3, "lightgbm": actual + 9,
                          "naive": actual + 7})  # fmt: skip
    monkeypatch.setattr(forecasters, "build_holdout_stack", lambda *a: stack)
    base = {"arima": _base(200.0), "prophet": _base(210.0), "xgboost": _base(190.0),
            "lightgbm": _base(230.0), "naive": _base(220.0)}  # fmt: skip
    out = forecasters.ensemble_stack(_data(_prices()), base, pd.DataFrame())
    assert np.allclose(out["yhat"], 200.0)  # all weight on the perfect arima column
    assert np.allclose(out["yhat_upper"] - out["yhat_lower"], 2.0)


def test_the_ensemble_adapter_refuses_a_missing_base_forecast():
    with pytest.raises(RuntimeError, match="prophet"):
        forecasters.ensemble_stack(
            _data(_prices()),
            {m: _base(1.0) for m in ensemble.BASE_MODELS if m != "prophet"},
            pd.DataFrame(),
        )


def test_every_model_and_baseline_is_registered():
    assert set(forecasters.FORECASTERS) == {"naive", "drift", *ensemble.BASE_MODELS}
    assert set(forecasters.COMBINERS) == {"ensemble_stack", "ensemble_rolling"}
    assert set(forecasters.VARIANTS) == {"xgboost_no_news", "lightgbm_no_news"}


def test_the_ablation_boosters_fit_without_the_news_features(monkeypatch):
    model = RecordingRegressor()
    monkeypatch.setattr(boosting, "_new_regressor", lambda name, params=None: model)
    forecasters.VARIANTS["xgboost_no_news"](_data(_prices()))
    assert not set(forecasters.NEWS_FEATURES) & set(model.last.columns)
    assert "close_lag_1" in model.last.columns
    assert set(forecasters.NEWS_FEATURES) <= set(get_feature_cols())


def test_naives_band_in_the_ensemble_is_its_forecast(monkeypatch):
    """Naive has no band; as live, its bounds are its forecast, not 0."""
    actual = np.linspace(100, 110, 30)
    stack = pd.DataFrame({"actual": actual, "arima": actual + 4, "prophet": actual + 5,
                          "xgboost": actual + 3, "lightgbm": actual + 9,
                          "naive": actual})  # fmt: skip
    monkeypatch.setattr(forecasters, "build_holdout_stack", lambda *a: stack)
    base = {"arima": _base(200.0), "prophet": _base(210.0), "xgboost": _base(190.0),
            "lightgbm": _base(230.0)}  # fmt: skip
    base["naive"] = _base(220.0).assign(yhat_lower=np.nan, yhat_upper=np.nan)
    out = forecasters.ensemble_stack(_data(_prices()), base, pd.DataFrame())
    assert np.allclose(out["yhat"], 220.0)  # all weight on the perfect naive column
    assert np.allclose(out["yhat_lower"], 220.0) and np.allclose(
        out["yhat_upper"], 220.0
    )
