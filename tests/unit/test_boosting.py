"""tests/unit/test_boosting.py

Covers the live 7-day forecast from XGBoost and LightGBM.

Two bugs, both in what the live forecast was built from:

  - Stale model: it used the model fitted on the train split, so the live
    forecast never saw the last 30 labelled days. ARIMA and Prophet refit
    on the full series before forecasting; the boosters did not.
  - Stale start: engineer_features drops the newest row (its target,
    tomorrow's close, is unknown), and the forecast started from what was
    left: the second-to-last day. Its first "forecast" was for the latest
    day, whose close was already known.

No model is trained: the regressors are fakes whose prediction is the number
of rows they were fitted on, which identifies the model a forecast came from.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest

from quantflow.evaluation.splits import HOLDOUT_DAYS
from quantflow.features.engineering import engineer_features
from quantflow.models import boosting
from quantflow.utils.trading_calendar import TRADING_DAY

DAYS = pd.date_range(end="2026-11-20", periods=120, freq=TRADING_DAY)


class FakeRegressor:
    """Predicts the number of rows it was fitted on; records what it saw."""

    def __init__(self, **params):
        self.X = None
        self.predicted_on = []

    def fit(self, X, y, **kwargs):
        self.X = X.copy()
        self.feature_importances_ = np.ones(X.shape[1])
        return self

    def predict(self, X):
        self.predicted_on.append(X.copy())
        return np.full(len(X), float(len(self.X)))


@pytest.fixture(autouse=True)
def price_targets(monkeypatch):
    """These fakes identify a model by the price it predicts, so both
    families predict prices here; the log-return target has its own tests
    (test_booster_targets.py)."""
    monkeypatch.setattr(boosting, "TARGETS", dict.fromkeys(boosting.TARGETS, "price"))


@pytest.fixture
def raw():
    """A merged feature frame like load_features returns, ending on a Friday."""
    rng = np.random.default_rng(3)
    close = 100 + np.cumsum(rng.normal(0, 1.0, len(DAYS)))
    n = len(DAYS)
    return pd.DataFrame(
        {
            "date": DAYS,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
            "rsi_14": rng.uniform(20, 80, n),
            "macd": rng.normal(0, 0.5, n),
            "macd_signal": rng.normal(0, 0.5, n),
            "macd_hist": rng.normal(0, 0.2, n),
            "bb_upper": close + 2.0,
            "bb_middle": close,
            "bb_lower": close - 2.0,
            "zscore": rng.normal(0, 1.0, n),
            "sentiment_compound": rng.uniform(-1, 1, n),
            "pos_count": rng.integers(0, 5, n).astype(float),
            "neg_count": rng.integers(0, 5, n).astype(float),
            "article_count": rng.integers(0, 9, n).astype(float),
        }
    )


@pytest.fixture
def forecasts(raw, monkeypatch):
    """Run boosting.run on raw with fake regressors; return what it saved."""
    return _run(raw, monkeypatch)


def _run(raw, monkeypatch):
    models = []

    def regressor(**params):
        model = FakeRegressor(**params)
        models.append(model)
        return model

    for name, cls in (("xgboost", "XGBRegressor"), ("lightgbm", "LGBMRegressor")):
        module = types.ModuleType(name)
        setattr(module, cls, regressor)
        monkeypatch.setitem(sys.modules, name, module)

    saved = {}
    monkeypatch.setattr(boosting, "load_news", lambda: None)
    monkeypatch.setattr(boosting, "load_features", lambda ticker, news: raw.copy())
    monkeypatch.setattr(
        boosting,
        "save_forecasts",
        lambda df: saved.setdefault(df["model"].iloc[0], df) is None or len(df),
    )
    monkeypatch.setattr(boosting, "save_model_metrics", lambda *a, **k: None)
    monkeypatch.setattr(boosting, "log_mlflow", lambda *a, **k: None)

    boosting.run(["AAPL"])
    return saved, models


def _labelled(raw):
    return engineer_features(raw)


# ── Stale model: the live model sees every labelled row ───────────────────────


@pytest.mark.parametrize("model_name", ["xgboost", "lightgbm"])
def test_the_live_forecast_comes_from_a_model_fit_on_every_labelled_row(
    forecasts, raw, model_name
):
    saved, _ = forecasts
    n_labelled = len(_labelled(raw))

    assert (saved[model_name]["predicted_close"] == n_labelled).all()
    assert n_labelled - HOLDOUT_DAYS not in set(saved[model_name]["predicted_close"])


def test_the_live_model_has_seen_the_final_labelled_row(forecasts, raw):
    _, models = forecasts
    labelled = _labelled(raw)
    live = [m for m in models if m.predicted_on and len(m.X) == len(labelled)]

    assert live, "no model fitted on every labelled row made a forecast"
    for model in live:
        final = model.X.iloc[-1]
        expected = labelled[model.X.columns].iloc[-1]
        assert np.allclose(
            final.to_numpy(dtype=float), expected.to_numpy(dtype=float), equal_nan=True
        )


# ── Missing is missing (finding F7) ───────────────────────────────────────────


def test_a_missing_feature_reaches_the_boosters_as_missing_not_zero(raw, monkeypatch):
    """XGBoost and LightGBM handle missing values natively. Filled with 0, an
    unknown RSI or a day without news reads as a real value."""
    gappy = raw.copy()
    gappy.loc[60:70, "article_count"] = np.nan  # news not known on those days
    gappy.loc[gappy.index[-1], "rsi_14"] = np.nan  # the newest row's RSI
    _, models = _run(gappy, monkeypatch)

    fitted = [m for m in models if m.X is not None]
    assert len(fitted) == 4  # scored and live, per family
    for model in fitted:
        assert model.X.loc[60:70, "has_news"].isna().all()
    first_steps = [m.predicted_on[-7] for m in models if len(m.predicted_on) >= 7]
    assert len(first_steps) == 2
    for X in first_steps:
        assert np.isnan(X["rsi_14"].iloc[0])


# ── Stale start: the forecast starts from the newest row ──────────────────────


@pytest.mark.parametrize("model_name", ["xgboost", "lightgbm"])
def test_the_first_forecast_date_is_the_next_trading_day_after_the_last_close(
    forecasts, model_name
):
    saved, _ = forecasts
    dates = [pd.Timestamp(d) for d in saved[model_name]["ds"]]

    assert dates[0] > DAYS[-1]
    assert [d.date().isoformat() for d in dates] == [
        "2026-11-23", "2026-11-24", "2026-11-25", "2026-11-27",
        "2026-11-30", "2026-12-01", "2026-12-02",
    ]  # fmt: skip


def test_the_first_step_is_predicted_from_the_newest_rows_features(forecasts, raw):
    """close_lag_1 of the newest row is the second-to-last close."""
    _, models = forecasts
    # The 7 forecast steps are each forecasting model's last 7 predict calls.
    first_steps = [m.predicted_on[-7] for m in models if len(m.predicted_on) >= 7]

    assert len(first_steps) == 2
    for X in first_steps:
        assert X["close_lag_1"].iloc[0] == pytest.approx(raw["close"].iloc[-2])


def test_an_unknown_model_family_is_refused():
    """Anything other than "xgboost" used to become a LightGBM regressor,
    whose forecasts were then stored under the name it was asked for."""
    with pytest.raises(ValueError, match="unknown model family"):
        boosting._new_regressor("xgb")


@pytest.mark.parametrize("family", ["xgboost", "lightgbm"])
def test_each_booster_fits_on_one_thread(family):
    """XGBoost draws its row subsample (subsample 0.8) differently for
    different thread counts, so with the default, all of the machine's
    threads, the same data gave forecasts up to about $3 apart on this
    laptop (8 threads, where the evaluation ran) and on GitHub's runner:
    the published forecast was not the evaluated one. One thread makes them
    the same on any machine. LightGBM does not depend on it; pinned too, so
    the evaluation's worker processes run one thread per model."""
    assert boosting._new_regressor(family).get_params()["n_jobs"] == 1


class SpyRegressor(FakeRegressor):
    """Records everything fit was given."""

    def fit(self, X, y, **kwargs):
        self.fit_kwargs = kwargs
        self.y = y.copy()
        return super().fit(X, y)


@pytest.mark.parametrize("family", ["xgboost", "lightgbm"])
def test_the_scored_fit_never_sees_the_holdout(raw, monkeypatch, family):
    """XGBoost's scored fit was handed the holdout as eval_set. Without
    early stopping that changes no prediction, but the holdout it is scored
    on must not reach the fit at all: early stopping or tuning reading it
    would quietly pick the model on the very rows that score it."""
    spy = SpyRegressor()
    monkeypatch.setattr(boosting, "_new_regressor", lambda name, params=None: spy)
    labelled = _labelled(raw)
    boosting._train_scored(family, labelled, "AAPL")
    assert spy.fit_kwargs == {}
    assert len(spy.X) == len(labelled) - HOLDOUT_DAYS
    pd.testing.assert_series_equal(
        spy.y, labelled["target_price"].iloc[:-HOLDOUT_DAYS], check_names=False
    )


@pytest.mark.parametrize("family", ["xgboost", "lightgbm"])
def test_tuning_settings_replace_only_what_they_name(family):
    before = dict(boosting.MODEL_PARAMS[family])
    model = boosting._new_regressor(family, {"max_depth": 5, "n_estimators": 50})
    params = model.get_params()
    assert params["max_depth"] == 5 and params["n_estimators"] == 50
    assert params["learning_rate"] == before["learning_rate"]
    assert params["n_jobs"] == 1
    assert boosting.MODEL_PARAMS[family] == before  # left as it was
