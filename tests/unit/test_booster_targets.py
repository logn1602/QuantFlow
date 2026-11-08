"""tests/unit/test_booster_targets.py

What the boosters learn to predict (boosting.TARGETS, Phase 3's E1): the
next close itself, or the next day's log return, turned back into a close
from the close it starts from. Every fit and forecast goes through the same
two functions, so the scored model, the live model and the rollout cannot
disagree about it.

No model is trained: a fake predicts one fixed value and records what it
was fitted on.
"""

import numpy as np
import pandas as pd
import pytest

from quantflow.evaluation.splits import HOLDOUT_DAYS
from quantflow.features.engineering import get_feature_cols
from quantflow.models import boosting

ROWS = pd.DataFrame(
    {"close": [100.0, 110.0, 99.0], "target_price": [110.0, 99.0, 105.0]}
)


class Constant:
    """Predicts `value` for every row; records what it was fitted on."""

    def __init__(self, value: float):
        self.value = value

    def fit(self, X, y, **kwargs):
        self.X, self.y = X.copy(), y.copy()
        self.feature_importances_ = np.ones(X.shape[1])
        return self

    def predict(self, X):
        return np.full(len(X), self.value)


@pytest.fixture
def log_returns(monkeypatch):
    monkeypatch.setattr(
        boosting, "TARGETS", dict.fromkeys(boosting.TARGETS, "log_return")
    )


def _labelled(n=80):
    close = 100 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {"close": close, "close_lag_1": close - 1, "target_price": close + 1}
    )


@pytest.mark.parametrize("target", ["price", "log_return"])
def test_a_predicted_target_turns_back_into_the_close(monkeypatch, target):
    monkeypatch.setitem(boosting.TARGETS, "xgboost", target)
    y = boosting.training_target("xgboost", ROWS)
    closes = boosting.predicted_close("xgboost", y, ROWS["close"])
    np.testing.assert_allclose(closes, ROWS["target_price"], rtol=1e-12)


def test_the_log_return_is_of_the_next_close_over_this_one(log_returns):
    y = boosting.training_target("lightgbm", ROWS)
    np.testing.assert_allclose(y, np.log([1.1, 0.9, 105 / 99]), rtol=1e-12)


def test_the_live_model_is_fitted_on_log_returns(log_returns, monkeypatch):
    monkeypatch.setattr(boosting, "_new_regressor", lambda name: Constant(0.0))
    labelled = _labelled()
    model = boosting.fit_live_model("xgboost", labelled, ["close_lag_1"])
    np.testing.assert_allclose(
        model.y, np.log(labelled["target_price"] / labelled["close"]), rtol=1e-12
    )


def test_the_scored_models_holdout_predictions_are_closes(log_returns, monkeypatch):
    """The ensemble's stack and the holdout score read them as closes: a
    zero log return predicts each holdout day's own close."""
    monkeypatch.setattr(boosting, "_new_regressor", lambda name: Constant(0.0))
    labelled = _labelled()
    result = boosting._train_scored("lightgbm", labelled, "AAPL")
    np.testing.assert_allclose(
        result["holdout_preds"], labelled["close"].iloc[-HOLDOUT_DAYS:], rtol=1e-12
    )


def test_the_rollout_compounds_the_predicted_log_returns(log_returns):
    closes = list(100 + np.arange(40, dtype=float))
    X = pd.DataFrame([dict.fromkeys(get_feature_cols(), 0.5)])
    forecast = boosting.generate_forecast(
        Constant(0.01), X, pd.Timestamp("2026-11-20"), closes, "xgboost", "AAPL"
    )
    expected = closes[-1] * np.exp(0.01 * np.arange(1, 8))
    np.testing.assert_allclose(forecast["predicted_close"], expected, rtol=1e-12)
    np.testing.assert_allclose(forecast["upper_bound"], expected * 1.02, rtol=1e-12)
