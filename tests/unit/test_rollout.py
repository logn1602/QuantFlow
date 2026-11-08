"""tests/unit/test_rollout.py

The boosters' 7-day rollout (boosting.generate_forecast) feeds each
prediction back in and rebuilds the price features from it. Fed the actual
closes instead of predictions, it must rebuild the features training saw on
those days, or the model is asked about rows unlike any it learned from.

No model is trained: a replay model "predicts" the actual next closes, in
order, and records every row it is asked about.
"""

import numpy as np
import pandas as pd

from quantflow.features.engineering import (
    build_features,
    engineer_features,
    get_feature_cols,
)
from quantflow.models import boosting
from quantflow.utils.trading_calendar import TRADING_DAY

DAYS = pd.date_range("2025-01-02", periods=160, freq=TRADING_DAY)
ARTICLE_COLUMNS = [
    "ticker", "source", "url_domain", "published_at", "inserted_at", "sentiment",
    "score_pos", "score_neg", "score_neu", "compound",
]  # fmt: skip

# The features the rollout recomputes from the closes; the rest stay frozen.
ROLLED = [
    "close_lag_1", "close_lag_2", "close_lag_3", "close_lag_5", "close_lag_10",
    "return_1d", "return_5d",
    "rolling_mean_5", "rolling_mean_10", "rolling_mean_20",
    "rolling_std_5", "rolling_std_10", "rolling_std_20",
    "bb_position", "bb_width",
]  # fmt: skip


class Replay:
    """Predicts the given closes in order; keeps every row it was asked about."""

    def __init__(self, closes):
        self.closes = list(closes)
        self.seen: list[pd.DataFrame] = []

    def predict(self, X):
        self.seen.append(X.copy())
        return np.array([self.closes[len(self.seen) - 1]])


def _features(jump_at: int) -> pd.DataFrame:
    """Training's feature rows for a quiet random walk with a 9% jump on
    DAYS[jump_at], which closes far above the upper Bollinger band."""
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0005, 0.008, len(DAYS))
    returns[jump_at] = 0.09
    close = 100 * np.exp(np.cumsum(returns))
    prices = pd.DataFrame(
        {"date": DAYS, "open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1e6}
    )  # fmt: skip
    raw = build_features(prices, pd.DataFrame(columns=ARTICLE_COLUMNS), "AAPL")
    return engineer_features(raw, drop_unlabeled=False).reset_index(drop=True)


def _rollout(features: pd.DataFrame, start: int) -> list[pd.DataFrame]:
    """The rows the rollout builds from row `start`, fed the actual closes."""
    columns = [c for c in get_feature_cols() if c in features.columns]
    model = Replay(features["close"].iloc[start + 1 : start + 8])
    boosting.generate_forecast(
        model,
        features.iloc[[start]][columns],
        features["date"].iloc[start],
        list(features["close"].iloc[: start + 1]),
        "xgboost",
        "AAPL",
    )
    return model.seen


def test_fed_the_actual_closes_the_rollout_rebuilds_trainings_price_features():
    """Training builds the Bollinger bands with ta (population standard
    deviation, rounded as stored) and never clips bb_position; the rollout
    used the sample standard deviation, so its bb_width came out about 2.6%
    wider, and clipped bb_position to [0, 1], so a close above the upper
    band looked like one on it."""
    features = _features(jump_at=150)
    start = int(features.index[features["date"] == DAYS[147]][0])
    seen = _rollout(features, start)
    assert len(seen) == 7
    # The jump falls inside the rollout and closes above the band.
    assert features["bb_position"].iloc[start + 3] > 1
    for step in range(1, 7):
        expected = features.iloc[start + step][ROLLED].astype(float)
        got = seen[step].iloc[0][ROLLED].astype(float)
        wrong = [c for c in ROLLED if not np.isclose(got[c], expected[c], rtol=1e-9)]
        np.testing.assert_allclose(
            got.to_numpy(), expected.to_numpy(), rtol=1e-9, atol=1e-12,
            err_msg=f"step {step}: {', '.join(wrong)}",
        )  # fmt: skip


def test_the_frozen_features_stay_at_the_starting_rows_values():
    """What the rollout does not recompute it leaves as it was."""
    features = _features(jump_at=150)
    seen = _rollout(features, len(features) - 12)
    frozen = [c for c in seen[0].columns if c not in ROLLED]
    for step in range(1, 7):
        pd.testing.assert_series_equal(
            seen[step].iloc[0][frozen], seen[0].iloc[0][frozen], check_names=False
        )


def test_a_band_with_no_width_gives_trainings_values_not_the_last_ones():
    """Twenty equal closes: training's band has no width, so bb_position is
    missing (it divides by the width) and bb_width is 0; the rollout kept
    the last row's values instead."""
    model = Replay([100.0] * 7)
    X = pd.DataFrame([dict.fromkeys(get_feature_cols(), 0.5)])
    boosting.generate_forecast(model, X, DAYS[-1], [100.0] * 40, "xgboost", "AAPL")
    rolled = model.seen[1].iloc[0]
    assert np.isnan(rolled["bb_position"])
    assert rolled["bb_width"] == 0.0
