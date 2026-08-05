"""
tests/test_features.py
----------------------
Covers xgboost_model.engineer_features and get_feature_cols.

The leakage test perturbs future rows and asserts no earlier feature value
moves — that is what "no future leakage" in the docstring actually means.
"""

import numpy as np
import pandas as pd
import pytest

from xgboost_model import engineer_features, get_feature_cols

N_ROWS = 80


@pytest.fixture
def raw():
    """A synthetic merged frame with every column engineer_features reads.

    `marker` is strictly monotonic so any forward-looking transform would be
    detectable, and it is deliberately NOT a declared feature.
    """
    rng   = np.random.default_rng(7)
    dates = pd.bdate_range("2026-01-01", periods=N_ROWS)
    close = 100 + np.cumsum(rng.normal(0, 1.0, N_ROWS))

    return pd.DataFrame({
        "date":               dates,
        "marker":             np.arange(N_ROWS, dtype=float),
        "open":               close + rng.normal(0, 0.4, N_ROWS),
        "high":               close + np.abs(rng.normal(1.0, 0.3, N_ROWS)),
        "low":                close - np.abs(rng.normal(1.0, 0.3, N_ROWS)),
        "close":              close,
        "volume":            (rng.integers(1_000_000, 5_000_000, N_ROWS)).astype(float),
        "rsi_14":             rng.uniform(20, 80, N_ROWS),
        "macd":               rng.normal(0, 0.5, N_ROWS),
        "macd_signal":        rng.normal(0, 0.5, N_ROWS),
        "macd_hist":          rng.normal(0, 0.2, N_ROWS),
        "bb_upper":           close + 2.0,
        "bb_middle":          close,
        "bb_lower":           close - 2.0,
        "zscore":             rng.normal(0, 1.0, N_ROWS),
        "sentiment_compound": rng.uniform(-1, 1, N_ROWS),
        "pos_count":          rng.integers(0, 5, N_ROWS).astype(float),
        "neg_count":          rng.integers(0, 5, N_ROWS).astype(float),
        "article_count":      rng.integers(0, 9, N_ROWS).astype(float),
    })


# ── get_feature_cols ──────────────────────────────────────────────────────────

def test_feature_count_is_33():
    """The README advertises 33 features. If this fails, either the code or
    the README is wrong — fix whichever is lying."""
    assert len(get_feature_cols()) == 33


def test_feature_names_are_unique():
    cols = get_feature_cols()
    assert len(cols) == len(set(cols))


def test_every_declared_feature_exists_after_engineering(raw):
    out = engineer_features(raw)
    missing = [c for c in get_feature_cols() if c not in out.columns]
    assert missing == [], f"declared but not produced: {missing}"


# ── Target construction ───────────────────────────────────────────────────────

def test_target_price_equals_next_day_close(raw):
    out = engineer_features(raw)

    expected = raw.set_index("date")["close"].shift(-1)
    for date, target in zip(out["date"], out["target_price"]):
        assert target == pytest.approx(expected.loc[date])


def test_target_direction_matches_target_vs_close(raw):
    out = engineer_features(raw)
    expected = (out["target_price"] > out["close"]).astype(int)
    assert (out["target_direction"] == expected).all()


def test_rows_with_unknown_target_are_dropped(raw):
    out = engineer_features(raw)
    # The final row has no next-day close, so it cannot survive.
    assert out["date"].max() < raw["date"].max()
    assert out["target_price"].notna().all()


# ── Leakage ───────────────────────────────────────────────────────────────────

def test_features_do_not_depend_on_future_rows(raw):
    """Feature values at row i must be computable from rows <= i.

    Perturb every input column from row k onward. Any declared feature whose
    value changes at a date before k is reading the future.
    """
    k = 55
    baseline = engineer_features(raw).set_index("date")

    tampered = raw.copy()
    numeric  = [c for c in tampered.columns if c != "date"]
    tampered.loc[k:, numeric] = tampered.loc[k:, numeric] * 3.0 + 17.0
    after = engineer_features(tampered).set_index("date")

    cutoff = raw["date"].iloc[k]
    past   = baseline.index[baseline.index < cutoff]
    assert len(past) > 10, "not enough pre-cutoff rows to make this meaningful"

    leaked = []
    for col in get_feature_cols():
        if not np.allclose(baseline.loc[past, col].values,
                           after.loc[past, col].values,
                           rtol=1e-9, atol=1e-9, equal_nan=True):
            leaked.append(col)

    assert leaked == [], f"features leaking future data: {leaked}"


def test_leakage_probe_would_catch_a_planted_violation(raw):
    """Sanity-check the probe itself: a deliberately forward-looking column
    must be flagged by the same comparison the real test uses."""
    k      = 55
    cutoff = raw["date"].iloc[k]

    def with_planted_leak(frame):
        out = engineer_features(frame)
        # A backwards-shifted close is future information by construction.
        out["planted_leak"] = frame.set_index("date")["close"] \
            .shift(-3).reindex(out["date"]).values
        return out.set_index("date")

    baseline = with_planted_leak(raw)
    tampered = raw.copy()
    numeric  = [c for c in tampered.columns if c != "date"]
    tampered.loc[k:, numeric] = tampered.loc[k:, numeric] * 3.0 + 17.0
    after = with_planted_leak(tampered)

    past = baseline.index[baseline.index < cutoff]
    assert not np.allclose(baseline.loc[past, "planted_leak"].values,
                           after.loc[past, "planted_leak"].values,
                           equal_nan=True)


def test_lag_features_match_manual_shift(raw):
    out = engineer_features(raw).set_index("date")
    close_by_date = raw.set_index("date")["close"]

    for lag in (1, 2, 3, 5, 10):
        expected = close_by_date.shift(lag).reindex(out.index)
        assert np.allclose(out[f"close_lag_{lag}"].values, expected.values)


def test_rolling_mean_uses_only_trailing_window(raw):
    out = engineer_features(raw).set_index("date")
    expected = raw.set_index("date")["close"].rolling(20).mean().reindex(out.index)
    assert np.allclose(out["rolling_mean_20"].values, expected.values)


def test_engineered_features_contain_no_nan(raw):
    out = engineer_features(raw)
    bad = [c for c in get_feature_cols() if out[c].isna().any()]
    assert bad == [], f"NaN left in features: {bad}"
