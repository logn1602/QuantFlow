"""
tests/test_ensemble.py
----------------------
Covers the NNLS meta-learner and the out-of-fold evaluation loop.

The look-ahead tests are the ones that matter: they are what stops a future
edit from quietly reverting the ensemble to in-sample scoring.
"""

import numpy as np
import pandas as pd
import pytest

from ensemble import (
    META_WARMUP_DAYS, NNLSMeta, _fit_nnls, _mape,
    out_of_fold_meta_predictions,
)

BASE_COLS = ["arima", "prophet", "xgboost", "lightgbm"]


@pytest.fixture
def stacked():
    """A 30-row holdout stack whose base models have differing accuracy."""
    rng   = np.random.default_rng(20260803)
    n     = 30
    truth = 100 + np.cumsum(rng.normal(0, 1.5, n))
    return pd.DataFrame({
        "actual":   truth,
        "arima":    truth + rng.normal(0, 3.0, n),
        "prophet":  truth + rng.normal(0, 2.0, n),
        "xgboost":  truth + rng.normal(0, 1.0, n),
        "lightgbm": truth + rng.normal(0, 1.2, n),
    })


# ── _mape ─────────────────────────────────────────────────────────────────────

def test_mape_known_input_known_output():
    actual    = np.array([100.0, 200.0])
    predicted = np.array([110.0, 180.0])
    # |10/100| = 0.10, |20/200| = 0.10  ->  mean 0.10  ->  10%
    assert _mape(actual, predicted) == pytest.approx(10.0)


def test_mape_is_zero_for_perfect_predictions():
    actual = np.array([50.0, 75.0, 120.0])
    assert _mape(actual, actual.copy()) == pytest.approx(0.0)


# ── _fit_nnls ─────────────────────────────────────────────────────────────────

def test_fit_nnls_weights_are_non_negative(stacked):
    meta = _fit_nnls(stacked[BASE_COLS].values, stacked["actual"].values)
    assert (meta.coef_ >= 0).all(), f"negative weight: {meta.coef_}"


def test_fit_nnls_weights_sum_to_one(stacked):
    meta = _fit_nnls(stacked[BASE_COLS].values, stacked["actual"].values)
    assert meta.coef_.sum() == pytest.approx(1.0, abs=1e-9)


def test_fit_nnls_accepts_dataframes_as_well_as_arrays(stacked):
    from_frame = _fit_nnls(stacked[BASE_COLS], stacked["actual"])
    from_array = _fit_nnls(stacked[BASE_COLS].values, stacked["actual"].values)
    assert np.allclose(from_frame.coef_, from_array.coef_)


def test_fit_nnls_favours_the_most_accurate_base_model(stacked):
    # xgboost carries the least noise in the fixture, so it should not be the
    # least-weighted model.
    meta    = _fit_nnls(stacked[BASE_COLS].values, stacked["actual"].values)
    weights = dict(zip(BASE_COLS, meta.coef_))
    assert weights["xgboost"] > weights["arima"]


# ── NNLSMeta.predict ──────────────────────────────────────────────────────────

def test_predict_is_a_convex_combination(stacked):
    """The class docstring promises predictions can never fall outside
    [min(base), max(base)] for a given day. Hold it to that."""
    meta  = _fit_nnls(stacked[BASE_COLS].values, stacked["actual"].values)
    base  = stacked[BASE_COLS].values
    preds = meta.predict(base)

    assert (preds >= base.min(axis=1) - 1e-9).all()
    assert (preds <= base.max(axis=1) + 1e-9).all()


def test_predict_returns_the_common_value_when_all_bases_agree():
    meta = NNLSMeta(np.array([0.25, 0.25, 0.25, 0.25]))
    X    = np.full((3, 4), 42.0)
    assert np.allclose(meta.predict(X), 42.0)


def test_predict_accepts_a_dataframe(stacked):
    meta = _fit_nnls(stacked[BASE_COLS].values, stacked["actual"].values)
    assert np.allclose(meta.predict(stacked[BASE_COLS]),
                       meta.predict(stacked[BASE_COLS].values))


# ── out_of_fold_meta_predictions ──────────────────────────────────────────────

def test_oof_output_length(stacked):
    preds, actuals = out_of_fold_meta_predictions(stacked)
    expected = len(stacked) - META_WARMUP_DAYS

    assert len(preds) == expected
    assert len(actuals) == expected


def test_oof_actuals_are_the_tail_of_the_stack(stacked):
    _, actuals = out_of_fold_meta_predictions(stacked)
    assert np.allclose(actuals, stacked["actual"].values[META_WARMUP_DAYS:])


def test_oof_respects_a_custom_warmup(stacked):
    preds, actuals = out_of_fold_meta_predictions(stacked, warmup=5)
    assert len(preds) == len(stacked) - 5
    assert len(actuals) == len(stacked) - 5


def test_oof_returns_empty_when_stack_is_too_short(stacked):
    preds, actuals = out_of_fold_meta_predictions(stacked.iloc[:META_WARMUP_DAYS])
    assert len(preds) == 0
    assert len(actuals) == 0


def test_oof_predictions_are_convex_combinations(stacked):
    preds, _ = out_of_fold_meta_predictions(stacked)
    base     = stacked[BASE_COLS].values[META_WARMUP_DAYS:]

    assert (preds >= base.min(axis=1) - 1e-9).all()
    assert (preds <= base.max(axis=1) + 1e-9).all()


def test_oof_predictions_are_finite(stacked):
    preds, _ = out_of_fold_meta_predictions(stacked)
    assert np.isfinite(preds).all()


# ── Look-ahead protection — the point of Phase 3 ──────────────────────────────

def test_no_look_ahead_changing_last_actual_leaves_earlier_preds_untouched(stacked):
    """If any OOF prediction shifts when a LATER actual changes, the loop is
    leaking future information."""
    baseline, _ = out_of_fold_meta_predictions(stacked)

    tampered = stacked.copy()
    tampered.loc[len(tampered) - 1, "actual"] += 500.0
    after, _ = out_of_fold_meta_predictions(tampered)

    assert np.allclose(baseline[:-1], after[:-1])
    # The final day is fitted on days before it, so it cannot see its own
    # actual either.
    assert baseline[-1] == pytest.approx(after[-1])


def test_no_look_ahead_changing_a_middle_actual_only_moves_later_preds(stacked):
    baseline, _ = out_of_fold_meta_predictions(stacked)

    k        = 20
    tampered = stacked.copy()
    tampered.loc[k, "actual"] += 500.0
    after, _ = out_of_fold_meta_predictions(tampered)

    split = k - META_WARMUP_DAYS
    # Day k's own prediction is fitted on [0, k), so it is unaffected too.
    assert np.allclose(baseline[:split + 1], after[:split + 1])
    # Later days train on the tampered row, so at least one must move.
    assert not np.allclose(baseline[split + 1:], after[split + 1:])


def test_oof_mape_is_not_better_than_in_sample_mape(stacked):
    """In-sample scoring flatters the ensemble. An honest OOF estimate must
    not come out ahead of it."""
    preds, actuals = out_of_fold_meta_predictions(stacked)
    meta           = _fit_nnls(stacked[BASE_COLS].values, stacked["actual"].values)

    in_sample = _mape(stacked["actual"].values,
                      meta.predict(stacked[BASE_COLS].values))
    oof       = _mape(actuals, preds)

    assert oof >= in_sample - 1e-9
