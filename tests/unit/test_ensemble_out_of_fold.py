"""tests/unit/test_ensemble_out_of_fold.py

The stacking ensemble's scoring path, end to end: stack, fit, out-of-fold
loop, comparison. On synthetic base-model predictions rather than a seeded
database, so they are reproducible anywhere and assert properties that must
hold for any input.

What they defend is the claim the README makes: that the ensemble's reported
figures are out-of-fold, and that the comparison against base models is
like-for-like. The earlier version of this project reported the ensemble as
best on all eight tickers by scoring it in-sample. These assertions are what
make that regression impossible to reintroduce silently.

They were in tests/evals, marked eval, as too slow for the commit loop, so no
automated run ever ran them: CI, both workflows and make test leave eval out.
They take about a tenth of a second, so they are unit tests now (found in the
full-diff review, 2026-11-12).
"""

import numpy as np
import pandas as pd
import pytest

from quantflow.evaluation.metrics import mape
from quantflow.evaluation.out_of_fold import expanding_window_predictions
from quantflow.models.ensemble import STACK_COLUMNS, _fit_nnls, tune_and_train_meta

# The columns the meta-learner weighs: the base models and naive.
COLUMNS = STACK_COLUMNS


def _stack(n=30, seed=7, noise=(3.0, 2.0, 1.0, 1.2, 4.0)):
    """A holdout stack with base models of deliberately differing accuracy."""
    rng = np.random.default_rng(seed)
    truth = 100 + np.cumsum(rng.normal(0, 1.5, n))
    frame = {"actual": truth}
    for model, sigma in zip(COLUMNS, noise, strict=True):
        frame[model] = truth + rng.normal(0, sigma, n)
    return pd.DataFrame(frame)


# ── The honesty guarantee ─────────────────────────────────────────────────────


def test_out_of_fold_scores_worse_than_in_sample_on_average():
    """In-sample scoring flatters the meta-learner: it has already seen every
    actual it is graded against. An honest out-of-fold estimate must therefore
    be worse *on average*.

    Deliberately an aggregate over many seeds rather than a per-seed assertion.
    Out-of-sample error exceeding in-sample error is a statistical property, not
    an arithmetic identity — on any single 20-day window the out-of-fold score
    can beat in-sample by chance, and it does so for roughly two seeds in five
    here. A per-seed version of this test would look like it was checking a
    guarantee while actually encoding a coincidence, and would fail the first
    time anyone touched the fixture.

    If the loop ever reverted to in-sample scoring, the two means would become
    equal rather than merely close, so this still catches the regression.
    """
    oof_scores, in_sample_scores = [], []
    for seed in range(40):
        stacked = _stack(seed=seed)
        preds, actuals = expanding_window_predictions(stacked, COLUMNS, _fit_nnls)

        meta = _fit_nnls(stacked[COLUMNS].values, stacked["actual"].values)
        in_sample_scores.append(
            mape(stacked["actual"].values, meta.predict(stacked[COLUMNS].values))
        )
        oof_scores.append(mape(actuals, preds))

    assert np.mean(oof_scores) > np.mean(in_sample_scores)


def test_reported_metrics_are_scored_on_the_evaluation_window_not_the_holdout():
    """Base models must be re-scored on the same window as the ensemble.
    Comparing a 20-day out-of-fold ensemble against 30-day base figures is the
    subtle version of the same leak, and reads as a win that is not there."""
    stacked = _stack()
    _, metrics = tune_and_train_meta(stacked)

    warmup = metrics["warmup_days"]
    for model in COLUMNS:
        expected = round(
            mape(stacked["actual"].values[warmup:], stacked[model].values[warmup:]), 2
        )
        assert metrics[f"{model}_mape"] == expected


def test_evaluation_window_is_shorter_than_the_holdout():
    """Fitting the first meta-learner consumes the warmup, so eval_days must be
    strictly less than the stack length. A run reporting eval_days equal to the
    holdout has skipped the warmup and is scoring in-sample."""
    stacked = _stack()
    _, metrics = tune_and_train_meta(stacked)
    assert 0 < metrics["eval_days"] < len(stacked)
    assert metrics["eval_days"] + metrics["warmup_days"] == len(stacked)


def test_improvement_is_free_to_be_negative():
    """The ensemble genuinely loses on most tickers. A metric that cannot go
    negative would be hiding that."""
    # Base models far more accurate than any convex blend of them can be.
    stacked = _stack(noise=(0.05, 8.0, 8.0, 8.0, 8.0))
    _, metrics = tune_and_train_meta(stacked)
    assert metrics["improvement_pct"] is not None
    assert isinstance(metrics["improvement_pct"], float)


# ── Weight behaviour ──────────────────────────────────────────────────────────


def test_weights_form_a_convex_combination():
    stacked = _stack()
    _, metrics = tune_and_train_meta(stacked)
    weights = np.array(list(metrics["coefficients"].values()))
    assert (weights >= 0).all()
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)


def test_predictions_stay_within_the_range_of_the_base_models():
    """The reason NNLS replaced Ridge: a convex combination cannot extrapolate
    outside its inputs when live prices drift beyond the training range."""
    stacked = _stack()
    meta, _ = tune_and_train_meta(stacked)
    base = stacked[COLUMNS].values
    preds = meta.predict(base)
    assert (preds >= base.min(axis=1) - 1e-9).all()
    assert (preds <= base.max(axis=1) + 1e-9).all()


def test_a_dominant_base_model_attracts_the_most_weight():
    stacked = _stack(noise=(6.0, 6.0, 0.1, 6.0, 6.0))  # xgboost far more accurate
    _, metrics = tune_and_train_meta(stacked)
    assert metrics["coefficients"]["xgboost"] == max(metrics["coefficients"].values())


# ── Degenerate input ──────────────────────────────────────────────────────────


def test_a_stack_too_short_to_score_reports_no_metrics_rather_than_guessing():
    """Real tickers with thin history hit this. It must return None metrics,
    not a number computed from zero evaluation days."""
    stacked = _stack(n=8)
    _, metrics = tune_and_train_meta(stacked)
    assert metrics["eval_days"] == 0
    assert metrics["ensemble_mape"] is None
    assert metrics["improvement_pct"] is None
