"""tests/unit/test_backtest_pipeline.py

Covers quantflow.pipelines.backtest, the run that wires the ensemble into
the strategy simulation, and evaluation.backtest.backtest_out_of_fold, the
window alignment it relies on. No model is trained: the holdout stack is
canned, and the out-of-fold loop is the real NNLS one.
"""

import json

import numpy as np
import pandas as pd
import pytest

from quantflow.evaluation import backtest as evaluation_backtest
from quantflow.models import ensemble
from quantflow.pipelines import backtest as pipeline

N_PRICES = 100


def _prices(seed=5):
    rng = np.random.default_rng(seed)
    closes = 100 + np.cumsum(rng.normal(0, 1, N_PRICES))
    return pd.DataFrame(
        {"ds": pd.bdate_range("2026-01-01", periods=N_PRICES), "y": closes}
    )


def _stack(prices, seed=6):
    """A holdout stack over the last 30 closes, base models near the truth."""
    rng = np.random.default_rng(seed)
    actual = prices["y"].to_numpy()[-30:]
    noisy = {m: actual + rng.normal(0, 1, 30) for m in ensemble.STACK_COLUMNS}
    return pd.DataFrame({"actual": actual} | noisy)


@pytest.fixture
def wired(monkeypatch):
    prices = _prices()
    stack = _stack(prices)
    monkeypatch.setattr(
        ensemble, "collect_holdout_stacks", lambda ticker, news=None: stack.copy()
    )
    monkeypatch.setattr(pipeline, "load_prices", lambda ticker: prices)
    saved: list[dict] = []
    monkeypatch.setattr(pipeline, "replace_backtest_result", saved.append)
    monkeypatch.setattr(pipeline, "log_to_mlflow", lambda *args, **kwargs: None)
    return saved


def test_the_backtest_runs_the_out_of_fold_loop_once(monkeypatch, wired):
    """Known problem #6: tune_and_train_meta was called first, for its log
    lines, and repeated the whole out-of-fold loop."""
    calls = []
    real = ensemble.expanding_window_predictions

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(ensemble, "expanding_window_predictions", counting)
    pipeline.run_backtest("AAPL")
    assert len(calls) == 1


def test_the_saved_result_covers_the_out_of_fold_window(wired):
    metrics = pipeline.run_backtest("AAPL")
    (row,) = wired
    assert row["ticker"] == "AAPL"
    assert row["holdout_days"] == 30 - ensemble.META_WARMUP_DAYS
    assert row["num_trades"] == metrics["num_trades"]
    assert len(json.loads(row["daily_values"])) == row["holdout_days"] + 1


def test_an_empty_stack_saves_nothing(monkeypatch, wired):
    monkeypatch.setattr(
        ensemble, "collect_holdout_stacks", lambda t, news=None: pd.DataFrame()
    )
    assert pipeline.run_backtest("AAPL") == {}
    assert wired == []


def test_one_tickers_error_does_not_stop_the_others(monkeypatch):
    news = object()  # read once, handed to every ticker

    def flaky(ticker, given):
        assert given is news
        if ticker == "AAPL":
            raise RuntimeError("boom")
        return {"total_return": 1.0}

    monkeypatch.setattr(pipeline, "load_news", lambda: news)
    monkeypatch.setattr(pipeline, "run_backtest", flaky)
    assert pipeline.run(["AAPL", "MSFT"]) == {"AAPL": {}, "MSFT": {"total_return": 1.0}}


GOOD = {"total_return": 1.0, "alpha": 0.5, "sharpe_ratio": 0.1, "win_rate": 50.0}


@pytest.mark.parametrize(
    ("results", "code"),
    [
        ({"AAPL": {}, "MSFT": {}}, 1),  # every ticker failed
        ({"AAPL": {}, "MSFT": GOOD}, 0),  # one thin ticker is not a failed run
        ({"AAPL": GOOD}, 0),
    ],
)
def test_the_cli_fails_only_when_every_ticker_fails(monkeypatch, results, code):
    monkeypatch.setattr(pipeline, "run", lambda tickers=None: results)
    assert pipeline.main([]) == code


def test_a_misaligned_stack_fails_loudly():
    """The stack's actuals must be the holdout's closes, day for day."""
    prices = _prices()
    stack = _stack(prices)
    stack["actual"] = np.roll(stack["actual"].to_numpy(), 1)
    with pytest.raises(AssertionError, match="misaligned"):
        evaluation_backtest.backtest_out_of_fold(
            stack,
            prices["y"].to_numpy(),
            ensemble.out_of_fold_meta_predictions,
            "AAPL",
        )


def test_no_out_of_fold_day_gives_no_result():
    prices = _prices()
    result = evaluation_backtest.backtest_out_of_fold(
        _stack(prices),
        prices["y"].to_numpy(),
        lambda stack: (np.array([]), np.array([])),
        "AAPL",
    )
    assert result is None
