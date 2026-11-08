"""tests/evals/test_parallel_models.py

The real models in worker processes give exactly the cells they give in one
process: every forecast, band and failure, for every model and both
ensembles. This holds only because each model is deterministic wherever it
runs: the boosters on one thread (XGBoost's subsample depends on the thread
count) and Prophet's band from a fixed seed. Marked eval: it fits every model
at ten origins for three tickers, twice.

Run with:  uv run pytest -m eval
"""

import numpy as np
import pandas as pd
import pytest

from quantflow.evaluation import rolling_origin as ro
from quantflow.evaluation.splits import origin_grid
from quantflow.models.forecasters import COMBINERS, FORECASTERS
from quantflow.utils.trading_calendar import TRADING_DAY

pytestmark = pytest.mark.eval

DATES = pd.date_range("2024-01-02", periods=530, freq=TRADING_DAY)
TICKERS = ["AAPL", "MSFT", "NVDA"]


def _prices():
    rng = np.random.default_rng(7)
    out = {}
    for i, ticker in enumerate(TICKERS):
        close = (80 + 40 * i) * np.exp(np.cumsum(rng.normal(0.0004, 0.015, len(DATES))))
        spread = np.abs(rng.normal(0, 0.01, len(DATES))) * close
        out[ticker] = pd.DataFrame(
            {"date": DATES, "open": close * (1 + rng.normal(0, 0.003, len(DATES))),
             "high": close + spread, "low": close - spread, "close": close,
             "volume": rng.integers(1_000_000, 5_000_000, len(DATES)).astype(float)}
        )  # fmt: skip
    return out


def _articles():
    stored = [DATES[-120] + pd.Timedelta(hours=22 + 24 * k) for k in range(0, 100, 3)]
    return pd.DataFrame(
        {"ticker": ["MSFT"] * len(stored), "source": "rss",
         "url_domain": "finance.yahoo.com",
         "published_at": [t - pd.Timedelta(hours=3) for t in stored],
         "inserted_at": stored, "sentiment": "positive", "score_pos": 0.8,
         "score_neg": 0.1, "score_neu": 0.1, "compound": 0.7}
    )  # fmt: skip


def test_worker_processes_reproduce_a_sequential_run_of_every_model():
    scored, warmup = origin_grid(len(DATES)).window("validation")
    origins = [(p, "scored") for p in scored[-2:]] + [
        (p, "warmup") for p in (*warmup, *scored[:-2])[-8:]
    ]
    run = [_prices(), _articles(), origins, FORECASTERS, COMBINERS]
    sequential = ro.run_origins(*run)
    parallel = ro.run_origins(*run, workers=3)

    assert set(sequential["model"]) == {*FORECASTERS, *COMBINERS}
    # Prophet's band and the ensembles' bands are there to compare.
    has_band = sequential.groupby("model")["yhat_lower"].apply(
        lambda s: s.notna().any()
    )
    assert has_band[["prophet", "arima", "ensemble_stack", "ensemble_rolling"]].all()
    pd.testing.assert_frame_equal(
        parallel.drop(columns="seconds"), sequential.drop(columns="seconds")
    )
