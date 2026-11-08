"""tests/evals/test_rolling_origin_evaluation.py

The whole rolling-origin path on a synthetic random walk, with fast fake
models whose right answers are known: an oracle that is told the future
closes must beat naive on the primary test, a model that is naive plus noise
must not, and naive must score skill 0 against itself. Marked eval: it runs
every validation origin for three tickers, which is slow for the commit loop.

Run with:  uv run pytest -m eval
"""

import numpy as np
import pandas as pd
import pytest

from quantflow.evaluation import report
from quantflow.evaluation import rolling_origin as ro
from quantflow.evaluation.splits import origin_grid
from quantflow.utils.trading_calendar import TRADING_DAY

pytestmark = pytest.mark.eval

DATES = pd.date_range("2024-01-02", periods=530, freq=TRADING_DAY)
TICKERS = ["AAA", "BBB", "CCC"]


def _walks():
    rng = np.random.default_rng(42)
    out = {}
    for i, ticker in enumerate(TICKERS):
        close = (50 + 50 * i) * np.exp(np.cumsum(rng.normal(0, 0.015, len(DATES))))
        out[ticker] = pd.DataFrame(
            {"date": DATES, "open": close, "high": close, "low": close,
             "close": close, "volume": 1e6}
        )  # fmt: skip
    return out


PRICES = _walks()


def oracle(data):
    """Told the true closes by the test, never by the harness."""
    truth = PRICES[data.ticker].set_index("date")["close"]
    return pd.DataFrame({"yhat": truth.loc[data.dates].to_numpy()})


def noisy(data):
    rng = np.random.default_rng(int(data.origin.value % 2**32))
    out = ro.naive_forecast(data)
    out["yhat"] = out["yhat"] * (1 + rng.normal(0, 0.02, len(out)))
    return out


@pytest.fixture(scope="module")
def cells():
    scored, _ = origin_grid(len(DATES)).window("validation")
    return ro.run_origins(
        PRICES,
        pd.DataFrame(),
        [(p, "scored") for p in scored],
        {**ro.BASELINES, "oracle": oracle, "noisy": noisy},
    )


def test_the_oracle_beats_naive_and_the_noisy_model_does_not(cells):
    head = report.headline(report.comparable(report.scored(cells))).set_index("model")
    assert head.loc["oracle", "beats_naive"]
    assert head.loc["oracle", "mean_skill"] == pytest.approx(1.0)
    assert head.loc["oracle", "tickers_positive_skill"] == len(TICKERS)
    assert not head.loc["noisy", "beats_naive"]
    assert head.loc["noisy", "mean_skill"] < 0
    assert head.loc["naive", "mean_skill"] == pytest.approx(0.0)


def test_every_validation_origin_is_scored_for_every_model(cells):
    assert cells["status"].eq("ok").all()
    per_model = cells.groupby("model")["origin"].nunique()
    assert (per_model == 42).all()
    assert len(cells) == 42 * 7 * len(TICKERS) * 4


def test_the_report_renders_from_the_cells(cells):
    text = report.markdown({"window": "validation"}, cells)
    assert "| oracle |" in text and "## Per horizon" in text
