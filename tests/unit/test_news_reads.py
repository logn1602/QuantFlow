"""tests/unit/test_news_reads.py

How often a daily run reads the stored news. Each stage used to read every
article of a ticker, and every fetch time in the table, once per ticker: the
model run (boosters, then the ensemble) and the backtest together read the
news three times per ticker, about 18.7 MB a run on the 2026-11-06 snapshot,
growing with every article stored. Now each process reads it once, summed
per news day (db.features.load_news_days).
"""

import numpy as np
import pandas as pd
import pytest

from quantflow.models import boosting, ensemble
from quantflow.pipelines import backtest, run_models

TICKERS = ["AAPL", "MSFT"]
DAYS = pd.bdate_range("2026-01-02", periods=120)

# Every column of the article rows and of the per-day summary, so the reads
# can be counted on the code before the change as well as after it.
NO_NEWS = pd.DataFrame(
    columns=[
        "ticker", "source", "url_domain", "published_at", "inserted_at",
        "sentiment", "score_pos", "score_neg", "score_neu", "compound",
        "available_day", "stored_day", "counts", "articles", "positive",
        "negative", "compounds", "compound_sum",
    ]
)  # fmt: skip


class FakeRegressor:
    def fit(self, X, y, **kwargs):
        self.mean_y = float(np.mean(y))
        self.feature_importances_ = np.full(X.shape[1], 1.0 / X.shape[1])
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_y)


@pytest.fixture
def news_reads(read_tables, price_bars, monkeypatch):
    """Two tickers' bars; every query of news_sentiment recorded."""
    closes = 100 * np.exp(
        np.cumsum(np.random.default_rng(3).normal(0, 0.01, len(DAYS)))
    )
    read_tables["raw_prices"] = price_bars.frame(
        [
            price_bars.daily(day.date(), float(close))
            for day, close in zip(DAYS, closes, strict=True)
        ]
    )
    reads = []

    def news(params):
        reads.append(params)
        return NO_NEWS

    read_tables["news_sentiment"] = news
    monkeypatch.setattr(boosting, "_new_regressor", lambda name: FakeRegressor())
    for name in ("save_forecasts", "save_model_metrics", "log_mlflow"):
        monkeypatch.setattr(boosting, name, lambda *args, **kwargs: 0)
    # The ensemble and the backtest read the features for the holdout stack;
    # building it is not what this counts.
    monkeypatch.setattr(ensemble, "build_holdout_stack", lambda *a: pd.DataFrame())
    return reads


def test_the_model_run_reads_the_news_once(news_reads, monkeypatch):
    """Boosters and ensemble, two tickers: one read, not one per ticker for
    each of them."""
    import quantflow.models.statistical as statistical

    monkeypatch.setattr(statistical, "run", lambda tickers=None: {})
    monkeypatch.setattr(run_models, "clear_forecasts", lambda tickers=None: None)

    results, failed = run_models.run(TICKERS)

    assert failed == []
    assert set(results["xgboost_lightgbm"]) == set(TICKERS)
    assert len(news_reads) == 1


def test_the_backtest_reads_the_news_once(news_reads):
    backtest.run(TICKERS)
    assert len(news_reads) == 1
