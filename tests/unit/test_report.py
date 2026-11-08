"""tests/unit/test_report.py

Covers the tables built from rolling-origin cells, on hand-built cells
whose answers can be worked out on paper.
"""

import numpy as np
import pandas as pd
import pytest

from quantflow.evaluation import report

ORIGINS = pd.bdate_range("2026-05-04", periods=6, freq="5B")


def _cells(errors: dict[str, float], tickers=("AAPL", "MSFT"), horizon=2):
    """Every model forecasts actual 100 plus its error, at every origin."""
    rows = []
    for ticker in tickers:
        for origin in ORIGINS:
            for h in range(1, horizon + 1):
                for model, error in errors.items():
                    rows.append(
                        {"ticker": ticker, "model": model, "origin": origin, "h": h,
                         "role": "scored", "status": "ok", "error": "",
                         "origin_close": 99.0, "actual": 100.0, "yhat": 100.0 + error,
                         "yhat_lower": np.nan, "yhat_upper": np.nan}
                    )  # fmt: skip
    return pd.DataFrame(rows)


def test_skill_is_measured_against_naive_on_the_same_cells():
    table = report.summary_table(
        _cells({"naive": -1.0, "good": 0.5}), ["ticker", "model"]
    )
    good = table[table["model"] == "good"]
    assert good["mape"].tolist() == pytest.approx([0.5, 0.5])
    assert good["skill"].tolist() == pytest.approx([0.5, 0.5])
    assert table[table["model"] == "naive"]["skill"].tolist() == [0.0, 0.0]


def test_a_failed_cell_leaves_every_comparison():
    cells = _cells({"naive": -1.0, "good": 0.5})
    failed = (
        (cells["model"] == "good")
        & (cells["origin"] == ORIGINS[0])
        & (cells["ticker"] == "AAPL")
    )
    cells.loc[failed, ["status", "yhat"]] = ["failed", np.nan]
    compared = report.comparable(cells)
    dropped = (compared["origin"] == ORIGINS[0]) & (compared["ticker"] == "AAPL")
    assert not dropped.any()  # naive's cells there are gone too
    assert len(compared) == len(cells) - 2 * 2  # 2 models x 2 days


def test_warmup_origins_are_not_scored():
    cells = _cells({"naive": -1.0})
    cells.loc[cells["origin"] == ORIGINS[0], "role"] = "warmup"
    assert ORIGINS[0] not in set(report.scored(cells)["origin"])


def test_the_primary_test_and_the_headline():
    rng = np.random.default_rng(0)
    cells = _cells({"naive": -1.0, "good": 0.0, "bad": 0.0})
    # good beats naive by about 0.5 points everywhere; bad is naive plus noise.
    cells.loc[cells["model"] == "good", "yhat"] = 100.5 + rng.normal(
        0, 0.01, (cells["model"] == "good").sum()
    )
    cells.loc[cells["model"] == "bad", "yhat"] = 99.0 + rng.normal(
        0, 0.3, (cells["model"] == "bad").sum()
    )
    head = report.headline(report.comparable(cells)).set_index("model")
    assert head.loc["good", "beats_naive"] and not head.loc["bad", "beats_naive"]
    assert head.loc["good", "mean_ape_difference"] == pytest.approx(-0.5, abs=0.02)
    assert head.loc["good", "tickers_positive_skill"] == 2
    assert head.loc["good", "p_holm"] >= head.loc["good", "p"]
    assert "beats_naive" not in head.loc["naive"] or pd.isna(head.loc["naive", "p"])


def test_failures_are_counted_per_model():
    cells = _cells({"naive": -1.0, "arima": 0.2})
    cells.loc[(cells["model"] == "arima") & (cells["h"] == 1), ["status", "error"]] = [
        "failed",
        "LinAlgError: x",
    ]
    table = report.failures(cells).set_index("model")
    assert table.loc["arima", "failed"] == 12 and table.loc["arima", "share"] == 0.5
    assert table.loc["arima", "example"] == "LinAlgError: x"
    assert table.loc["naive", "failed"] == 0


def _article(ticker, day):
    stored = pd.Timestamp(f"{day} 15:00", tz="UTC")
    return {"ticker": ticker, "source": "rss", "url_domain": "finance.yahoo.com",
            "published_at": stored, "inserted_at": stored, "sentiment": "positive",
            "score_pos": 0.8, "score_neg": 0.1, "score_neu": 0.1, "compound": 0.7}  # fmt: skip


def test_news_coverage_per_window_and_the_ablation_gate():
    dates = pd.bdate_range("2026-03-02", periods=20)
    windows = {"validation": list(dates[10:14]), "test": list(dates[16:18])}
    articles = pd.DataFrame(
        [_article("AAPL", d.date()) for d in dates[10:13]]  # 3 of 4 validation origins
        + [_article("AAPL", dates[2].date())]  # one day of history
    )
    coverage = report.news_coverage(articles, dates, ["AAPL", "MSFT"], windows)
    row = coverage.set_index(["window", "span", "ticker"])["share"]
    assert row[("validation", "origins", "AAPL")] == 0.75
    assert row[("validation", "origins", "MSFT")] == 0.0
    assert row[("validation", "origins", "all")] == 3 / 8
    assert row[("validation", "history", "AAPL")] == 1 / 10
    assert row[("test", "origins", "all")] == 0.0
    assert report.news_ablation_status(coverage) == "not enough data"

    articles = pd.DataFrame(
        [_article(t, d.date()) for t in ("AAPL", "MSFT") for d in dates[10:12]]
    )
    coverage = report.news_coverage(articles, dates, ["AAPL", "MSFT"], windows)
    assert report.news_ablation_status(coverage) == "testable"  # exactly half


def test_the_markdown_report_has_every_section():
    cells = _cells({"naive": -1.0, "good": 0.5})
    text = report.markdown({"window": "validation", "snapshot": "abc"}, cells)
    for section in ("## Headline", "## Per ticker", "## Per horizon", "## Failures"):
        assert section in text
    assert "- window: validation" in text and "descriptive" in text
