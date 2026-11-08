"""tests/unit/test_decision.py

The Phase 3 keep-or-drop rule (docs/evaluation.md, section 16), on hand-built
cells: what keeps, what drops, and what is refused outright.
"""

import numpy as np
import pandas as pd
import pytest

from quantflow.evaluation import decision

TICKERS = ["AAPL", "AMZN", "GOOGL", "JPM", "META", "MSFT", "NVDA", "TSLA"]
ORIGINS = pd.to_datetime(["2025-07-01", "2025-07-08", "2025-07-15"])


def _cells(errors: dict[str, float], model="xgboost", window="validation"):
    """Every ticker, origin and day with actual 100 and a forecast `errors`
    percent too high for that ticker (default 2%)."""
    rows = [
        {"ticker": t, "window": window, "role": "scored", "model": model,
         "origin": o, "h": h, "origin_close": 100.0, "actual": 100.0,
         "yhat": 100.0 + errors.get(t, 2.0), "yhat_lower": 98.0,
         "yhat_upper": 102.0, "status": "ok", "error": "", "seconds": 0.1}
        for t in TICKERS for o in ORIGINS for h in range(1, 8)
    ]  # fmt: skip
    return pd.DataFrame(rows)


def _fail(cells, ticker, n):
    """Fail the first n cells of a ticker."""
    cells = cells.copy()
    index = cells.index[cells["ticker"] == ticker][:n]
    cells.loc[index, ["status", "yhat"]] = ["failed", np.nan]
    return cells


def test_better_on_every_ticker_is_kept():
    result = decision.compare(
        _cells({}), _cells(dict.fromkeys(TICKERS, 1.0)), "xgboost"
    )
    assert result.keep
    assert result.mape_incumbent == pytest.approx(2.0)
    assert result.mape_candidate == pytest.approx(1.0)
    assert result.tickers_better == 8 and result.cells == 8 * 3 * 7


def test_lower_overall_but_on_too_few_tickers_drops():
    """4 tickers much better and 4 a little worse: the pooled MAPE falls,
    but a change must help most tickers."""
    errors = {t: (0.5 if i < 4 else 2.1) for i, t in enumerate(TICKERS)}
    result = decision.compare(_cells({}), _cells(errors), "xgboost")
    assert result.mape_candidate < result.mape_incumbent
    assert result.tickers_better == 4
    assert not result.keep and "4 of 8" in result.reason


def test_five_of_eight_is_enough():
    errors = {t: (1.0 if i < 5 else 2.1) for i, t in enumerate(TICKERS)}
    assert decision.compare(_cells({}), _cells(errors), "xgboost").keep


def test_a_tie_drops():
    result = decision.compare(_cells({}), _cells({}), "xgboost")
    assert not result.keep and "not below" in result.reason


def test_higher_overall_drops_even_if_most_tickers_improve():
    errors = {t: (1.9 if i < 7 else 9.0) for i, t in enumerate(TICKERS)}
    result = decision.compare(_cells({}), _cells(errors), "xgboost")
    assert result.tickers_better == 7
    assert not result.keep


def test_more_failed_cells_than_the_incumbent_drops():
    better = _fail(_cells(dict.fromkeys(TICKERS, 1.0)), "AAPL", 1)
    result = decision.compare(_cells({}), better, "xgboost")
    assert result.mape_candidate < result.mape_incumbent
    assert not result.keep and "fails 1 cells" in result.reason


def test_failing_over_the_limit_drops():
    candidate = _cells(dict.fromkeys(TICKERS, 1.0))
    for t in TICKERS[:2]:
        candidate = _fail(candidate, t, 21)  # 42 of 168 cells, 25%
    result = decision.compare(_cells({}), candidate, "xgboost")
    assert not result.keep and "limit" in result.reason


def test_cells_only_one_version_forecast_are_left_out_of_both():
    """The incumbent failed AAPL's first cell: that cell leaves both sides,
    so neither is scored on an easier subset."""
    incumbent = _fail(_cells({}), "AAPL", 1)
    candidate = _cells({})
    candidate.loc[candidate.index[0], "yhat"] = 150.0  # AAPL's first cell, way off
    result = decision.compare(incumbent, candidate, "xgboost")
    assert result.cells == 8 * 3 * 7 - 1
    assert result.mape_candidate == pytest.approx(result.mape_incumbent)


def test_the_test_dates_are_never_used_to_choose():
    with pytest.raises(ValueError, match="D13"):
        decision.compare(_cells({}), _cells({}, window="test"), "xgboost")


def test_runs_over_different_cells_are_refused():
    candidate = _cells({})
    candidate = candidate[candidate["h"] < 7]
    with pytest.raises(ValueError, match="same xgboost cells"):
        decision.compare(_cells({}), candidate, "xgboost")


def test_a_model_missing_from_a_run_is_refused():
    with pytest.raises(ValueError, match="no scored lightgbm"):
        decision.compare(_cells({}), _cells({}), "lightgbm")


def test_warmup_cells_are_not_compared():
    incumbent = _cells({})
    warm = incumbent.assign(role="warmup", origin=pd.Timestamp("2025-06-01"))
    result = decision.compare(pd.concat([incumbent, warm]), _cells({}), "xgboost")
    assert result.cells == 8 * 3 * 7


def test_the_paired_test_and_the_tables_are_reported():
    result = decision.compare(
        _cells({}), _cells(dict.fromkeys(TICKERS, 1.0)), "xgboost"
    )
    assert result.paired_test["mean"] == pytest.approx(-1.0)
    assert list(result.per_ticker["ticker"]) == TICKERS
    assert list(result.per_horizon["h"]) == list(range(1, 8))
    row = result.per_ticker.iloc[0]
    assert row["coverage_incumbent"] == pytest.approx(100.0)
    assert row["coverage_candidate"] == pytest.approx(100.0)
    assert decision.summary_row(result)["decision"] == "keep"
