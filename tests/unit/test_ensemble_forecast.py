"""tests/unit/test_ensemble_forecast.py

Covers the live 7-day ensemble forecast.

It pivoted the four base forecasts by date and dropped any date one model was
missing. With the boosters dated one day early (fixed separately), ARIMA's
first step met the boosters' second, and the published ensemble silently had
six dates. Base forecasts that do not cover the same seven dates are a broken
upstream run, so the ensemble now refuses to build on them.
"""

import numpy as np
import pandas as pd
import pytest

from quantflow.models import ensemble

LATEST_CLOSE = pd.Timestamp("2026-11-20")
SEVEN = pd.bdate_range("2026-11-23", periods=8).drop(pd.Timestamp("2026-11-26"))[:7]
SHIFTED = pd.bdate_range(LATEST_CLOSE, periods=7)  # starts on the known close


def _base(models_dates):
    rows = []
    for model, dates in models_dates.items():
        for i, day in enumerate(dates):
            rows.append(
                {
                    "model": model,
                    "forecast_date": day.date(),
                    "predicted_close": 100.0 + i,
                    "lower_bound": 98.0 + i,
                    "upper_bound": 102.0 + i,
                }
            )
    return pd.DataFrame(rows)


def _meta():
    return ensemble.NNLSMeta(np.array([0.25, 0.25, 0.25, 0.25]))


def test_base_forecasts_on_different_dates_fail_loudly(read_tables):
    read_tables["forecasts"] = _base(
        {"arima": SEVEN, "prophet": SEVEN, "xgboost": SHIFTED, "lightgbm": SHIFTED}
    )
    with pytest.raises(ValueError, match="same 7 dates"):
        ensemble.generate_ensemble_forecast("AAPL", _meta())


def test_a_missing_base_model_fails_loudly(read_tables):
    read_tables["forecasts"] = _base(
        {"arima": SEVEN, "prophet": SEVEN, "xgboost": SEVEN}
    )
    with pytest.raises(ValueError, match="lightgbm"):
        ensemble.generate_ensemble_forecast("AAPL", _meta())


def test_two_stored_runs_of_one_model_fail_as_misaligned_not_as_a_crash(read_tables):
    """A CLI run or `run_models --no-clear` can leave two runs of a model in
    the table. Pivoting them raised a bare ValueError that ensemble.run did
    not catch, so every later ticker lost its forecast."""
    base = _base(dict.fromkeys(ensemble.BASE_MODELS, SEVEN))
    read_tables["forecasts"] = pd.concat(
        [base, base[base["model"] == "arima"]], ignore_index=True
    )
    with pytest.raises(ensemble.ForecastAlignmentError, match="more than one"):
        ensemble.generate_ensemble_forecast("AAPL", _meta())


def test_aligned_base_forecasts_give_one_ensemble_row_per_date(read_tables):
    read_tables["forecasts"] = _base(dict.fromkeys(ensemble.BASE_MODELS, SEVEN))
    forecast = ensemble.generate_ensemble_forecast("AAPL", _meta())

    assert [pd.Timestamp(d) for d in forecast["ds"]] == list(SEVEN)
    assert np.allclose(forecast["predicted_close"], 100.0 + np.arange(7))


def test_one_misaligned_ticker_fails_the_run_after_the_others_finish(
    read_tables, monkeypatch
):
    by_ticker = {
        "MSFT": _base(
            {"arima": SEVEN, "prophet": SEVEN, "xgboost": SHIFTED, "lightgbm": SHIFTED}
        ),
        "AAPL": _base(dict.fromkeys(ensemble.BASE_MODELS, SEVEN)),
    }
    read_tables["forecasts"] = lambda params: by_ticker[params["ticker"]]

    rng = np.random.default_rng(1)
    truth = 100 + np.cumsum(rng.normal(0, 1, 30))
    stack = pd.DataFrame({"actual": truth} | dict.fromkeys(ensemble.BASE_MODELS, truth))
    saved = []
    monkeypatch.setattr(ensemble, "collect_holdout_stacks", lambda t: stack.copy())
    monkeypatch.setattr(ensemble, "save_model_metrics", lambda *a, **k: None)
    monkeypatch.setattr(ensemble, "log_to_mlflow", lambda *a, **k: None)
    monkeypatch.setattr(
        ensemble, "save_forecasts", lambda df: saved.append(df["ticker"].iloc[0]) or 7
    )

    with pytest.raises(ValueError, match="MSFT"):
        ensemble.run(["MSFT", "AAPL"])
    assert saved == ["AAPL"]
