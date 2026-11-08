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


def _meta(naive=0.0):
    rest = (1.0 - naive) / 4
    return ensemble.NNLSMeta(np.array([rest, rest, rest, rest, naive]))


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
    forecast = ensemble.generate_ensemble_forecast("AAPL", _meta(), 97.0, LATEST_CLOSE)

    assert [pd.Timestamp(d) for d in forecast["ds"]] == list(SEVEN)
    assert np.allclose(forecast["predicted_close"], 100.0 + np.arange(7))


def test_naives_column_is_the_latest_close_with_no_band(read_tables):
    """Naive is not stored with the base forecasts: its forecast is the
    latest close, and its band that same price."""
    read_tables["forecasts"] = _base(dict.fromkeys(ensemble.BASE_MODELS, SEVEN))
    forecast = ensemble.generate_ensemble_forecast(
        "AAPL", _meta(naive=1.0), 97.0, LATEST_CLOSE
    )
    assert np.allclose(forecast["predicted_close"], 97.0)
    assert np.allclose(forecast["lower_bound"], 97.0)
    assert np.allclose(forecast["upper_bound"], 97.0)


def test_base_forecasts_from_an_earlier_run_are_refused_beside_todays_close(
    read_tables,
):
    """Naive's forecast is the latest close; the stored base forecasts must
    start the trading day after it. A failed base step leaves the previous
    run's forecasts in the table, whose first date is the latest close's own
    day: combined with it they would be published as today's forecast."""
    read_tables["forecasts"] = _base(dict.fromkeys(ensemble.BASE_MODELS, SEVEN))
    with pytest.raises(ensemble.ForecastAlignmentError, match="2026-11-23"):
        ensemble.generate_ensemble_forecast("AAPL", _meta(), 97.0, SEVEN[0])


def test_without_the_latest_close_naives_column_cannot_be_built(read_tables):
    read_tables["forecasts"] = _base(dict.fromkeys(ensemble.BASE_MODELS, SEVEN))
    with pytest.raises(ValueError, match="latest close"):
        ensemble.generate_ensemble_forecast("AAPL", _meta())


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
    stack = pd.DataFrame(
        {"actual": truth} | dict.fromkeys(ensemble.STACK_COLUMNS, truth)
    )
    saved = []
    monkeypatch.setattr(ensemble, "load_news", lambda: None)
    history = pd.bdate_range(end=LATEST_CLOSE, periods=5)
    monkeypatch.setattr(
        ensemble,
        "load_prices",
        lambda t: pd.DataFrame({"ds": history, "y": [93.0, 94.0, 95.0, 96.0, 97.0]}),
    )
    monkeypatch.setattr(ensemble, "load_features", lambda t, news=None: pd.DataFrame())
    monkeypatch.setattr(ensemble, "build_holdout_stack", lambda *a: stack.copy())
    monkeypatch.setattr(ensemble, "save_model_metrics", lambda *a, **k: None)
    monkeypatch.setattr(ensemble, "log_to_mlflow", lambda *a, **k: None)
    monkeypatch.setattr(ensemble, "save_forecasts", lambda df: saved.append(df) or 7)
    # All weight on naive: the published forecast is the newest close.
    monkeypatch.setattr(
        ensemble, "tune_and_train_meta",
        lambda stacked: (_meta(naive=1.0), {"coefficients": {}, "eval_days": 0,
                                           "ensemble_mape": None}),
    )  # fmt: skip

    with pytest.raises(ValueError, match="MSFT"):
        ensemble.run(["MSFT", "AAPL"])
    assert [df["ticker"].iloc[0] for df in saved] == ["AAPL"]
    assert np.allclose(saved[0]["predicted_close"], 97.0)
