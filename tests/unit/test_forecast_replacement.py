"""tests/unit/test_forecast_replacement.py

How the daily run (pipelines/run_models.py) replaces the stored forecasts.

It used to delete every ticker's forecasts before fitting anything, then
write each model's as it finished. For the minutes the models took, the
dashboard found some of a ticker's forecasts or none, and kept what it found
for its cache lifetime, an hour. Now the run holds its forecasts until the
last model is done and replaces each ticker's in one transaction: a reader
finds the stored set or the new one, never neither, and a ticker the run made
nothing for keeps the set it had.
"""

import datetime as dt

import pandas as pd
import pytest

import quantflow.db.forecasts as db_forecasts
from quantflow.features import engineering
from quantflow.models import boosting, ensemble, statistical
from quantflow.pipelines import run_models

TICKERS = ["AAPL", "MSFT"]
MODELS = ["arima", "prophet", "xgboost", "lightgbm", "ensemble_stack"]
OLD_RUN = dt.datetime(2026, 11, 6, 23, 10)
OLD_DATES = pd.bdate_range("2026-11-09", periods=7)
NEW_DATES = pd.bdate_range("2026-11-10", periods=7)


def _forecast(ticker, model, dates=NEW_DATES):
    return pd.DataFrame(
        {
            "ticker": ticker,
            "model": model,
            "ds": dates,
            "predicted_close": 100.0,
            "lower_bound": 98.0,
            "upper_bound": 102.0,
        }
    )


@pytest.fixture
def stored(fake_db):
    """The previous run's forecasts: every model, both tickers."""
    fake_db.seed(
        "forecasts",
        [
            {
                "ticker": ticker,
                "model": model,
                "forecast_date": day.date(),
                "predicted_close": 99.0,
                "lower_bound": 97.0,
                "upper_bound": 101.0,
                "run_at": OLD_RUN,
            }
            for ticker in TICKERS
            for model in MODELS
            for day in OLD_DATES
        ],
    )
    return fake_db


def _found(db):
    """What a reader finds: each ticker's models with a stored forecast."""
    return {
        ticker: {r["model"] for r in db.tables["forecasts"] if r["ticker"] == ticker}
        for ticker in TICKERS
    }


@pytest.fixture
def steps(stored, monkeypatch):
    """The three model steps replaced by fakes that make a forecast for each
    model and ticker, storing it the way the real steps do, and record what a
    reader finds as each step starts. A ticker in skip gets nothing from a
    step, as when its fit fails."""
    found = []
    skip: dict[str, set] = {"statistical": set(), "boosting": set(), "ensemble": set()}

    def step(name, models):
        def run(tickers=None, news=None, **kwargs):
            found.append(_found(stored))
            held = kwargs.get("forecasts")
            save = held.save if held is not None else db_forecasts.save_forecasts
            for ticker in tickers or TICKERS:
                if ticker not in skip[name]:
                    for model in models:
                        save(_forecast(ticker, model))
            return {}

        return run

    monkeypatch.setattr(statistical, "run", step("statistical", ["arima", "prophet"]))
    monkeypatch.setattr(boosting, "run", step("boosting", ["xgboost", "lightgbm"]))
    monkeypatch.setattr(ensemble, "run", step("ensemble", ["ensemble_stack"]))
    monkeypatch.setattr(engineering, "load_news", lambda: None)
    return found, skip


def _stored_dates(db, ticker):
    return {r["forecast_date"] for r in db.tables["forecasts"] if r["ticker"] == ticker}


def test_a_reader_finds_every_tickers_forecasts_throughout_the_run(stored, steps):
    found, _ = steps
    _, failed = run_models.run(TICKERS)

    assert failed == []
    found.append(_found(stored))
    for seen in found:
        assert seen == {ticker: set(MODELS) for ticker in TICKERS}
    # Afterwards only the new run's forecasts are stored.
    for ticker in TICKERS:
        assert _stored_dates(stored, ticker) == {d.date() for d in NEW_DATES}


def test_each_ticker_is_replaced_in_one_transaction_after_every_model_ran(
    stored, steps, monkeypatch
):
    """Each replacement is one call holding every model's rows, after the
    last step, and nothing is deleted outside it."""
    found, _ = steps
    replaced = []
    real = db_forecasts.replace_forecasts

    def replace(ticker, rows):
        replaced.append((ticker, len(found), sorted({r["model"] for r in rows})))
        return real(ticker, rows)

    monkeypatch.setattr(run_models, "replace_forecasts", replace)
    _, failed = run_models.run(TICKERS)

    assert failed == []
    assert replaced == [(ticker, 3, sorted(MODELS)) for ticker in TICKERS]
    assert len(stored.deletes("forecasts")) == len(TICKERS)


def test_a_ticker_the_run_made_nothing_for_keeps_its_stored_forecasts(stored, steps):
    _, skip = steps
    for name in skip:
        skip[name].add("AAPL")

    run_models.run(TICKERS)

    assert _found(stored)["AAPL"] == set(MODELS)
    assert _stored_dates(stored, "AAPL") == {d.date() for d in OLD_DATES}
    assert _stored_dates(stored, "MSFT") == {d.date() for d in NEW_DATES}


def test_a_model_the_run_did_not_make_is_not_kept_from_the_old_run(stored, steps):
    """The new set is what the run made: an old forecast is never shown beside
    the new ones, as when the run deleted everything first."""
    _, skip = steps
    skip["boosting"].add("MSFT")

    run_models.run(TICKERS)

    assert _found(stored)["MSFT"] == {"arima", "prophet", "ensemble_stack"}
    assert _stored_dates(stored, "MSFT") == {d.date() for d in NEW_DATES}


def test_a_failed_replacement_fails_the_run_and_keeps_the_stored_set(stored, steps):
    stored.fail_next_insert = RuntimeError("connection lost")

    _, failed = run_models.run(TICKERS)

    assert failed == ["save_forecasts"]
    # AAPL's insert failed, so its delete rolled back; MSFT's went through.
    assert _stored_dates(stored, "AAPL") == {d.date() for d in OLD_DATES}
    assert _found(stored)["AAPL"] == set(MODELS)
    assert _stored_dates(stored, "MSFT") == {d.date() for d in NEW_DATES}


def test_no_clear_keeps_the_old_rows_beside_the_new_ones(stored, steps):
    """--no-clear stores each forecast as it is made, beside the old ones;
    readers take each model's latest run."""
    run_models.run(TICKERS, clear=False)

    for ticker in TICKERS:
        assert _stored_dates(stored, ticker) == {
            d.date() for d in OLD_DATES.union(NEW_DATES)
        }


def test_replacing_a_tickers_forecasts_with_none_is_refused(stored):
    with pytest.raises(ValueError, match="no forecast rows"):
        db_forecasts.replace_forecasts("AAPL", [])
    assert _found(stored)["AAPL"] == set(MODELS)


# ── The set the run holds ─────────────────────────────────────────────────────


def test_held_base_forecasts_have_the_shape_the_ensemble_reads():
    """ForecastSet.base stands in for load_base_forecasts: model,
    forecast_date as a date, and the prices rounded as they are stored."""
    held = db_forecasts.ForecastSet()
    for model in ("prophet", "arima"):
        frame = _forecast("AAPL", model)
        frame["predicted_close"] = 100.123456
        held.save(frame)
    held.save(_forecast("MSFT", "arima"))

    base = held.base("AAPL", ["arima", "prophet", "xgboost"])

    assert list(base.columns) == [
        "model", "forecast_date", "predicted_close", "lower_bound", "upper_bound"
    ]  # fmt: skip
    assert len(base) == 14
    assert list(base["model"][:2]) == ["arima", "prophet"]
    assert base["forecast_date"].iloc[0] == NEW_DATES[0].date()
    assert set(base["predicted_close"]) == {100.1235}


def test_a_model_held_twice_keeps_its_latest_forecast():
    """As readers take each model's latest stored run."""
    held = db_forecasts.ForecastSet()
    held.save(_forecast("AAPL", "arima", OLD_DATES))
    held.save(_forecast("AAPL", "arima", NEW_DATES))

    base = held.base("AAPL", ["arima"])
    assert list(base["forecast_date"]) == [d.date() for d in NEW_DATES]


# ── clear_forecasts: the laptop scheduler's path ──────────────────────────────


def test_clearing_named_models_keeps_every_other_models_forecasts(stored):
    """A scheduler job clears only the models it is about to rewrite: a job
    must never blank another model's forecasts (Known problem 4). No test
    reached this path before the full-diff review (2026-11-12)."""
    cleared = db_forecasts.clear_forecasts(["AAPL"], ["arima", "prophet"])
    assert cleared == 2 * len(OLD_DATES)
    found = _found(stored)
    assert found["AAPL"] == set(MODELS) - {"arima", "prophet"}
    assert found["MSFT"] == set(MODELS)


def test_clearing_without_models_clears_every_model_of_those_tickers_only(stored):
    cleared = db_forecasts.clear_forecasts(["MSFT"])
    assert cleared == len(MODELS) * len(OLD_DATES)
    found = _found(stored)
    assert found["AAPL"] == set(MODELS)
    assert "MSFT" not in found or found["MSFT"] == set()
