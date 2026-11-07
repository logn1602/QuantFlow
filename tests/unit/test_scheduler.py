"""tests/unit/test_scheduler.py

Covers the split-guard gate in quantflow.pipelines.scheduler.

After a stock split, every stored row before it sits on the old share basis,
and anything computed from that mix is wrong and gets stored. The daily
workflow stops at its split-guard step; the scheduler has no steps, so it
remembers the guard's last answer and the jobs that read prices skip while it
is a break. These tests pin who pauses, what resumes, and that the daemon
never dies over it.
"""

import importlib
import logging
from datetime import time

import pytest
from sqlalchemy.exc import OperationalError

from quantflow.ingestion import split_guard, yfinance_fetcher
from quantflow.pipelines import scheduler

# Every job that reads raw_prices, and where its work function lives.
PRICE_JOBS = [
    ("run_indicators_job", "quantflow.features.indicators", "run"),
    ("run_anomaly_job", "quantflow.features.anomalies", "run"),
    ("run_forecasting_job", "quantflow.models.statistical", "run"),
    ("run_xgboost_job", "quantflow.models.boosting", "run"),
    ("run_ensemble_job", "quantflow.models.ensemble", "run"),
    ("run_backtest_job", "quantflow.evaluation.backtest", "run"),
]


@pytest.fixture(autouse=True)
def open_gate(monkeypatch):
    """Each test starts with no outstanding break and restores the module
    state afterwards, since the gate is a module global."""
    monkeypatch.setattr(scheduler, "_price_basis_error", None)
    monkeypatch.setattr(scheduler, "_clear_model_forecasts", lambda models: 0)


@pytest.fixture
def work(monkeypatch):
    """Replace each price job's work function with a recorder."""
    called = []
    for job, module, attr in PRICE_JOBS:
        target = importlib.import_module(module)
        monkeypatch.setattr(
            target, attr, lambda *a, job=job, **k: called.append(job) or {}
        )
    return called


def _guard_answers(monkeypatch, answer):
    """split_guard.run returns answer, or raises it if it is an exception."""

    def fake_run(*args, **kwargs):
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(split_guard, "run", fake_run)


def _run_price_jobs():
    for job, _, _ in PRICE_JOBS:
        getattr(scheduler, job)()


def test_a_split_pauses_every_job_that_reads_prices(monkeypatch, work, caplog):
    _guard_answers(monkeypatch, split_guard.PriceBasisError("disagree for AAPL"))
    scheduler.run_split_guard_job()  # must not raise: the daemon keeps running

    with caplog.at_level(logging.ERROR):
        _run_price_jobs()
    assert work == []
    assert caplog.text.count("skipped: stored prices are on more than one basis") == 6


def test_without_a_break_every_price_job_runs(monkeypatch, work):
    _guard_answers(monkeypatch, [])
    scheduler.run_split_guard_job()
    _run_price_jobs()
    assert work == [job for job, _, _ in PRICE_JOBS]


def test_a_passing_guard_resumes_the_price_jobs(monkeypatch, work):
    _guard_answers(monkeypatch, split_guard.PriceBasisError("disagree for AAPL"))
    scheduler.run_split_guard_job()
    _guard_answers(monkeypatch, [])  # the rows were rescaled
    scheduler.run_split_guard_job()

    _run_price_jobs()
    assert len(work) == len(PRICE_JOBS)


def test_a_pass_that_could_not_check_every_ticker_keeps_a_break_standing(
    monkeypatch, work
):
    _guard_answers(monkeypatch, split_guard.PriceBasisError("disagree for AAPL"))
    scheduler.run_split_guard_job()
    _guard_answers(monkeypatch, ["AAPL"])  # Yahoo returned nothing for AAPL
    scheduler.run_split_guard_job()

    _run_price_jobs()
    assert work == []


@pytest.mark.parametrize(
    "outstanding", [None, "stored prices are on more than one basis"]
)
def test_a_guard_failure_that_is_not_a_break_leaves_the_last_answer(
    monkeypatch, caplog, outstanding
):
    """A database blip neither pauses the price jobs nor clears a real break."""
    monkeypatch.setattr(scheduler, "_price_basis_error", outstanding)
    _guard_answers(monkeypatch, OperationalError("SELECT", {}, Exception("down")))

    scheduler.run_split_guard_job()  # must not raise
    assert scheduler._price_basis_error == outstanding
    assert "Split guard job failed: OperationalError" in caplog.text


def test_the_daily_bars_job_runs_the_guard_after_the_fetch(monkeypatch, work):
    calls = []
    monkeypatch.setattr(
        yfinance_fetcher,
        "fetch_historical",
        lambda tickers=None, period="2y": calls.append(("fetch", period)) or {},
    )

    def guard(*args, **kwargs):
        calls.append(("guard",))
        raise split_guard.PriceBasisError("disagree for AAPL")

    monkeypatch.setattr(split_guard, "run", guard)
    scheduler.run_daily_bars_job()  # must not raise

    assert calls == [("fetch", "5d"), ("guard",)]
    scheduler.run_indicators_job()
    assert work == []


def test_a_failed_daily_fetch_still_runs_the_guard(monkeypatch, caplog):
    def unreachable(tickers=None, period="2y"):
        raise ConnectionError("yahoo unreachable")

    ran = []
    monkeypatch.setattr(yfinance_fetcher, "fetch_historical", unreachable)
    monkeypatch.setattr(split_guard, "run", lambda *a, **k: ran.append(True) or [])

    scheduler.run_daily_bars_job()
    assert ran == [True]
    assert "Daily bars job failed: ConnectionError" in caplog.text


def test_the_guard_runs_before_the_model_chain_and_daily_bars_after_the_close():
    guard_at, daily_bars_at = scheduler.SPLIT_GUARD_AT, scheduler.DAILY_BARS_AT
    assert guard_at < time(16, 30)  # the forecasting job
    assert daily_bars_at == time(17, 5)  # DAILY_BAR_FINAL_AFTER + 5 min
