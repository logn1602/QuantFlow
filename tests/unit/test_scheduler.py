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
    _, chain_check = scheduler.SPLIT_GUARD_TIMES
    daily_bars_at = scheduler.DAILY_BARS_AT
    assert chain_check < time(16, 30)  # the forecasting job
    assert daily_bars_at == time(17, 5)  # DAILY_BAR_FINAL_AFTER + 5 min


def test_the_guard_runs_early_in_the_session_too():
    """On the day a split takes effect, the 15-minute indicator job and the
    hourly anomaly job read the mixed basis from the open until 16:20. A check
    at 09:45 pauses them before the session's first 15-min bar is final
    (09:47), so the new basis never mixes in."""
    open_check, _ = scheduler.SPLIT_GUARD_TIMES
    assert open_check == time(9, 45)


class _RecordingScheduler:
    """Stands in for BlockingScheduler: records each job, runs nothing."""

    def __init__(self, **kwargs):
        self.jobs = {}

    def add_job(self, func, trigger, id, **kwargs):
        self.jobs[id] = {"trigger": trigger, **kwargs}

    def start(self):
        pass


@pytest.fixture
def scheduled(monkeypatch):
    """The jobs start() registers, with config, the database and the jobs
    that run at startup faked out."""
    recorder = _RecordingScheduler()
    monkeypatch.setattr(scheduler, "BlockingScheduler", lambda **kwargs: recorder)
    monkeypatch.setattr(scheduler.config, "validate", lambda: True)
    monkeypatch.setattr(scheduler.config, "require_or_exit", lambda: None)
    monkeypatch.setattr(scheduler, "test_connection", lambda: True)
    monkeypatch.setattr(split_guard, "configured_threshold", lambda raw=None: 0.15)
    for job in ("run_yfinance_job", "run_split_guard_job", "run_indicators_job"):
        monkeypatch.setattr(scheduler, job, lambda: None)
    scheduler.start()
    return recorder.jobs


@pytest.mark.parametrize(
    "job", ["split_guard_0945", "split_guard_1620", "yfinance_daily"]
)
def test_the_guard_and_the_daily_bars_still_run_when_their_slot_was_missed(
    scheduled, job
):
    """APScheduler skips a run that starts more than a second late, so a laptop
    waking at 16:25 skipped the 16:20 guard and the model chain ran on the
    previous day's answer. Both jobs are safe to run late."""
    assert scheduled[job]["misfire_grace_time"] is None
    assert scheduled[job]["coalesce"] is True


def test_each_guard_run_is_scheduled_at_its_new_york_time(scheduled):
    for job, (hour, minute) in [
        ("split_guard_0945", (9, 45)),
        ("split_guard_1620", (16, 20)),
    ]:
        trigger = scheduled[job]["trigger"]
        assert f"hour='{hour}', minute='{minute}'" in str(trigger), trigger
        assert str(trigger.timezone) == "America/New_York"
