"""tests/unit/test_scheduler.py

Covers the split-guard gate in quantflow.pipelines.scheduler.

After a stock split, every stored row before it sits on the old share basis,
and anything computed from that mix is wrong and gets stored. The daily
workflow stops at its split-guard step; the scheduler has no steps, so it
remembers the guard's last answer and the jobs that read prices skip while it
is a break, or while the guard could not check every ticker and its last full
pass is older than the previous trading day. These tests pin who pauses, what
resumes, and that the daemon never dies over it.
"""

import importlib
import logging
from datetime import time

import pandas as pd
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
    ("run_backtest_job", "quantflow.pipelines.backtest", "run"),
]


# Monday 2026-11-09 09:45 New York, and passes on the trading days before it.
MONDAY_0945 = pd.Timestamp("2026-11-09 09:45", tz="America/New_York").tz_convert("UTC")
FRIDAY_PASS = pd.Timestamp("2026-11-06 17:10", tz="America/New_York").tz_convert("UTC")
THURSDAY_PASS = pd.Timestamp("2026-11-05 17:10", tz="America/New_York").tz_convert(
    "UTC"
)


@pytest.fixture(autouse=True)
def open_gate(monkeypatch):
    """Each test starts with no outstanding break and no full pass yet, at a
    fixed time, and restores the module state afterwards, since the gate is
    module globals."""
    monkeypatch.setattr(scheduler, "_price_basis_error", None)
    monkeypatch.setattr(scheduler, "_last_full_pass", None)
    monkeypatch.setattr(scheduler, "_now", lambda: MONDAY_0945)
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
    # Worded for both checks: the jump check also fails on a real move that
    # is waiting to be acknowledged.
    assert (
        caplog.text.count("skipped: the split guard found a possible unadjusted split")
        == 6
    )


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


# A guard run that could not check every ticker, three ways.
INCOMPLETE = [
    pytest.param(["AAPL"], id="yahoo-returned-nothing"),
    pytest.param(split_guard.GuardError("could not check AAPL"), id="a-check-raised"),
    pytest.param(
        OperationalError("SELECT", {}, Exception("down")), id="the-run-failed"
    ),
]


@pytest.mark.parametrize("answer", INCOMPLETE)
def test_an_incomplete_check_reuses_a_pass_from_the_previous_trading_day(
    monkeypatch, work, answer
):
    """Friday's full pass still stands on Monday morning, so a Yahoo or
    database hiccup does not stop a day of models."""
    monkeypatch.setattr(scheduler, "_last_full_pass", FRIDAY_PASS)
    _guard_answers(monkeypatch, answer)
    scheduler.run_split_guard_job()  # must not raise

    _run_price_jobs()
    assert work == [job for job, _, _ in PRICE_JOBS]


@pytest.mark.parametrize("answer", INCOMPLETE)
def test_an_incomplete_check_pauses_when_the_last_pass_is_older(
    monkeypatch, work, caplog, answer
):
    """A guard that could not check a ticker is not a pass. An older pass
    used to be kept however old it was, so the price jobs ran on prices no
    one had checked for days."""
    monkeypatch.setattr(scheduler, "_last_full_pass", THURSDAY_PASS)
    _guard_answers(monkeypatch, answer)
    scheduler.run_split_guard_job()

    with caplog.at_level(logging.ERROR):
        _run_price_jobs()
    assert work == []
    assert "older than the previous trading day (2026-11-06)" in caplog.text


@pytest.mark.parametrize("answer", INCOMPLETE)
def test_an_incomplete_check_pauses_when_there_was_no_full_pass_since_startup(
    monkeypatch, work, answer
):
    _guard_answers(monkeypatch, answer)
    scheduler.run_split_guard_job()

    _run_price_jobs()
    assert work == []


@pytest.mark.parametrize("answer", INCOMPLETE)
def test_an_incomplete_check_keeps_a_break_standing(monkeypatch, work, answer):
    monkeypatch.setattr(scheduler, "_last_full_pass", FRIDAY_PASS)
    _guard_answers(monkeypatch, split_guard.PriceBasisError("disagree for AAPL"))
    scheduler.run_split_guard_job()
    _guard_answers(monkeypatch, answer)
    scheduler.run_split_guard_job()

    _run_price_jobs()
    assert work == []


@pytest.mark.parametrize("broken", ["no-reason", "missing", "a-directory"])
def test_an_acknowledged_moves_list_broken_while_running_pauses_the_price_jobs(
    monkeypatch, tmp_path, work, caplog, broken
):
    """The guard reads the list again on every run, so an acknowledgement
    takes effect without a restart. A list it cannot read must then pause the
    price jobs, as it stops the daemon at startup, not pass for a hiccup:
    until it is fixed neither check runs, and reusing Friday's pass would let
    a split through."""
    path = tmp_path / "split_jump_acknowledged.txt"
    if broken == "no-reason":
        path.write_text("AAPL:2026-05-11\n", encoding="utf-8")
    elif broken == "a-directory":
        path.mkdir()
    monkeypatch.setattr(scheduler.config, "SPLIT_JUMP_ACKNOWLEDGED_FILE", path)
    monkeypatch.setattr(split_guard, "configured_threshold", lambda raw=None: 0.15)
    monkeypatch.setattr(scheduler, "_last_full_pass", FRIDAY_PASS)

    scheduler.run_split_guard_job()  # the real guard, which stops at the list

    with caplog.at_level(logging.ERROR):
        _run_price_jobs()
    assert work == []
    assert "split_jump_acknowledged.txt" in caplog.text


def test_a_failed_run_is_logged_with_its_error(monkeypatch, caplog):
    monkeypatch.setattr(scheduler, "_last_full_pass", FRIDAY_PASS)
    _guard_answers(monkeypatch, OperationalError("SELECT", {}, Exception("down")))
    scheduler.run_split_guard_job()  # must not raise
    assert "Split guard job failed: OperationalError" in caplog.text


def test_a_full_pass_is_recorded_and_resumes_the_jobs(monkeypatch, work):
    _guard_answers(monkeypatch, ["AAPL"])
    scheduler.run_split_guard_job()  # no full pass since startup: paused
    _guard_answers(monkeypatch, [])
    scheduler.run_split_guard_job()

    _run_price_jobs()
    assert len(work) == len(PRICE_JOBS)
    assert scheduler._last_full_pass == MONDAY_0945


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


def test_the_scheduler_exits_at_startup_on_a_bad_acknowledged_moves_list(
    monkeypatch, tmp_path
):
    """Like a bad threshold: a daemon started with a list it cannot read
    should stop at once, not fail every guard run and pause its price jobs."""
    bad = tmp_path / "split_jump_acknowledged.txt"
    bad.write_text("AAPL:2026-05-11\n", encoding="utf-8")  # no reason given
    monkeypatch.setattr(scheduler.config, "SPLIT_JUMP_ACKNOWLEDGED_FILE", bad)
    monkeypatch.setattr(scheduler.config, "validate", lambda: True)
    monkeypatch.setattr(scheduler.config, "require_or_exit", lambda: None)
    monkeypatch.setattr(split_guard, "configured_threshold", lambda raw=None: 0.15)
    monkeypatch.setattr(
        scheduler, "test_connection", lambda: pytest.fail("connected before the list")
    )

    with pytest.raises(SystemExit) as exited:
        scheduler.start()
    assert exited.value.code == 1


def test_each_guard_run_is_scheduled_at_its_new_york_time(scheduled):
    for job, (hour, minute) in [
        ("split_guard_0945", (9, 45)),
        ("split_guard_1620", (16, 20)),
    ]:
        trigger = scheduled[job]["trigger"]
        assert f"hour='{hour}', minute='{minute}'" in str(trigger), trigger
        assert str(trigger.timezone) == "America/New_York"
