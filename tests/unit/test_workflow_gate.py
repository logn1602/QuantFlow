"""tests/unit/test_workflow_gate.py

scripts/workflow_gate.py, which decides whether a run of daily_refresh.yml or
trending.yml still has work to do. GitHub can drop a scheduled run entirely,
so the refresh has a backup schedule; the gate makes whichever run of a
session comes second stop once the other has succeeded, and the trending run
that follows it stop too.
"""

import importlib.util
import io
import json
import urllib.error
from datetime import UTC, datetime
from pathlib import Path

import pytest

from quantflow.utils.trading_calendar import DAILY_BAR_FINAL_AFTER, EXCHANGE_TZ

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "workflow_gate", ROOT / "scripts" / "workflow_gate.py"
)
assert _spec and _spec.loader
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def _utc(text):
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def _run(started, conclusion="success", event="schedule"):
    return {"run_started_at": started, "conclusion": conclusion, "event": event}


# The refresh's runs as GitHub listed them on 2026-11-16, newest first. The
# 22:30 UTC runs started about four hours late, after midnight UTC.
RUNS_SEEN_2026_11_16 = [
    _run("2026-11-14T02:31:31Z"),  # Friday 11-13's session
    _run("2026-11-13T02:43:12Z"),  # Thursday 11-12's
    _run("2026-11-12T21:53:11Z", event="workflow_dispatch"),  # before the close
    _run("2026-11-12T03:10:13Z"),  # Wednesday 11-11's
]


def test_the_gate_uses_the_calendars_close_and_zone():
    """The gate cannot import the package (its job installs nothing), so it
    keeps its own copy of when a session's daily bar is final."""
    assert gate.EXCHANGE_TZ == EXCHANGE_TZ
    assert gate.DAILY_BAR_FINAL_AFTER == DAILY_BAR_FINAL_AFTER


@pytest.mark.parametrize(
    ("now", "final_at"),
    [
        # The 22:30 run, the same evening (EST: 17:00 is 22:00 UTC).
        ("2026-11-16 22:30", "2026-11-16 22:00"),
        # Exactly at the close's finality.
        ("2026-11-16 22:00", "2026-11-16 22:00"),
        # The backup, next morning UTC, still Monday evening in New York.
        ("2026-11-17 03:17", "2026-11-16 22:00"),
        # A late run after midnight New York (05:30 UTC is 00:30 EST).
        ("2026-11-17 05:30", "2026-11-16 22:00"),
        # Saturday's backup is for Friday 2026-11-13.
        ("2026-11-14 03:17", "2026-11-13 22:00"),
        # Sunday, and Monday before the close, are still Friday's.
        ("2026-11-15 18:00", "2026-11-13 22:00"),
        ("2026-11-16 15:00", "2026-11-13 22:00"),
        # Summer (EDT: 17:00 is 21:00 UTC).
        ("2026-07-14 03:17", "2026-07-13 21:00"),
        ("2026-07-13 21:30", "2026-07-13 21:00"),
    ],
)
def test_a_run_is_for_the_latest_session_whose_daily_bar_is_final(now, final_at):
    assert gate.session_final_at(_utc(now)) == _utc(final_at)


def test_saturdays_backup_stops_after_fridays_late_run_succeeded():
    """Friday 2026-11-13's 22:30 run started at 02:31 UTC Saturday; a backup
    at 03:17 would have found it and stopped."""
    now = _utc("2026-11-14 03:17")
    assert not gate.should_run("schedule", now, RUNS_SEEN_2026_11_16)


def test_saturdays_backup_refreshes_a_dropped_fridays_run():
    """Had GitHub dropped that run, nothing would have succeeded since 17:00
    New York on Friday."""
    now = _utc("2026-11-14 03:17")
    assert gate.should_run("schedule", now, RUNS_SEEN_2026_11_16[1:])


def test_the_backup_stops_when_the_22_30_run_succeeded():
    runs = [_run("2026-11-17T02:31:00Z"), *RUNS_SEEN_2026_11_16]
    assert not gate.should_run("schedule", _utc("2026-11-17 03:20"), runs)


def test_a_22_30_run_started_after_the_backup_succeeded_stops():
    """GitHub can start the 22:30 run hours late, after the backup."""
    runs = [_run("2026-11-17T03:20:00Z"), *RUNS_SEEN_2026_11_16]
    assert not gate.should_run("schedule", _utc("2026-11-17 07:40"), runs)


def test_the_backup_runs_again_after_a_failed_run():
    runs = [_run("2026-11-17T02:31:00Z", conclusion="failure"), *RUNS_SEEN_2026_11_16]
    assert gate.should_run("schedule", _utc("2026-11-17 03:20"), runs)


def test_a_run_before_the_close_does_not_count_for_the_session():
    """A manual run at 21:53 UTC on 2026-11-12, before 17:00 New York, stored
    no daily bar for that day: the evening's run still refreshes."""
    runs = [_run("2026-11-12T21:53:11Z", event="workflow_dispatch")]
    assert gate.should_run("schedule", _utc("2026-11-13 02:43"), runs)


def test_a_manual_run_after_the_close_counts_for_the_session():
    runs = [_run("2026-11-16T22:10:00Z", event="workflow_dispatch")]
    assert not gate.should_run("schedule", _utc("2026-11-17 03:17"), runs)


def test_a_run_started_by_hand_always_runs():
    runs = [_run("2026-11-17T02:31:00Z")]
    now = _utc("2026-11-17 03:20")
    assert gate.should_run("workflow_dispatch", now, runs)
    assert gate.decide("trending.yml", {"GITHUB_EVENT_NAME": "workflow_dispatch"}, now)[
        0
    ]


def test_trending_runs_once_a_session_after_any_refresh():
    """Started by the backup's refresh, which had nothing to do, the tier
    stops when it already succeeded after the 22:30 refresh; after a failed
    tier run it runs again."""
    now = _utc("2026-11-17 03:25")
    done = [_run("2026-11-17T02:37:00Z", event="workflow_run")]
    failed = [_run("2026-11-17T02:37:00Z", conclusion="failure", event="workflow_run")]
    assert not gate.should_run("workflow_run", now, done)
    assert gate.should_run("workflow_run", now, failed)


def test_a_re_run_counts_from_when_it_started():
    """GitHub's run_started_at is the latest attempt's start: a run created
    before the close and re-run after it stored the session's bar."""
    run = _run("2026-11-16T23:00:00Z") | {"created_at": "2026-11-16T20:00:00Z"}
    assert not gate.should_run("schedule", _utc("2026-11-17 03:17"), [run])


def _env(**extra):
    return {
        "GITHUB_EVENT_NAME": "schedule",
        "GITHUB_REPOSITORY": "owner/repo",
        "GH_TOKEN": "token",
        **extra,
    }


def test_the_gate_fails_open_when_github_cannot_be_read(monkeypatch):
    """A session never fetched loses its news for good; two runs of one
    session are safe. So an API error runs the workflow."""

    def unreachable(*_args):
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(gate, "read_runs", unreachable)
    run, why = gate.decide("daily_refresh.yml", _env(), _utc("2026-11-17 03:17"))
    assert run
    assert "could not check" in why


def test_the_gate_fails_open_without_a_token():
    env = {"GITHUB_EVENT_NAME": "schedule"}
    run, _ = gate.decide("daily_refresh.yml", env, _utc("2026-11-17 03:17"))
    assert run


def test_the_gate_reads_this_workflows_successful_runs(monkeypatch):
    seen = {}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["auth"] = request.get_header("Authorization")
        seen["timeout"] = timeout
        body = {"workflow_runs": [_run("2026-11-17T02:31:00Z")]}
        return Response(json.dumps(body).encode())

    monkeypatch.setattr(gate.urllib.request, "urlopen", urlopen)
    run, why = gate.decide("daily_refresh.yml", _env(), _utc("2026-11-17 03:17"))
    assert not run
    assert "nothing to do" in why
    assert seen["url"] == (
        "https://api.github.com/repos/owner/repo/actions/workflows/"
        "daily_refresh.yml/runs?status=success&per_page=30"
    )
    assert seen["auth"] == "Bearer token"
    assert seen["timeout"] == 30


@pytest.mark.parametrize(
    ("runs", "written"),
    [([], "run=true"), ([_run("2026-11-17T02:31:00Z")], "run=false")],
)
def test_the_gate_writes_its_answer_for_the_next_job_and_exits_0(
    monkeypatch, tmp_path, runs, written
):
    output = tmp_path / "github_output"
    monkeypatch.setattr(gate, "read_runs", lambda *_args: runs)
    for name, value in _env(GITHUB_OUTPUT=str(output)).items():
        monkeypatch.setenv(name, value)

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return _utc("2026-11-17 03:17")

    monkeypatch.setattr(gate, "datetime", Frozen)
    assert gate.main(["workflow_gate.py", "daily_refresh.yml"]) == 0
    assert output.read_text(encoding="utf-8") == f"{written}\n"
