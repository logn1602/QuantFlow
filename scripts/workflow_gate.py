"""scripts/workflow_gate.py

Whether a run of daily_refresh.yml or trending.yml still has work to do. Each
workflow's first job runs it and starts the real job only on `run=true`.

daily_refresh.yml has two schedules: 22:30 UTC, and a backup at 03:17 UTC,
because GitHub says a scheduled run can be dropped when it is busy, and the
forward test of the news needs a fetch every trading day
(docs/evaluation.md, section 4). A run is needed unless the same workflow
already succeeded for the same session: a run that started at or after the
latest New York session's daily bar became final (17:00 New York,
DAILY_BAR_FINAL_AFTER in utils/trading_calendar.py) and concluded success.
The backup therefore runs only when the 22:30 run was dropped or failed, and
the 22:30 run only when the backup has not already done the session's work
(GitHub can start a scheduled run hours late). trending.yml, which starts when
any refresh completes, is gated the same way, so a refresh that had nothing
to do does not run the tier a second time for the session.

A run started by hand always runs. Weekends are skipped; holidays are not
(the standard library has no exchange calendar), which costs one run that
fetches nothing new.

Standard library only, so the gate job installs nothing, and it fails open:
if GitHub's API cannot be read, the run goes ahead. Two runs of a session are
safe (every write is a replace or ON CONFLICT DO NOTHING); a session never
fetched loses its news for good.

    python3 scripts/workflow_gate.py daily_refresh.yml

Reads GITHUB_EVENT_NAME, GITHUB_REPOSITORY and GH_TOKEN, and writes `run=true`
or `run=false` to $GITHUB_OUTPUT. Always exits 0.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

# The same as utils/trading_calendar.py, which this cannot import (the gate
# job installs nothing); tests/unit/test_workflow_gate.py checks they agree.
EXCHANGE_TZ = "America/New_York"
DAILY_BAR_FINAL_AFTER = time(17, 0)

# Successful runs read back: more than a week of both schedules and manual runs.
RUNS_READ = 30
API = "https://api.github.com"


def session_final_at(now: datetime) -> datetime:
    """When the latest weekday session's daily bar became final, at `now`.

    17:00 New York on the latest weekday on or before `now` whose 17:00 has
    passed: a run at 03:17 UTC Tuesday, still Monday evening in New York, is
    for Monday's session, and one on a Saturday or a Monday morning is for
    Friday's.
    """
    zone = ZoneInfo(EXCHANGE_TZ)
    day: date = now.astimezone(zone).date()
    while True:
        final_at = datetime.combine(day, DAILY_BAR_FINAL_AFTER, tzinfo=zone)
        if day.weekday() < 5 and final_at <= now:
            return final_at
        day -= timedelta(days=1)


def _started(run: Mapping[str, object]) -> datetime | None:
    started = run.get("run_started_at") or run.get("created_at")
    if not isinstance(started, str):
        return None
    return datetime.fromisoformat(started.replace("Z", "+00:00"))


def succeeded_since(runs: Iterable[Mapping[str, object]], since: datetime) -> bool:
    """Whether any run concluded success and started at or after `since`.

    A run this gate stopped also concluded success; it counts, since it was
    stopped only because an earlier run had succeeded for the same session.
    """
    for run in runs:
        started = _started(run)
        if run.get("conclusion") == "success" and started and started >= since:
            return True
    return False


def should_run(event: str, now: datetime, runs: Iterable[Mapping[str, object]]) -> bool:
    """True unless a run not started by hand has nothing left to do."""
    if event == "workflow_dispatch":
        return True
    return not succeeded_since(runs, session_final_at(now))


def read_runs(repository: str, workflow: str, token: str) -> list[dict]:
    """The workflow's latest successful runs, newest first, from GitHub."""
    url = (
        f"{API}/repos/{repository}/actions/workflows/{workflow}/runs"
        f"?status=success&per_page={RUNS_READ}"
    )
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return list(json.load(response)["workflow_runs"])


def decide(workflow: str, env: Mapping[str, str], now: datetime) -> tuple[bool, str]:
    """(run, why). Any failure to read or decide runs the workflow."""
    event = env.get("GITHUB_EVENT_NAME", "")
    if event == "workflow_dispatch":
        return True, "started by hand: runs"
    try:
        runs = read_runs(env["GITHUB_REPOSITORY"], workflow, env["GH_TOKEN"])
        final_at = session_final_at(now)
        if should_run(event, now, runs):
            return (
                True,
                f"no {workflow} run has succeeded since {final_at:%Y-%m-%d %H:%M %Z}: runs",
            )
        return False, (
            f"{workflow} already succeeded since {final_at:%Y-%m-%d %H:%M %Z}, "
            "this session's close: nothing to do"
        )
    except Exception as error:  # fail open: a skipped session loses its news
        return True, f"could not check earlier runs ({error!r}): runs anyway"


def main(argv: list[str]) -> int:
    workflow = argv[1] if len(argv) > 1 else "daily_refresh.yml"
    run, why = decide(workflow, os.environ, datetime.now(UTC))
    print(f"::notice title=Gate::{why}")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"run={'true' if run else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
