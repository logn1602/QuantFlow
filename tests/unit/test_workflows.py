"""tests/unit/test_workflows.py

Checks on the GitHub Actions workflows, read as text.

The workflows decide what runs in production and in CI, and nothing else
tests them: a step that is dropped or reordered still produces a green run.
These read the YAML as plain text rather than parsing it, so they need no
YAML dependency.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DAILY_REFRESH = (ROOT / ".github" / "workflows" / "daily_refresh.yml").read_text(
    encoding="utf-8"
)


def _step_order(workflow, *runs):
    """Positions of each `run:` command in the workflow text, in order given."""
    positions = []
    for command in runs:
        index = workflow.find(command)
        assert index != -1, f"missing step: {command}"
        positions.append(index)
    return positions


def test_daily_refresh_stores_daily_bars_before_any_stage_reads_prices():
    positions = _step_order(
        DAILY_REFRESH,
        "quantflow.ingestion.yfinance_fetcher --intraday",
        "quantflow.ingestion.yfinance_fetcher --daily",
        "python -m quantflow.features.indicators",
        "python -m quantflow.pipelines.run_models",
    )
    assert positions == sorted(positions)


def test_the_split_guard_runs_after_the_fetch_and_before_any_stage_reads_prices():
    positions = _step_order(
        DAILY_REFRESH,
        "quantflow.ingestion.yfinance_fetcher --daily",
        "python -m quantflow.ingestion.split_guard",
        "python -m quantflow.features.indicators",
    )
    assert positions == sorted(positions)
