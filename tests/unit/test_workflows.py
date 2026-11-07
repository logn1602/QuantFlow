"""tests/unit/test_workflows.py

Checks on the GitHub Actions workflows and the Makefile's full-pipeline
target, read as text.

The workflows decide what runs in production and in CI, and nothing else
tests them: a step that is dropped or reordered still produces a green run.
These read the YAML as plain text rather than parsing it, so they need no
YAML dependency.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
DAILY_REFRESH = (WORKFLOWS / "daily_refresh.yml").read_text(encoding="utf-8")
CI = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")


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


def _pytest_markers(workflow):
    match = re.search(r'pytest[^\n]*?-m "([^"]+)"', workflow)
    return match.group(1) if match else None


def test_the_daily_refresh_runs_the_same_test_selection_as_ci():
    """A bare `pytest -q` also ran the eval suite, which CI deliberately
    excludes, so the two disagreed about what a passing run means."""
    assert _pytest_markers(CI) == "not eval and not integration"
    assert _pytest_markers(DAILY_REFRESH) == _pytest_markers(CI)


def test_ci_runs_on_pushes_to_every_phase_branch_prefix():
    match = re.search(r"push:\s*\n\s*branches:\s*\[([^\]]*)\]", CI)
    assert match, "no push branches list in ci.yml"
    branches = {b.strip().strip("'\"") for b in match.group(1).split(",")}
    assert {"main", "fix/**", "feat/**", "docs/**", "chore/**"} <= branches


def test_the_workflow_tracks_the_same_tickers_as_env_example():
    workflow = re.search(r"^\s*TICKERS:\s*(\S+)", DAILY_REFRESH, re.MULTILINE)
    example = re.search(
        r"^TICKERS=(\S+)",
        (ROOT / ".env.example").read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert workflow and example
    assert workflow.group(1) == example.group(1)


PRICE_READING_STAGES = (
    "python -m quantflow.features.indicators",
    "python -m quantflow.features.anomalies",
    "python -m quantflow.pipelines.run_models",
    "python -m quantflow.evaluation.backtest",
)


def test_the_split_guard_runs_after_the_fetch_and_before_any_stage_reads_prices():
    guard = "python -m quantflow.ingestion.split_guard"
    fetch, guard_at = _step_order(
        DAILY_REFRESH, "quantflow.ingestion.yfinance_fetcher --daily", guard
    )
    assert fetch < guard_at
    for stage, at in zip(
        PRICE_READING_STAGES,
        _step_order(DAILY_REFRESH, *PRICE_READING_STAGES),
        strict=True,
    ):
        assert guard_at < at, f"{stage} runs before the split guard"


def test_make_all_runs_the_split_guard_before_any_stage_reads_prices():
    """make all is the README's re-runnable full pipeline, run against the
    same database as the workflow, so it needs the same guard."""
    target = re.search(r"^all:\n((?:\t.*\n)+)", MAKEFILE, re.MULTILINE)
    assert target, "no all: target in the Makefile"
    seed, guard_at = _step_order(
        target.group(1),
        "python -m quantflow.pipelines.seed",
        "python -m quantflow.ingestion.split_guard",
    )
    assert seed < guard_at
    for stage, at in zip(
        PRICE_READING_STAGES,
        _step_order(target.group(1), *PRICE_READING_STAGES),
        strict=True,
    ):
        assert guard_at < at, f"{stage} runs before the split guard"


def _steps(workflow):
    """Each step's text keyed by its name, in workflow order."""
    parts = re.split(r"^      - name: (.+)$", workflow, flags=re.MULTILINE)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


STEPS = _steps(DAILY_REFRESH)
SENTIMENT = "Analyze news sentiment"


def _condition(step):
    match = re.search(r"^\s+if: (.+)$", STEPS[step], re.MULTILINE)
    return match.group(1) if match else None


def test_daily_refresh_runs_one_at_a_time_without_cancelling():
    """A manual run started during the scheduled one interleaved its writes
    with it. A running refresh is never cancelled; the newer one waits."""
    assert re.search(
        r"^concurrency:\n  group: daily-refresh\n  cancel-in-progress: false$",
        DAILY_REFRESH,
        re.MULTILINE,
    )


def test_a_failed_sentiment_step_fails_the_run_but_the_models_still_run():
    """FinBERT not loading fails the sentiment step, and so the run, but the
    models still train, on the sentiment already stored."""
    assert re.search(r"^\s+id: sentiment$", STEPS[SENTIMENT], re.MULTILINE)
    assert "continue-on-error" not in STEPS[SENTIMENT]
    assert _condition("Run all 5 forecasting models") == (
        "${{ !cancelled() && steps.sentiment.outcome != 'skipped' }}"
    )
    assert _condition("Run strategy backtest") == (
        "${{ !cancelled() && steps.models.outcome == 'success' }}"
    )


def test_a_failed_split_guard_still_stops_every_stage_after_it():
    """The models run after a failed sentiment step because that step ran, and
    it runs only if every step before it succeeded. A plain !cancelled() on
    the models would run them after a failed guard too."""
    names = list(STEPS)
    guard = names.index("Check stored prices against Yahoo (split guard)")
    for name in names[guard : names.index(SENTIMENT) + 1]:
        assert _condition(name) is None, name
        assert "continue-on-error" not in STEPS[name], name


def test_the_finbert_model_is_cached_by_the_revision_the_code_pins():
    from quantflow.features.sentiment import FINBERT_REVISION

    cache = STEPS["Cache the FinBERT model"]
    assert "uses: actions/cache@v4" in cache
    assert "path: ~/.cache/huggingface" in cache
    assert FINBERT_REVISION in cache
    names = list(STEPS)
    assert names.index("Cache the FinBERT model") < names.index(SENTIMENT)
