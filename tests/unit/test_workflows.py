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

import pytest

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


def test_the_daily_refresh_runs_cis_test_selection_less_the_trending_tests():
    """A bare `pytest -q` also ran the eval suite, which CI deliberately
    excludes, so the two disagreed about what a passing run means. The one
    difference now: the trending tier's tests, which CI and trending.yml run,
    are left out, so one can never stop the core's data steps on a
    forward-test day (docs/trending.md, section 12)."""
    assert _pytest_markers(CI) == "not eval and not integration"
    assert _pytest_markers(DAILY_REFRESH) == f"{_pytest_markers(CI)} and not trending"


LOCKED_INSTALL = "uv sync --locked --extra dev"


def test_the_daily_refresh_installs_the_versions_in_uv_lock_as_ci_does():
    """The evaluation ran on uv.lock's versions. requirements.txt sets floors,
    so pip installed whatever was newest on the runner, and a different
    library version can give a different forecast from the same data.
    --locked fails the install, in CI first, if uv.lock no longer matches
    pyproject.toml."""
    assert "pip install -r" not in DAILY_REFRESH
    assert LOCKED_INSTALL in DAILY_REFRESH
    assert CI.count("uv sync") == CI.count(LOCKED_INSTALL) > 0
    # Each command runs in the locked environment, not the runner's Python.
    commands = re.findall(r"(\S+ \S+ )?(python -m \S+|pytest -m)", DAILY_REFRESH)
    assert commands, "no python or pytest commands found"
    for prefix, command in commands:
        assert prefix == "uv run ", command


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
    "python -m quantflow.pipelines.backtest",
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


def test_the_daily_refresh_no_longer_fetches_newsapi():
    """NewsAPI's free plan is for development and testing only; without the
    key the sentiment stage reads Yahoo's RSS feeds alone."""
    assert "NEWS_API_KEY" not in DAILY_REFRESH


AUDIT_WORKFLOW = WORKFLOWS / "dependency_audit.yml"


def test_the_daily_refresh_does_not_audit_dependencies():
    """A red daily refresh means a data problem, nothing else: the audit runs
    in its own workflow. In the refresh, a failed audit also failed the job,
    and actions/cache saves the FinBERT model only when the whole job
    succeeds."""
    assert "pip-audit" not in DAILY_REFRESH


def test_the_audit_workflow_audits_uv_lock_weekly_and_on_lock_changes():
    """A red audit means a dependency needs a security update. It runs weekly,
    for advisories published against versions already locked, and on every
    pull request that changes uv.lock, before the change merges."""
    audit = AUDIT_WORKFLOW.read_text(encoding="utf-8")
    cron = re.search(r"^  schedule:\n    - cron: '([^']+)'", audit, re.MULTILINE)
    assert cron, "no schedule"
    assert re.fullmatch(r"\d+ \d+ \* \* [0-6]", cron.group(1)), "not weekly"
    paths = re.search(
        r"^  pull_request:\n    paths:\n((?:      - .+\n)+)", audit, re.MULTILINE
    )
    assert paths, "no pull_request paths"
    assert "      - uv.lock\n" in paths.group(1)
    # The exact versions in uv.lock, nothing installed or resolved for it.
    assert "uv export --quiet --locked --extra dev" in audit
    assert "uvx pip-audit --requirement" in audit
    # pip-audit exits 1 on an advisory, which must fail the run.
    assert "continue-on-error" not in audit
    assert "secrets." not in audit
    assert re.search(r"^permissions:\n  contents: read$", audit, re.MULTILINE)


def _make_targets():
    phony = re.search(r"^\.PHONY:(.+)$", MAKEFILE, re.MULTILINE)
    assert phony, "no .PHONY line in the Makefile"
    return phony.group(1).split()


def test_make_help_lists_every_target():
    """make help is the list people read; a target missing from it is a
    target nobody finds."""
    helped = set(re.findall(r'@echo "  make (\S+)', MAKEFILE))
    targets = set(_make_targets()) - {"help"}
    assert targets <= helped, sorted(targets - helped)
    for target in targets:
        assert re.search(rf"^{re.escape(target)}:", MAKEFILE, re.MULTILINE), target


@pytest.mark.trending
def test_make_trending_runs_one_trending_run_with_an_optional_cap():
    body = MAKEFILE.split("\ntrending:\n", 1)[1].split("\n\n", 1)[0]
    assert "python -m quantflow.pipelines.trending" in body
    assert "$(if $(CAP),--cap $(CAP),)" in body


# ── The trending tier's workflow (docs/trending.md, section 12) ──────────────

TRENDING = (WORKFLOWS / "trending.yml").read_text(encoding="utf-8")


def _workflow_name(workflow):
    match = re.search(r"^name: (.+)$", workflow, re.MULTILINE)
    assert match, "no workflow name"
    return match.group(1).strip()


@pytest.mark.trending
def test_trending_starts_when_the_daily_refresh_completes_or_by_hand():
    """After the refresh, whatever its result, and never beside it; no
    schedule of its own, so it stops when the refresh's schedule does."""
    on = TRENDING.split("\non:\n", 1)[1].split("\n\n# ", 1)[0]
    assert re.search(r"^  workflow_run:\n", on, re.MULTILINE)
    workflows = re.search(r"^    workflows: \[\"([^\"]+)\"\]$", on, re.MULTILINE)
    assert workflows and workflows.group(1) == _workflow_name(DAILY_REFRESH)
    assert re.search(r"^    types: \[completed\]$", on, re.MULTILINE)
    assert re.search(r"^    branches: \[main\]$", on, re.MULTILINE)
    assert re.search(r"^  workflow_dispatch:$", on, re.MULTILINE)
    for trigger in ("schedule:", "pull_request", "push:"):
        assert trigger not in on, trigger


@pytest.mark.trending
def test_trending_has_its_own_timeout_and_concurrency_group():
    assert re.search(r"^    timeout-minutes: 30$", TRENDING, re.MULTILINE)
    group = re.search(r"^  group: (\S+)$", TRENDING, re.MULTILINE)
    daily_group = re.search(r"^  group: (\S+)$", DAILY_REFRESH, re.MULTILINE)
    assert group and daily_group and group.group(1) != daily_group.group(1)
    assert "cancel-in-progress: false" in TRENDING
    assert re.search(r"^permissions:\n  contents: read$", TRENDING, re.MULTILINE)


@pytest.mark.trending
def test_trending_installs_locked_runs_every_offline_test_then_the_tier():
    positions = _step_order(
        TRENDING,
        LOCKED_INSTALL,
        'uv run pytest -m "not eval and not integration"',
        "uv run python -m quantflow.pipelines.trending",
    )
    assert positions == sorted(positions)
    assert _pytest_markers(TRENDING) == _pytest_markers(CI)


@pytest.mark.trending
def test_trending_runs_no_core_stage_and_the_refresh_runs_no_trending_one():
    for stage in (
        "quantflow.ingestion.yfinance_fetcher",
        "quantflow.ingestion.split_guard",
        "quantflow.features.indicators",
        "quantflow.features.anomalies",
        "quantflow.features.sentiment",
        "quantflow.pipelines.run_models",
        "quantflow.pipelines.backtest",
    ):
        assert stage not in TRENDING, stage
    assert "quantflow.pipelines.trending" not in DAILY_REFRESH
    assert "continue-on-error" not in TRENDING


@pytest.mark.trending
def test_trending_caches_finbert_by_the_revision_the_code_pins():
    from quantflow.features.sentiment import FINBERT_REVISION

    assert f"key: finbert-${{{{ runner.os }}}}-{FINBERT_REVISION}" in TRENDING


@pytest.mark.trending
def test_trending_tracks_the_core_tickers_and_pins_what_config_pins():
    """TRENDING_PINNED's default lives in config.py, .env.example and here;
    the core list here matches the refresh's and .env.example's."""
    from quantflow import config

    tickers = re.search(r"^\s*TICKERS:\s*(\S+)", TRENDING, re.MULTILINE)
    daily = re.search(r"^\s*TICKERS:\s*(\S+)", DAILY_REFRESH, re.MULTILINE)
    assert tickers and daily and tickers.group(1) == daily.group(1)
    pinned = re.search(r"^\s*TRENDING_PINNED:\s*(\S+)", TRENDING, re.MULTILINE)
    assert pinned and pinned.group(1) == config.DEFAULT_TRENDING_PINNED


def _trending_test_files():
    tests = ROOT / "tests"
    return sorted(
        path
        for path in tests.rglob("test_*.py")
        if "pytest.mark.trending" in path.read_text(encoding="utf-8")
        and path.name != "test_workflows.py"
    )


@pytest.mark.trending
def test_no_trending_test_reads_the_tickers_from_config():
    """CI sets two tickers and the daily refresh eight: a trending test that
    read config.TICKERS could pass in one and fail in the other. Setting it
    in a test (monkeypatch, by name) is fine; reading it is not."""
    import ast

    files = _trending_test_files()
    assert len(files) >= 5, files
    offenders = []
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            reads = (
                (isinstance(node, ast.Attribute) and node.attr == "TICKERS")
                or (isinstance(node, ast.Name) and node.id == "TICKERS")
                or (
                    isinstance(node, ast.ImportFrom)
                    and any(alias.name == "TICKERS" for alias in node.names)
                )
            )
            if reads:
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []


@pytest.mark.trending
def test_the_tickers_check_sees_a_read(tmp_path):
    """Or the test above proves nothing."""
    import ast

    for source in (
        "from quantflow.config import TICKERS\n",
        "x = config.TICKERS\n",
        "x = TICKERS\n",
    ):
        nodes = list(ast.walk(ast.parse(source)))
        assert any(
            (isinstance(n, ast.Attribute) and n.attr == "TICKERS")
            or (isinstance(n, ast.Name) and n.id == "TICKERS")
            or (isinstance(n, ast.ImportFrom) and n.names[0].name == "TICKERS")
            for n in nodes
        ), source
    setting = 'monkeypatch.setattr(config, "TICKERS", ["AAPL"])\n'
    assert not any(
        isinstance(n, ast.Attribute | ast.Name)
        and getattr(n, "attr", getattr(n, "id", "")) == "TICKERS"
        for n in ast.walk(ast.parse(setting))
    )
