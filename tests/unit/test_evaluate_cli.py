"""tests/unit/test_evaluate_cli.py

Covers the evaluation's command line on a synthetic snapshot, with the
baselines as the only forecasters and a fake MLflow, so nothing trains and
nothing reaches a real tracking store.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest

from quantflow.evaluation import rolling_origin as ro
from quantflow.evaluation.snapshot import write_snapshot
from quantflow.pipelines import evaluate
from quantflow.utils.trading_calendar import TRADING_DAY

DATES = pd.date_range("2024-01-02", periods=530, freq=TRADING_DAY)


def _bars(seed):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(DATES))))
    return pd.DataFrame(
        {"date": DATES, "open": close, "high": close + 1, "low": close - 1,
         "close": close, "volume": 1e6}
    )  # fmt: skip


def _snapshot(tmp_path, built=None):
    articles = pd.DataFrame(
        {"ticker": ["AAPL"], "source": ["rss"], "url_domain": ["finance.yahoo.com"],
         "published_at": [pd.Timestamp("2025-09-02 14:00", tz="UTC")],
         "inserted_at": [pd.Timestamp("2025-09-02 22:40", tz="UTC")],
         "sentiment": ["positive"], "score_pos": [0.8], "score_neg": [0.1],
         "score_neu": [0.1], "compound": [0.7]}
    )  # fmt: skip
    empty = pd.DataFrame(columns=["ticker", "date", "zscore"])
    return write_snapshot(
        tmp_path / "snap",
        as_of=DATES[-1],
        exported_at=pd.Timestamp("2026-01-01", tz="UTC"),
        prices={"AAPL": _bars(1), "MSFT": _bars(2)},
        articles=articles,
        stored_indicators=empty.drop(columns="zscore"),
        stored_anomalies=empty,
        built_dates=built or {"AAPL": [], "MSFT": []},
    )


class FakeMlflow(types.ModuleType):
    def __init__(self, fail_start=False, fail_log=False):
        super().__init__("mlflow")
        self.fail_start, self.fail_log = fail_start, fail_log
        self.calls: list[tuple] = []

    def set_experiment(self, name):
        self.calls.append(("experiment", name))

    def start_run(self, run_name, tags):
        if self.fail_start:
            raise ConnectionError("tracking store unavailable")
        self.calls.append(("start", run_name, tags))

    def active_run(self):
        return types.SimpleNamespace(info=types.SimpleNamespace(run_id="run-1"))

    def get_tracking_uri(self):
        return "sqlite:///fake.db"

    def log_params(self, params):
        if self.fail_log:
            raise OSError("disk full")
        self.calls.append(("params", params))

    def log_metrics(self, metrics):
        self.calls.append(("metrics", metrics))

    def log_artifacts(self, path):
        self.calls.append(("artifacts", path))

    def log_text(self, text, name):
        self.calls.append(("text", name))

    def end_run(self, status):
        self.calls.append(("end", status))

    def call(self, kind):
        return [c for c in self.calls if c[0] == kind]


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setattr(evaluate, "FORECASTERS", dict(ro.BASELINES))
    monkeypatch.setattr(evaluate, "COMBINERS", {})
    monkeypatch.setattr(evaluate, "VARIANTS", {"naive_again": ro.naive_forecast})
    monkeypatch.setattr(evaluate, "git_state", lambda: ("abc1234def", False))
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    return fake


def _run(tmp_path, *extra):
    return evaluate.main(
        ["run", "--snapshot", str(tmp_path / "snap"), "--out", str(tmp_path / "runs"),
         "--label", "r", *extra]
    )  # fmt: skip


def test_a_validation_run_writes_its_report_and_logs_to_mlflow(tmp_path, harness):
    _snapshot(tmp_path)
    assert _run(tmp_path) == 0
    out = tmp_path / "runs" / "r"
    for name in ("report.md", "cells.csv", "headline.csv", "coverage.csv"):
        assert (out / name).exists(), name
    cells = pd.read_csv(out / "cells.csv")
    assert set(cells["window"]) == {"validation"} and set(cells["role"]) == {"scored"}
    assert cells["origin"].nunique() == 42
    ((_, _, tags),) = harness.call("start")
    assert tags["window"] == "validation" and tags["commit"] == "abc1234def"
    ((_, params),) = harness.call("params")
    assert len(params["snapshot_digest"]) == 64 and params["origins"] == 42
    assert harness.call("end") == [("end", "FINISHED")]
    report = (out / "report.md").read_text("utf-8")
    assert "window: validation" in report and "News coverage" in report


def test_a_variant_runs_only_when_named(tmp_path, harness):
    _snapshot(tmp_path)
    assert _run(tmp_path) == 0
    default = pd.read_csv(tmp_path / "runs" / "r" / "cells.csv")
    assert "naive_again" not in set(default["model"])
    evaluate.main(
        ["run", "--snapshot", str(tmp_path / "snap"), "--out", str(tmp_path / "runs"),
         "--label", "v", "--models", "naive,naive_again"]
    )  # fmt: skip
    named = pd.read_csv(tmp_path / "runs" / "v" / "cells.csv")
    assert set(named["model"]) == {"naive", "naive_again"}


def test_an_unknown_model_is_refused_before_reading_anything(tmp_path, harness):
    assert _run(tmp_path, "--models", "naive,xgbost") == 2  # no snapshot exists
    assert harness.calls == []


def test_a_test_run_without_a_reason_is_refused_before_reading_anything(
    tmp_path, harness
):
    assert _run(tmp_path, "--window", "test") == 2  # no snapshot exists: never read
    assert harness.calls == []


def test_a_test_run_is_refused_on_uncommitted_changes(tmp_path, harness, monkeypatch):
    _snapshot(tmp_path)
    monkeypatch.setattr(evaluate, "git_state", lambda: ("abc1234def", True))
    assert _run(tmp_path, "--window", "test", "--reason", "Phase 2 T1") == 2


def test_a_test_run_is_tagged_and_asks_for_its_ledger_entry(tmp_path, harness, capsys):
    _snapshot(tmp_path)
    assert _run(tmp_path, "--window", "test", "--reason", "Phase 2 T1") == 0
    ((_, _, tags),) = harness.call("start")
    assert tags == {"window": "test", "reason": "Phase 2 T1", "commit": "abc1234def"}
    assert "add it to the ledger" in capsys.readouterr().out
    cells = pd.read_csv(tmp_path / "runs" / "r" / "cells.csv")
    assert cells["origin"].nunique() == 25


def test_a_run_that_raises_is_logged_as_failed(tmp_path, harness, monkeypatch):
    _snapshot(tmp_path)
    monkeypatch.setattr(
        evaluate,
        "FORECASTERS",
        {"short": lambda data: ro.naive_forecast(data).iloc[:3]},
    )
    assert _run(tmp_path) == 1
    assert harness.call("text") == [("text", "traceback.txt")]
    assert harness.call("end") == [("end", "FAILED")]


def test_without_mlflow_the_files_are_written_and_the_run_fails(tmp_path, monkeypatch):
    _snapshot(tmp_path)
    monkeypatch.setattr(evaluate, "FORECASTERS", dict(ro.BASELINES))
    monkeypatch.setattr(evaluate, "COMBINERS", {})
    monkeypatch.setattr(evaluate, "git_state", lambda: ("abc1234def", False))
    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow(fail_start=True))
    assert _run(tmp_path) == 1
    assert (tmp_path / "runs" / "r" / "report.md").exists()


def test_a_date_built_from_15min_bars_at_an_origin_is_refused(
    tmp_path, harness, capsys
):
    from quantflow.evaluation.splits import origin_grid

    origin = DATES[origin_grid(len(DATES)).validation[3]]
    _snapshot(tmp_path, built={"AAPL": [origin], "MSFT": []})
    assert _run(tmp_path) == 1
    assert "built from 15-min bars" in capsys.readouterr().err


def test_coverage_reports_the_news_ablation_status(tmp_path, capsys):
    _snapshot(tmp_path)
    assert evaluate.main(["coverage", "--snapshot", str(tmp_path / "snap")]) == 0
    assert "News ablation: not enough data" in capsys.readouterr().out


def test_export_refuses_a_remote_database_without_live(monkeypatch, tmp_path):
    import quantflow.config as config

    monkeypatch.setattr(config, "DB_HOST", "aws-0.pooler.supabase.com")
    assert (
        evaluate.main(["export", "--as-of", "2026-11-06", "--out", str(tmp_path)]) == 2
    )
    assert not any(tmp_path.iterdir())


def _compare(tmp_path, incumbent="a", candidate="b", models="drift"):
    return evaluate.main(
        ["compare", "--incumbent", str(tmp_path / "runs" / incumbent),
         "--candidate", str(tmp_path / "runs" / candidate), "--models", models]
    )  # fmt: skip


def test_compare_applies_the_rule_and_logs_it(tmp_path, harness, capsys):
    _snapshot(tmp_path)
    for label in ("a", "b"):
        assert _run(tmp_path, "--label", label) == 0
    harness.calls.clear()
    assert _compare(tmp_path, models="drift,naive") == 0
    out = tmp_path / "runs" / "b"
    table = pd.read_csv(out / "compare-a.csv")
    # The same code twice: a tie, which drops.
    assert list(table["decision"]) == ["drop", "drop"]
    assert "Decisions" in (out / "compare-a.md").read_text("utf-8")
    assert "drift" in capsys.readouterr().out
    ((_, name, tags),) = harness.call("start")
    assert name == "b vs a" and tags["kind"] == "compare"
    ((_, metrics),) = harness.call("metrics")
    assert metrics["drift.keep"] == 0.0
    assert harness.call("text") == [("text", "compare-a.md")]
    assert harness.call("end") == [("end", "FINISHED")]


def test_compare_refuses_runs_on_different_snapshots(tmp_path, harness, capsys):
    _snapshot(tmp_path)
    for label in ("a", "b"):
        assert _run(tmp_path, "--label", label) == 0
    report = tmp_path / "runs" / "b" / "report.md"
    text = report.read_text("utf-8")
    report.write_text(text.replace("snapshot digest: ", "snapshot digest: 0"), "utf-8")
    harness.calls.clear()
    assert _compare(tmp_path) == 2
    assert "same snapshot" in capsys.readouterr().err
    assert not (tmp_path / "runs" / "b" / "compare-a.csv").exists()
    assert harness.calls == []


def test_compare_refuses_test_date_cells(tmp_path, harness, capsys):
    """Decision D13: the test dates are reported, never used to choose."""
    _snapshot(tmp_path)
    assert _run(tmp_path, "--label", "a") == 0
    assert _run(tmp_path, "--label", "b", "--window", "test", "--reason", "x") == 0
    assert _compare(tmp_path) == 1
    assert "D13" in capsys.readouterr().err


# ── tune (Phase 3, E3) ────────────────────────────────────────────────────────

LONG = pd.date_range("2024-01-02", periods=600, freq=TRADING_DAY)


def _long_snapshot(tmp_path, dates=LONG):
    rng = np.random.default_rng(5)
    prices = {}
    for ticker in ("AAPL", "MSFT"):
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(dates))))
        prices[ticker] = pd.DataFrame(
            {"date": dates, "open": close, "high": close + 1, "low": close - 1,
             "close": close, "volume": 1e6}
        )  # fmt: skip
    empty = pd.DataFrame(columns=["ticker", "date", "zscore"])
    articles = pd.DataFrame(columns=["ticker", "source", "url_domain", "published_at",
                                     "inserted_at", "sentiment", "score_pos",
                                     "score_neg", "score_neu", "compound"])  # fmt: skip
    return write_snapshot(
        tmp_path / "long", as_of=dates[-1], exported_at=pd.Timestamp("2026-01-01", tz="UTC"),
        prices=prices, articles=articles, stored_indicators=empty.drop(columns="zscore"),
        stored_anomalies=empty, built_dates={"AAPL": [], "MSFT": []},
    )  # fmt: skip


def _off_by(percent):
    def forecast(data):
        return ro.naive_forecast(data).assign(yhat=data.close * (1 + percent / 100))

    return forecast


def test_tune_scores_every_candidate_on_the_tuning_origins(
    tmp_path, harness, monkeypatch
):
    from quantflow.evaluation.splits import origin_grid

    _long_snapshot(tmp_path)
    fake = {
        "xgboost#00": ("xgboost", {"max_depth": 4}, _off_by(2)),
        "xgboost#01": ("xgboost", {"max_depth": 3}, _off_by(1)),
        "lightgbm#00": ("lightgbm", {"max_depth": 4}, _off_by(1)),
        "lightgbm#01": ("lightgbm", {"max_depth": 3}, _off_by(1)),
    }
    monkeypatch.setattr(evaluate, "tuning_candidates", lambda models: fake)
    code = evaluate.main(
        ["tune", "--snapshot", str(tmp_path / "long"), "--out", str(tmp_path / "runs"),
         "--label", "t"]
    )  # fmt: skip
    assert code == 0
    out = tmp_path / "runs" / "t"
    cells = pd.read_csv(out / "cells.csv", parse_dates=["origin", "target_date"])
    grid = origin_grid(len(LONG))
    assert set(cells["window"]) == {"tuning"}
    assert cells["origin"].nunique() == len(grid.tuning) > 0
    # No tuning target reaches the validation warmup.
    assert cells["target_date"].max() < LONG[grid.validation_warmup[0]]
    table = pd.read_csv(out / "tuning.csv").set_index("candidate")
    assert table.loc["xgboost#01", "chosen"] and not table.loc["xgboost#00", "chosen"]
    # A tie goes to the current settings, listed first.
    assert table.loc["lightgbm#00", "chosen"] and not table.loc["lightgbm#01", "chosen"]
    ((_, _, tags),) = harness.call("start")
    assert tags["window"] == "tuning"
    assert harness.call("end") == [("end", "FINISHED")]


def test_tune_refuses_a_snapshot_with_no_tuning_origin(tmp_path, harness, capsys):
    _snapshot(tmp_path)  # 530 dates: the grid leaves no room before its warmup
    code = evaluate.main(
        ["tune", "--snapshot", str(tmp_path / "snap"), "--out", str(tmp_path / "runs")]
    )
    assert code == 2 and "no tuning origin" in capsys.readouterr().err


def test_the_tuning_candidates_are_the_current_settings_and_the_grid():
    from quantflow.models import boosting

    candidates = evaluate.tuning_candidates(["xgboost"])
    assert len(candidates) == 1 + len(boosting.TUNING_GRID) == 13
    family, settings, forecaster = candidates["xgboost#00"]
    assert family == "xgboost"
    assert settings == {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300}
    assert forecaster.params == settings and forecaster.model_name == "xgboost"
    grid = [candidates[f"xgboost#{i:02d}"][1] for i in range(1, 13)]
    assert grid == boosting.TUNING_GRID


def test_tune_leaves_out_an_origin_whose_days_hold_an_unknown_closure(
    tmp_path, harness, monkeypatch
):
    """Like 2025-01-09 in the real tuning window: the data skips a day the
    calendar trades, so the origin whose 7 days span it is left out and
    named, instead of the run failing."""
    from quantflow.evaluation.splits import origin_grid

    tuning = origin_grid(len(LONG)).tuning
    origin = tuning[3]
    full = pd.date_range(LONG[0], periods=len(LONG) + 1, freq=TRADING_DAY)
    _long_snapshot(tmp_path, dates=full.delete(origin + 3))
    fake = {"xgboost#00": ("xgboost", {"max_depth": 4}, _off_by(1))}
    monkeypatch.setattr(evaluate, "tuning_candidates", lambda models: fake)
    code = evaluate.main(
        ["tune", "--snapshot", str(tmp_path / "long"), "--out", str(tmp_path / "runs"),
         "--label", "t"]
    )  # fmt: skip
    assert code == 0
    cells = pd.read_csv(tmp_path / "runs" / "t" / "cells.csv", parse_dates=["origin"])
    assert cells["origin"].nunique() == len(tuning) - 1
    assert full[origin] not in set(cells["origin"])
    report = (tmp_path / "runs" / "t" / "report.md").read_text("utf-8")
    assert str(full[origin].date()) in report
