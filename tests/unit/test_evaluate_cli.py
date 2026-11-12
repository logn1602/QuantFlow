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

# The registered calendar: the 2026-11-06 snapshot's 648 trading dates, less
# 2025-01-09's closure, since every command takes its windows by date
# (splits.registered_grid, D21).
DATES = pd.date_range("2024-04-10", "2026-11-06", freq=TRADING_DAY).drop(
    pd.Timestamp("2025-01-09")
)


def _bars(seed, dates=DATES):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(dates))))
    return pd.DataFrame(
        {"date": dates, "open": close, "high": close + 1, "low": close - 1,
         "close": close, "volume": 1e6}
    )  # fmt: skip


def _snapshot(tmp_path, built=None, dates=DATES):
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
        as_of=dates[-1],
        exported_at=dates[-1].tz_localize("UTC") + pd.Timedelta(days=1),
        prices={"AAPL": _bars(1, dates), "MSFT": _bars(2, dates)},
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


def test_the_report_states_its_origins_the_way_publish_reads_them(tmp_path, harness):
    """publish reads the origin count and dates back out of the report's
    "scored origins" line, so a rewording here would break publishing."""
    from quantflow.evaluation import publish

    _snapshot(tmp_path)
    assert _run(tmp_path) == 0
    out = tmp_path / "runs" / "r"
    run = publish.read_run(out, evaluate.run_header(out))
    scored = pd.read_csv(out / "cells.csv").query("role == 'scored'")["origin"]
    assert run.origins == (scored.nunique(), scored.min(), scored.max())


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


def test_a_run_without_the_naive_baseline_is_refused_before_reading_anything(
    tmp_path, harness, capsys
):
    """Every figure is a comparison with "no change": without it the run
    scored every origin, wrote its cells, then failed on the missing skill
    column with no report (found in the full-diff review, 2026-11-12)."""
    assert _run(tmp_path, "--models", "drift") == 2  # no snapshot exists
    assert harness.calls == []
    assert "naive" in capsys.readouterr().err
    assert not (tmp_path / "runs" / "r").exists()


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
        {
            "naive": ro.naive_forecast,
            "short": lambda data: ro.naive_forecast(data).iloc[:3],
        },
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
    from quantflow.evaluation.splits import registered_grid

    origin = DATES[registered_grid(DATES).validation[3]]
    _snapshot(tmp_path, built={"AAPL": [origin], "MSFT": []})
    assert _run(tmp_path) == 1
    assert "built from 15-min bars" in capsys.readouterr().err


def test_a_newer_snapshot_scores_the_registered_dates(tmp_path, harness):
    """The windows are fixed by date (D21). Counted back from a newer
    snapshot's last date they moved: with data to 2026-11-20 the test
    window reached 2026-11-11, inside the forward test's window, and the
    validation window held test dates (full-diff review, 2026-11-12)."""
    newer = pd.date_range(DATES[0], "2026-11-20", freq=TRADING_DAY)
    _snapshot(tmp_path, dates=newer[newer != pd.Timestamp("2025-01-09")])
    assert _run(tmp_path, "--window", "test", "--reason", "x") == 0
    cells = pd.read_csv(tmp_path / "runs" / "r" / "cells.csv", parse_dates=["origin"])
    first, last = cells["origin"].min(), cells["origin"].max()
    assert (first.date().isoformat(), last.date().isoformat()) == (
        "2026-05-07",
        "2026-10-28",
    )
    assert cells["origin"].nunique() == 25


def test_a_snapshot_whose_windows_cannot_be_matched_is_refused(
    tmp_path, harness, capsys
):
    """A date lost inside the windows would shift them; the run refuses
    before scoring anything."""
    _snapshot(tmp_path, dates=DATES.delete(DATES.get_loc(pd.Timestamp("2025-09-02"))))
    assert _run(tmp_path) == 2
    assert "as registered" in capsys.readouterr().err
    assert not (tmp_path / "runs" / "r").exists()


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


def test_export_live_is_refused_without_the_read_only_role(
    monkeypatch, tmp_path, capsys
):
    """The export reads the live database as the SELECT-only role
    (sql/schema_roles.sql) and refuses before connecting without it."""
    import quantflow.config as config

    exported = []
    monkeypatch.setattr(config, "DB_HOST", "aws-0.pooler.supabase.com")
    monkeypatch.setattr(config, "DB_READONLY_USER", "")
    monkeypatch.setattr(config, "DB_READONLY_PASSWORD", "")
    monkeypatch.setattr(evaluate, "export_snapshot", lambda *a, **k: exported.append(a))
    code = evaluate.main(
        ["export", "--as-of", "2026-11-06", "--out", str(tmp_path), "--live"]
    )

    assert code == 2
    assert exported == []
    assert "DB_READONLY_USER" in capsys.readouterr().err


def test_export_stops_when_a_table_the_snapshot_reads_shows_no_row(
    monkeypatch, tmp_path, capsys
):
    """Every table the snapshot reads goes to db.readonly.require_rows: with
    row-level security and no policy for the read-only role, each showed it
    no row, and the export would have written a snapshot without data."""
    from contextlib import contextmanager

    import quantflow.config as config
    from quantflow.db import readonly
    from quantflow.db.snapshot import SNAPSHOT_TABLES

    seen = []

    @contextmanager
    def nothing_visible(statement_timeout="120s", expect_rows=()):
        seen.append(tuple(expect_rows))
        raise readonly.EmptyTablesError("anomalies: no row visible to the role")
        yield

    monkeypatch.setattr(config, "DB_HOST", "127.0.0.1")
    monkeypatch.setattr(evaluate, "read_only_connection", nothing_visible)
    code = evaluate.main(["export", "--as-of", "2026-11-06", "--out", str(tmp_path)])

    assert code == 1
    assert seen == [SNAPSHOT_TABLES]
    assert "Failed: EmptyTablesError: anomalies" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())
    assert set(SNAPSHOT_TABLES) <= readonly.ROWS_EXPECTED


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

# Long enough for the tuning window: the registered calendar itself.
LONG = DATES


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
        tmp_path / "long", as_of=dates[-1],
        exported_at=dates[-1].tz_localize("UTC") + pd.Timedelta(days=1),
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
    from quantflow.evaluation.rolling_origin import calendar_mismatched
    from quantflow.evaluation.splits import registered_grid

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
    grid = registered_grid(LONG)
    closed = calendar_mismatched(LONG, grid.tuning)  # 2025-01-09, left out
    assert set(cells["window"]) == {"tuning"}
    assert cells["origin"].nunique() == len(grid.tuning) - len(closed) > 0
    # No tuning target reaches the validation warmup.
    assert cells["target_date"].max() < LONG[grid.validation_warmup[0]]
    table = pd.read_csv(out / "tuning.csv").set_index("candidate")
    assert table.loc["xgboost#01", "chosen"] and not table.loc["xgboost#00", "chosen"]
    # A tie goes to the current settings, listed first.
    assert table.loc["lightgbm#00", "chosen"] and not table.loc["lightgbm#01", "chosen"]
    ((_, _, tags),) = harness.call("start")
    assert tags["window"] == "tuning"
    assert harness.call("end") == [("end", "FINISHED")]
    ((_, metrics),) = harness.call("metrics")
    # MLflow refuses "#" in a metric name.
    assert set(metrics) == {
        "xgboost_00.mape",
        "xgboost_01.mape",
        "lightgbm_00.mape",
        "lightgbm_01.mape",
    }


def test_tune_refuses_a_snapshot_with_no_tuning_origin(tmp_path, harness, capsys):
    # History from 2024-10-02: the windows fit, with no room before the warmup.
    _long_snapshot(tmp_path, dates=LONG[120:])
    code = evaluate.main(
        ["tune", "--snapshot", str(tmp_path / "long"), "--out", str(tmp_path / "runs")]
    )
    assert code == 2 and "no tuning origin" in capsys.readouterr().err


def test_the_tuning_candidates_are_the_current_settings_and_the_grid():
    from quantflow.models import boosting

    candidates = evaluate.tuning_candidates(["xgboost"])
    assert len(candidates) == 1 + len(boosting.TUNING_GRID) == 13
    family, settings, forecaster = candidates["xgboost#00"]
    assert family == "xgboost"
    assert settings == {
        k: boosting.MODEL_PARAMS["xgboost"][k] for k in evaluate.TUNED_KEYS
    }
    assert forecaster.params == settings and forecaster.model_name == "xgboost"
    grid = [candidates[f"xgboost#{i:02d}"][1] for i in range(1, 13)]
    assert grid == boosting.TUNING_GRID


def test_tune_leaves_out_an_origin_whose_days_hold_an_unknown_closure(
    tmp_path, harness, monkeypatch
):
    """2025-01-09 in the real tuning window: the data skips a day the
    calendar trades, so each origin whose 7 days span it is left out and
    named, instead of the run failing."""
    from quantflow.evaluation.rolling_origin import calendar_mismatched
    from quantflow.evaluation.splits import registered_grid

    tuning = registered_grid(LONG).tuning
    left_out = calendar_mismatched(LONG, tuning)
    assert left_out, "the registered calendar leaves 2025-01-09 out"
    _long_snapshot(tmp_path)
    fake = {"xgboost#00": ("xgboost", {"max_depth": 4}, _off_by(1))}
    monkeypatch.setattr(evaluate, "tuning_candidates", lambda models: fake)
    code = evaluate.main(
        ["tune", "--snapshot", str(tmp_path / "long"), "--out", str(tmp_path / "runs"),
         "--label", "t"]
    )  # fmt: skip
    assert code == 0
    cells = pd.read_csv(tmp_path / "runs" / "t" / "cells.csv", parse_dates=["origin"])
    assert cells["origin"].nunique() == len(tuning) - len(left_out)
    report = (tmp_path / "runs" / "t" / "report.md").read_text("utf-8")
    for position in left_out:
        assert LONG[position] not in set(cells["origin"])
        assert str(LONG[position].date()) in report


def _fails_every_other_origin(data):
    if data.origin.dayofyear % 2:
        raise RuntimeError("no forecast")
    return ro.naive_forecast(data).assign(yhat=data.close * 1.001)


def _tune(tmp_path, *extra):
    return evaluate.main(
        ["tune", "--snapshot", str(tmp_path / "long"), "--out", str(tmp_path / "runs"),
         "--label", "t", *extra]
    )  # fmt: skip


def test_a_candidate_over_the_failure_limit_is_never_chosen(
    tmp_path, harness, monkeypatch
):
    """Scored only on the cells it managed, a candidate that failed half of
    them could look best; the harness's 10% limit rules it out instead."""
    _long_snapshot(tmp_path)
    fake = {
        "xgboost#00": ("xgboost", {"max_depth": 4}, _off_by(2)),
        "xgboost#01": ("xgboost", {"max_depth": 3}, _fails_every_other_origin),
    }
    monkeypatch.setattr(evaluate, "tuning_candidates", lambda models: fake)
    assert _tune(tmp_path) == 0
    table = pd.read_csv(tmp_path / "runs" / "t" / "tuning.csv").set_index("candidate")
    assert not table.loc["xgboost#01", "eligible"]
    assert table.loc["xgboost#00", "chosen"] and not table.loc["xgboost#01", "chosen"]


def test_tune_refuses_an_unknown_booster(tmp_path, harness):
    assert _tune(tmp_path, "--models", "xgbost") == 2
    assert harness.calls == []


def test_a_tuning_run_that_raises_is_logged_as_failed(tmp_path, harness, monkeypatch):
    _long_snapshot(tmp_path)
    short = {
        "xgboost#00": ("xgboost", {}, lambda data: ro.naive_forecast(data).iloc[:3])
    }
    monkeypatch.setattr(evaluate, "tuning_candidates", lambda models: short)
    assert _tune(tmp_path) == 1
    assert harness.call("text") == [("text", "traceback.txt")]
    assert harness.call("end") == [("end", "FAILED")]


def test_compare_refuses_a_run_from_uncommitted_code(tmp_path, harness, capsys):
    """A decision must rest on code that is in a commit."""
    _snapshot(tmp_path)
    for label in ("a", "b"):
        assert _run(tmp_path, "--label", label) == 0
    report = tmp_path / "runs" / "b" / "report.md"
    text = report.read_text("utf-8").replace(
        "- commit: abc1234def", "- commit: abc1234def (uncommitted changes)"
    )
    report.write_text(text, "utf-8")
    assert _compare(tmp_path) == 2
    assert "uncommitted" in capsys.readouterr().err


def test_a_compare_whose_logging_fails_ends_its_mlflow_run(tmp_path, monkeypatch):
    _snapshot(tmp_path)
    monkeypatch.setattr(evaluate, "FORECASTERS", dict(ro.BASELINES))
    monkeypatch.setattr(evaluate, "COMBINERS", {})
    monkeypatch.setattr(evaluate, "git_state", lambda: ("abc1234def", False))
    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow())
    for label in ("a", "b"):
        assert _run(tmp_path, "--label", label) == 0
    failing = FakeMlflow(fail_log=True)
    monkeypatch.setitem(sys.modules, "mlflow", failing)
    assert _compare(tmp_path) == 1
    assert failing.call("end") == [("end", "FAILED")]
    assert (tmp_path / "runs" / "b" / "compare-a.csv").exists()
