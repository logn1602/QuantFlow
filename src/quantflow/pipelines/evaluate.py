"""
pipelines/evaluate.py
---------------------
The rolling-origin evaluation's command line (docs/evaluation.md).

    python -m quantflow.pipelines.evaluate export --as-of 2026-11-06 [--live]
    python -m quantflow.pipelines.evaluate coverage --snapshot eval_snapshots/2026-11-06
    python -m quantflow.pipelines.evaluate run --snapshot eval_snapshots/2026-11-06
    python -m quantflow.pipelines.evaluate run --snapshot ... --window test --reason "..."
    python -m quantflow.pipelines.evaluate compare --incumbent eval_runs/B0 --candidate eval_runs/E1 --models xgboost
    python -m quantflow.pipelines.evaluate tune --snapshot eval_snapshots/2026-11-06 --workers 3
    python -m quantflow.pipelines.evaluate publish --validation eval_runs/P4 --test eval_runs/T3

export reads the database once, read-only, into a snapshot directory;
coverage, run and tune read only the snapshot, and compare only two run
folders, applying Phase 3's keep-or-drop rule (evaluation/decision.py).
publish writes what the README and the dashboard show (evaluation/publish.py)
from a validation run and a test-date run on the same snapshot. run
scores the validation dates unless given --window test with a reason: the
test dates are reported, never used to choose, and every scoring of them is
counted (docs/evaluation.md, section 2). A test-date run also refuses
uncommitted changes to tracked files, so each scoring is tied to a commit.

It lives in pipelines because it wires the models into the harness, and
evaluation never imports models.
"""

import argparse
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from quantflow.config import TICKERS
from quantflow.dashboard import accuracy
from quantflow.db.prices import classify_bars, daily_bars
from quantflow.db.readonly import live_refusal, masked, read_only_connection
from quantflow.db.snapshot import read_snapshot_tables
from quantflow.evaluation import decision, metrics, publish, report
from quantflow.evaluation.rolling_origin import (
    MAX_FAILURE_SHARE,
    MAX_WORKERS,
    Forecaster,
    calendar_mismatched,
    check_failure_rate,
    run_origins,
)
from quantflow.evaluation.snapshot import (
    Snapshot,
    SnapshotError,
    check_built_dates,
    read_snapshot,
    write_snapshot,
)
from quantflow.evaluation.splits import HORIZON, ORIGIN_STEP, origin_grid
from quantflow.features.engineering import (
    INDICATOR_COLUMNS,
    build_features,
    daily_sentiment,
    merge_feature_inputs,
    stage_anomaly_scores,
)
from quantflow.models.boosting import MODEL_PARAMS, TUNING_GRID
from quantflow.models.forecasters import (
    BAND_VARIANTS,
    COMBINERS,
    FORECASTERS,
    VARIANTS,
    Booster,
)
from quantflow.utils.logger import get_logger

logger = get_logger("evaluate")

MLFLOW_EXPERIMENT = "rolling_origin_evaluation"
SNAPSHOT_ROOT = Path("eval_snapshots")
RUNS_ROOT = Path("eval_runs")


# ── export ────────────────────────────────────────────────────────────────────


def export_snapshot(as_of, tickers: list[str], root: Path = SNAPSHOT_ROOT) -> Path:
    """Read the database once, read-only, and write the snapshot for as_of."""
    as_of = pd.Timestamp(as_of).normalize()
    with read_only_connection() as conn:
        tables = read_snapshot_tables(conn, tickers, as_of)
    raw = tables["raw_bars"]
    prices, built = {}, {}
    for ticker in tickers:
        bars = raw[raw["ticker"] == ticker].drop(columns="ticker")
        if bars.empty:
            raise SnapshotError(f"no prices for {ticker} up to {as_of.date()}")
        daily = daily_bars(bars, ticker)
        classified = classify_bars(bars)
        official = set(classified.loc[classified["is_daily"], "date"])
        built[ticker] = sorted(d for d in daily["date"] if d not in official)
        prices[ticker] = daily
    return write_snapshot(
        root / as_of.date().isoformat(),
        as_of=as_of,
        exported_at=pd.Timestamp.now(tz="UTC"),
        prices=prices,
        articles=tables["articles"],
        stored_indicators=tables["stored_indicators"],
        stored_anomalies=tables["stored_anomalies"],
        built_dates=built,
    )


def stored_feature_check(snapshot: Snapshot) -> pd.DataFrame:
    """Rebuilt against stored: the largest difference per ticker and column
    between what the stages would store for the whole snapshot and the rows
    they did store: the indicators build_features computes, and the anomaly
    stage's scores (engineering.stage_anomaly_scores; the features use
    point-in-time scores instead, decision D14). Expected to be zero, or
    rounding-sized (docs/evaluation.md, section 2)."""
    rows = []
    for ticker in snapshot.tickers:
        articles = snapshot.articles[snapshot.articles["ticker"] == ticker]
        prices = snapshot.prices[ticker]
        rebuilt = merge_feature_inputs(
            prices,
            build_features(prices, articles, ticker)[["date", *INDICATOR_COLUMNS]],
            stage_anomaly_scores(prices, ticker),
            daily_sentiment(articles, prices["date"]),
        )

        def stored_rows(table: pd.DataFrame, t: str = ticker) -> pd.DataFrame:
            return table[table["ticker"] == t].drop(columns="ticker")

        stored = merge_feature_inputs(
            prices,
            stored_rows(snapshot.stored_indicators),
            stored_rows(snapshot.stored_anomalies),
            daily_sentiment(articles, prices["date"]),
        )
        for column in [*INDICATOR_COLUMNS, "zscore"]:
            a = pd.to_numeric(rebuilt[column], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(stored[column], errors="coerce").to_numpy(dtype=float)
            both = np.isfinite(a) & np.isfinite(b)
            rows.append(
                {"ticker": ticker, "column": column, "compared": int(both.sum()),
                 "only_one_side": int((np.isfinite(a) ^ np.isfinite(b)).sum()),
                 "max_abs_difference": float(np.max(np.abs(a[both] - b[both])))
                 if both.any() else np.nan}
            )  # fmt: skip
    return pd.DataFrame(rows)


# ── coverage ──────────────────────────────────────────────────────────────────


def coverage_table(snapshot: Snapshot) -> pd.DataFrame:
    dates = snapshot.trading_dates
    grid = origin_grid(len(dates))
    windows = {
        "validation": list(dates[list(grid.validation)]),
        "test": list(dates[list(grid.test)]),
    }
    return report.news_coverage(snapshot.articles, dates, snapshot.tickers, windows)


# ── run ───────────────────────────────────────────────────────────────────────


def git_state() -> tuple[str, bool]:
    """The commit checked out, and whether tracked files have changes."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    changes = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return commit, bool(changes)


def _start_mlflow(label: str, tags: dict):
    """(mlflow module, None) with a run started, or (None, the error)."""
    try:
        import mlflow

        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        mlflow.start_run(run_name=label, tags=tags)
        return mlflow, None
    except Exception as e:
        return None, e


def run_evaluation(args) -> int:
    unknown = sorted(
        set(args.models) - {*FORECASTERS, *VARIANTS, *COMBINERS, *BAND_VARIANTS}
    )
    if unknown:
        print(f"Refusing: no model named {', '.join(unknown)}.", file=sys.stderr)
        return 2
    if args.window == "test" and not args.reason:
        print(
            "Refusing: a test-date run needs --reason (it is counted).", file=sys.stderr
        )
        return 2
    commit, dirty = git_state()
    if args.window == "test" and dirty:
        print(
            "Refusing: tracked files have uncommitted changes; a test-date "
            "scoring must be tied to a commit.",
            file=sys.stderr,
        )
        return 2

    snapshot = read_snapshot(args.snapshot)
    dates = snapshot.trading_dates
    scored, warmup = origin_grid(len(dates)).window(args.window)
    tickers = args.tickers or snapshot.tickers
    known = {**FORECASTERS, **VARIANTS}
    forecasters = {m: f for m, f in known.items() if m in args.models}
    combiners = {
        m: c for m, c in {**COMBINERS, **BAND_VARIANTS}.items() if m in args.models
    }
    needs_history = any(getattr(c, "needs_history", False) for c in combiners.values())
    origins = [(p, "scored") for p in scored]
    if needs_history:
        origins += [(p, "warmup") for p in warmup]
    used = sorted({p + k for p, _ in origins for k in range(HORIZON + 1)})
    check_built_dates(snapshot, dates[used])

    started = pd.Timestamp.now(tz="UTC")
    label = args.label or f"{args.window}-{started:%Y%m%dT%H%M%S}"
    out = Path(args.out) / label
    out.mkdir(parents=True, exist_ok=False)
    tags = {"window": args.window, "reason": args.reason or "", "commit": commit}
    mlflow, mlflow_error = _start_mlflow(label, tags)
    header = {
        "window": args.window,
        "reason": args.reason or "",
        "snapshot": str(snapshot.directory),
        "snapshot digest": snapshot.digest,
        "as of": snapshot.as_of.date().isoformat(),
        "commit": commit + (" (uncommitted changes)" if dirty else ""),
        "tickers": ", ".join(tickers),
        "models": ", ".join([*forecasters, *combiners]),
        "scored origins": f"{len(scored)}, every {ORIGIN_STEP} trading days, "
        f"{dates[scored[0]].date()} to {dates[scored[-1]].date()}",
        "horizon": HORIZON,
        "workers": args.workers,
        "started": started.isoformat(timespec="seconds"),
    }
    clock = time.perf_counter()
    try:
        cells = run_origins(
            {t: snapshot.prices[t] for t in tickers},
            snapshot.articles,
            origins,
            forecasters,
            combiners,
            window=args.window,
            workers=args.workers,
        )
        cells.to_csv(out / "cells.csv", index=False)
        check_failure_rate(cells)
        coverage = coverage_table(snapshot)
        header["seconds"] = round(time.perf_counter() - clock)
        if mlflow is not None:
            header["mlflow"] = (
                f"{mlflow.get_tracking_uri()} run {mlflow.active_run().info.run_id}"
            )
        compared = report.comparable(report.scored(cells))
        headline = report.headline(compared)
        tables = {
            "headline": headline,
            "per_ticker": report.summary_table(compared, ["ticker", "model"]),
            "per_horizon": report.summary_table(compared, ["model", "h"]),
            "failures": report.failures(report.scored(cells)),
            "coverage": coverage,
        }
        for name, table in tables.items():
            table.to_csv(out / f"{name}.csv", index=False)
        (out / "report.md").write_text(
            report.markdown(header, cells, coverage), encoding="utf-8"
        )
    except Exception:
        if mlflow is not None:
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.end_run(status="FAILED")
        raise
    print(f"Report: {out / 'report.md'}")

    if mlflow is None:
        print(
            f"MLflow failed, so this run is not logged: {mlflow_error}", file=sys.stderr
        )
        return 1
    try:
        mlflow.log_params(
            {"window": args.window, "snapshot_digest": snapshot.digest,
             "as_of": header["as of"], "commit": commit, "dirty": dirty,
             "horizon": HORIZON, "origin_step": ORIGIN_STEP, "origins": len(scored),
             "tickers": ",".join(tickers), "models": header["models"],
             "workers": args.workers}
        )  # fmt: skip
        metrics = {}
        for row in headline.to_dict("records"):
            for key in ("mean_mape", "mean_skill", "p_holm"):
                value = row.get(key)
                if value is not None and np.isfinite(value):
                    metrics[f"{row['model']}.{key}"] = float(value)
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(out))
        mlflow.end_run(status="FINISHED")
    except Exception as e:
        print(f"MLflow logging failed, so this run is not logged: {e}", file=sys.stderr)
        return 1
    if args.window == "test":
        print(
            "Test-date scoring: add it to the ledger in docs/evaluation.md: "
            f"{started.date()} | {commit[:7]} | {header['mlflow']} | {args.reason}"
        )
    return 0


# ── tune ──────────────────────────────────────────────────────────────────────

TUNED_KEYS = ("max_depth", "learning_rate", "n_estimators")


def tuning_candidates(models: list[str]) -> dict[str, tuple[str, dict, Forecaster]]:
    """name -> (family, settings, forecaster): each booster's current
    settings (#00) and every TUNING_GRID setting."""
    out: dict[str, tuple[str, dict, Forecaster]] = {}
    for family in models:
        current = {k: MODEL_PARAMS[family][k] for k in TUNED_KEYS}
        for i, settings in enumerate([current, *TUNING_GRID]):
            out[f"{family}#{i:02d}"] = (
                family,
                settings,
                Booster(family, params=settings),
            )
    return out


def tuning_table(cells: pd.DataFrame, candidates: dict) -> pd.DataFrame:
    """Per candidate, its failed cells and its MAPE over the cells every
    eligible candidate of its family forecast. A candidate that failed more
    than the harness's limit (MAX_FAILURE_SHARE) is not eligible, as a model
    over it fails a run: scored only on the cells it managed it could look
    best. The eligible candidate with the lowest MAPE per family is chosen."""
    rows = []
    for family in sorted({fam for fam, _, _ in candidates.values()}):
        names = [n for n, (fam, _, _) in candidates.items() if fam == family]
        failed = {
            n: cells.loc[cells["model"] == n, "status"].eq("failed") for n in names
        }
        eligible = [n for n in names if failed[n].mean() <= MAX_FAILURE_SHARE]
        shared = report.comparable(cells, eligible) if eligible else cells.iloc[:0]
        for name in names:
            group = shared[shared["model"] == name]
            rows.append(
                {"family": family, "candidate": name, **candidates[name][1],
                 "cells": len(group), "failed": int(failed[name].sum()),
                 "eligible": name in eligible,
                 "mape": metrics.mape(group["actual"], group["yhat"])
                 if len(group) else np.nan}
            )  # fmt: skip
    table = pd.DataFrame(rows)
    best = table[table["eligible"]].groupby("family")["mape"].min()
    # The first of equal MAPEs: the current settings, listed first, win a tie.
    table["chosen"] = (
        table["eligible"]
        & (table["mape"] == table["family"].map(best))
        & ~table.duplicated(["family", "eligible", "mape"])
    )
    return table


def tune_boosters(args) -> int:
    """Phase 3's E3: every candidate's 7-day forecasts at the tuning
    origins, which lie before the validation warmup (splits.OriginGrid
    .tuning), and the setting with the lowest MAPE per booster."""
    unknown = sorted(set(args.models) - set(MODEL_PARAMS))
    if unknown:
        print(f"Refusing: no booster named {', '.join(unknown)}.", file=sys.stderr)
        return 2
    commit, dirty = git_state()
    snapshot = read_snapshot(args.snapshot)
    dates = snapshot.trading_dates
    origins, _ = origin_grid(len(dates)).window("tuning")
    if not origins:
        print(
            "Refusing: the snapshot has no tuning origin (too little history "
            "before the validation warmup).",
            file=sys.stderr,
        )
        return 2
    # The window reaches 2025-01-09, a one-off closure the calendar does not
    # know: an origin whose forecast days include it is left out, and listed.
    left_out = calendar_mismatched(dates, origins)
    origins = tuple(p for p in origins if p not in left_out)
    used = sorted({p + k for p in origins for k in range(HORIZON + 1)})
    check_built_dates(snapshot, dates[used])
    candidates = tuning_candidates(args.models)
    tickers = args.tickers or snapshot.tickers

    started = pd.Timestamp.now(tz="UTC")
    label = args.label or f"tuning-{started:%Y%m%dT%H%M%S}"
    out = Path(args.out) / label
    out.mkdir(parents=True, exist_ok=False)
    mlflow, mlflow_error = _start_mlflow(label, {"window": "tuning", "commit": commit})
    clock = time.perf_counter()
    try:
        cells = run_origins(
            {t: snapshot.prices[t] for t in tickers},
            snapshot.articles,
            [(p, "scored") for p in origins],
            {name: forecaster for name, (_, _, forecaster) in candidates.items()},
            window="tuning",
            workers=args.workers,
        )
        cells.to_csv(out / "cells.csv", index=False)
        table = tuning_table(cells, candidates)
        table.to_csv(out / "tuning.csv", index=False)
        header = {
            "window": "tuning",
            "snapshot digest": snapshot.digest,
            "commit": commit + (" (uncommitted changes)" if dirty else ""),
            "tickers": ", ".join(tickers),
            "origins": f"{len(origins)}, {dates[origins[0]].date()} to "
            f"{dates[origins[-1]].date()}, last target "
            f"{dates[origins[-1] + HORIZON].date()}",
            "left out (a closure the calendar does not know in their days)": ", ".join(
                str(dates[p].date()) for p in left_out
            )
            or "none",
            "workers": args.workers,
            "seconds": round(time.perf_counter() - clock),
        }
        text = "\n".join(
            ["# Booster tuning (Phase 3, E3)\n"]
            + [f"- {key}: {value}" for key, value in header.items()]
            + ["", "## Candidates\n", report.markdown_table(table, digits=4)]
        )
        (out / "report.md").write_text(text, encoding="utf-8")
    except Exception:
        if mlflow is not None:
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.end_run(status="FAILED")
        raise
    chosen = table[table["chosen"]]
    for family in sorted(set(table["family"]) - set(chosen["family"])):
        print(f"No eligible candidate for {family}: none chosen.", file=sys.stderr)
    shown = [c for c in ("family", "candidate", *TUNED_KEYS, "mape") if c in chosen]
    print(chosen[shown].to_string(index=False))
    print(f"Report: {out / 'report.md'}")
    if mlflow is None:
        print(f"MLflow failed, so this is not logged: {mlflow_error}", file=sys.stderr)
        return 1
    try:
        mlflow.log_params(
            {"window": "tuning", "snapshot_digest": snapshot.digest, "commit": commit,
             "dirty": dirty, "origins": len(origins), "left_out": len(left_out),
             "tickers": ",".join(tickers),
             "workers": args.workers}
        )  # fmt: skip
        mlflow.log_metrics(tuning_metrics(table))
        mlflow.log_artifacts(str(out))
        mlflow.end_run(status="FINISHED")
    except Exception as e:
        print(f"MLflow logging failed, so this is not logged: {e}", file=sys.stderr)
        mlflow.end_run(status="FAILED")
        return 1
    return 0


def tuning_metrics(table: pd.DataFrame) -> dict[str, float]:
    """Each candidate's MAPE under a name MLflow accepts (no "#")."""
    return {
        f"{row['candidate'].replace('#', '_')}.mape": float(row["mape"])
        for row in table.to_dict("records")
    }


# ── compare ───────────────────────────────────────────────────────────────────


def run_header(directory: Path) -> dict:
    """The "- key: value" lines at the top of a run's report.md."""
    header: dict[str, str] = {}
    for line in (directory / "report.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            break
        if line.startswith("- ") and ": " in line:
            key, value = line[2:].split(": ", 1)
            header.setdefault(key, value)
    return header


def compare_runs(args) -> int:
    """The keep-or-drop rule (evaluation/decision.py) for each named model,
    the candidate run against the incumbent's, on the same snapshot."""
    incumbent, candidate = Path(args.incumbent), Path(args.candidate)
    heads = {"incumbent": run_header(incumbent), "candidate": run_header(candidate)}
    digests = {side: head.get("snapshot digest") for side, head in heads.items()}
    if not digests["incumbent"] or digests["incumbent"] != digests["candidate"]:
        print(
            "Refusing: the runs are not on the same snapshot "
            f"({digests['incumbent']} against {digests['candidate']}).",
            file=sys.stderr,
        )
        return 2
    # A decision must rest on code that is in a commit.
    dirty = [s for s, h in heads.items() if "uncommitted" in h.get("commit", "")]
    if dirty:
        print(
            f"Refusing: the {' and '.join(dirty)} run had uncommitted changes.",
            file=sys.stderr,
        )
        return 2
    old = pd.read_csv(incumbent / "cells.csv")
    new = pd.read_csv(candidate / "cells.csv")
    results = [decision.compare(old, new, model) for model in args.models]
    table = pd.DataFrame([decision.summary_row(r) for r in results])

    parts = [f"# {candidate.name} against {incumbent.name}\n"]
    parts += [
        f"- incumbent: {incumbent} (commit {heads['incumbent'].get('commit', '?')})",
        f"- candidate: {candidate} (commit {heads['candidate'].get('commit', '?')})",
        f"- snapshot digest: {digests['candidate']}",
        f"- rule: MAPE over the cells both forecast strictly lower, and lower on "
        f"at least {decision.TICKERS_NEEDED} tickers; more failed cells drops "
        "(docs/evaluation.md, section 16)",
        "",
        "## Decisions\n",
        report.markdown_table(table, digits=4),
    ]
    for result in results:
        parts += [
            f"## {result.model}: per ticker\n",
            report.markdown_table(result.per_ticker, digits=4),
            f"## {result.model}: per horizon\n",
            report.markdown_table(result.per_horizon, digits=4),
        ]
    text = "\n".join(parts)
    out = candidate / f"compare-{incumbent.name}.md"
    out.write_text(text, encoding="utf-8")
    table.to_csv(candidate / f"compare-{incumbent.name}.csv", index=False)
    print(table[["model", "decision", "reason"]].to_string(index=False))
    print(f"Comparison: {out}")

    tags = {"kind": "compare", "incumbent": incumbent.name, "candidate": candidate.name}
    mlflow, error = _start_mlflow(f"{candidate.name} vs {incumbent.name}", tags)
    if mlflow is None:
        print(f"MLflow failed, so this is not logged: {error}", file=sys.stderr)
        return 1
    try:
        mlflow.log_params(
            {"snapshot_digest": digests["candidate"], "models": ",".join(args.models),
             "tickers_needed": decision.TICKERS_NEEDED}
        )  # fmt: skip
        mlflow.log_metrics(
            {f"{r.model}.{key}": float(value) for r in results
             for key, value in (("keep", r.keep), ("mape_incumbent", r.mape_incumbent),
                                ("mape_candidate", r.mape_candidate))}
        )  # fmt: skip
        mlflow.log_text(text, out.name)
        mlflow.end_run(status="FINISHED")
    except Exception as e:
        print(f"MLflow logging failed, so this is not logged: {e}", file=sys.stderr)
        mlflow.end_run(status="FAILED")
        return 1
    return 0


# ── publish ───────────────────────────────────────────────────────────────────

# The table is written where the dashboard reads it, next to accuracy.py, and
# the chart under the same checkout's docs/, wherever the command runs from.
PUBLISHED_TABLE = accuracy.PUBLISHED
PUBLISHED_CHART = (
    PUBLISHED_TABLE.parents[3] / "docs" / "images" / "mape_by_days_ahead.png"
)


def publish_results(args) -> int:
    """The dashboard's accuracy table and the README's chart, from a
    validation run and a test-date run of the same snapshot, tickers and
    models, both run on committed code. Different tickers or models would put
    two incomparable "all tickers" figures side by side, and drop the chart
    panel of any model one run lacks."""
    runs = []
    for window, directory in (("validation", args.validation), ("test", args.test)):
        header = run_header(directory)
        if header.get("window") != window:
            print(
                f"Refusing: {directory} is not a {window} run "
                f"(window {header.get('window')!r}).",
                file=sys.stderr,
            )
            return 2
        if "uncommitted" in header.get("commit", "uncommitted"):
            print(
                f"Refusing: {directory} ran with uncommitted changes.", file=sys.stderr
            )
            return 2
        runs.append(publish.read_run(directory, header))
    for key in ("snapshot digest", "tickers", "models"):
        values = {run.header.get(key) for run in runs}
        if len(values) != 1 or None in values:
            print(
                f"Refusing: the runs differ in {key} ({sorted(map(str, values))}).",
                file=sys.stderr,
            )
            return 2
    table = publish.accuracy_table(runs)
    # The chart first: if it fails, the table is left as it was, so the
    # dashboard and the README never describe different runs.
    publish.horizon_chart(runs, args.chart)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.table, index=False, lineterminator="\n")
    print(f"Table: {args.table} ({len(table)} rows)")
    print(f"Chart: {args.chart}")
    return 0


# ── command line ──────────────────────────────────────────────────────────────


def _tickers(value: str) -> list[str]:
    return [t.strip().upper() for t in value.split(",") if t.strip()]


def _models(value: str) -> list[str]:
    return [m.strip().lower() for m in value.split(",") if m.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rolling-origin evaluation")
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export", help="read the database into a snapshot")
    export.add_argument(
        "--as-of", required=True, help="the last trading date, YYYY-MM-DD"
    )
    export.add_argument("--tickers", type=_tickers, default=None)
    export.add_argument("--out", type=Path, default=SNAPSHOT_ROOT)
    export.add_argument(
        "--live",
        action="store_true",
        help="read the non-local database in .env, as the read-only role",
    )

    coverage = commands.add_parser("coverage", help="news coverage per window")
    coverage.add_argument("--snapshot", type=Path, required=True)

    run = commands.add_parser("run", help="score the models on one window")
    run.add_argument("--snapshot", type=Path, required=True)
    run.add_argument("--window", choices=["validation", "test"], default="validation")
    run.add_argument("--reason", default="", help="required for --window test")
    run.add_argument("--tickers", type=_tickers, default=None)
    run.add_argument(
        "--models",
        type=_models,
        default=[*FORECASTERS, *COMBINERS],
    )
    run.add_argument("--label", default="", help="the run's folder and MLflow name")
    run.add_argument("--out", type=Path, default=RUNS_ROOT)
    run.add_argument(
        "--workers",
        type=int,
        choices=range(1, MAX_WORKERS + 1),
        default=1,
        help=f"tickers in parallel processes, at most {MAX_WORKERS}",
    )

    tune = commands.add_parser(
        "tune", help="the boosters' settings on the tuning origins (Phase 3, E3)"
    )
    tune.add_argument("--snapshot", type=Path, required=True)
    tune.add_argument(
        "--models",
        type=_models,
        default=sorted(MODEL_PARAMS),
    )
    tune.add_argument("--tickers", type=_tickers, default=None)
    tune.add_argument("--label", default="", help="the run's folder and MLflow name")
    tune.add_argument("--out", type=Path, default=RUNS_ROOT)
    tune.add_argument(
        "--workers", type=int, choices=range(1, MAX_WORKERS + 1), default=1
    )

    compare = commands.add_parser(
        "compare", help="the keep-or-drop rule, a candidate run against the incumbent"
    )
    compare.add_argument("--incumbent", type=Path, required=True, help="a run folder")
    compare.add_argument("--candidate", type=Path, required=True, help="a run folder")
    compare.add_argument(
        "--models",
        type=_models,
        required=True,
    )

    published = commands.add_parser(
        "publish", help="the README chart and the dashboard's accuracy table"
    )
    published.add_argument(
        "--validation", type=Path, required=True, help="a validation run folder"
    )
    published.add_argument(
        "--test", type=Path, required=True, help="a test-date run folder"
    )
    published.add_argument("--table", type=Path, default=PUBLISHED_TABLE)
    published.add_argument("--chart", type=Path, default=PUBLISHED_CHART)

    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            # The live database is read only as the read-only role.
            refusal = live_refusal(args.live)
            if refusal:
                print(f"Refusing: {refusal}", file=sys.stderr)
                return 2
            directory = export_snapshot(args.as_of, args.tickers or TICKERS, args.out)
            snapshot = read_snapshot(directory)
            print(f"Snapshot: {directory} (digest {snapshot.digest[:12]})")
            print(stored_feature_check(snapshot).to_string(index=False))
            return 0
        if args.command == "coverage":
            snapshot = read_snapshot(args.snapshot)
            table = coverage_table(snapshot)
            print(table.to_string(index=False))
            print(f"News ablation: {report.news_ablation_status(table)}")
            return 0
        if args.command == "compare":
            return compare_runs(args)
        if args.command == "tune":
            return tune_boosters(args)
        if args.command == "publish":
            return publish_results(args)
        return run_evaluation(args)
    except Exception as e:
        print(f"Failed: {type(e).__name__}: {masked(str(e))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
