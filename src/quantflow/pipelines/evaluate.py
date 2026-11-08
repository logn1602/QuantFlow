"""
pipelines/evaluate.py
---------------------
The rolling-origin evaluation's command line (docs/evaluation.md).

    python -m quantflow.pipelines.evaluate export --as-of 2026-11-06 [--live]
    python -m quantflow.pipelines.evaluate coverage --snapshot eval_snapshots/2026-11-06
    python -m quantflow.pipelines.evaluate run --snapshot eval_snapshots/2026-11-06
    python -m quantflow.pipelines.evaluate run --snapshot ... --window test --reason "..."
    python -m quantflow.pipelines.evaluate compare --incumbent eval_runs/B0 --candidate eval_runs/E1 --models xgboost

export reads the database once, read-only, into a snapshot directory;
coverage and run read only the snapshot, and compare only two run folders,
applying Phase 3's keep-or-drop rule (evaluation/decision.py). run scores the validation dates unless
given --window test with a reason: the test dates are reported, never used
to choose, and every scoring of them is counted (docs/evaluation.md,
section 2). A test-date run also refuses uncommitted changes to tracked
files, so each scoring is tied to a commit.

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
from quantflow.db.prices import classify_bars, daily_bars
from quantflow.db.readonly import is_local_database, masked, read_only_connection
from quantflow.db.snapshot import read_snapshot_tables
from quantflow.evaluation import decision, report
from quantflow.evaluation.rolling_origin import (
    MAX_WORKERS,
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
from quantflow.models.forecasters import COMBINERS, FORECASTERS, VARIANTS
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
    unknown = sorted(set(args.models) - {*FORECASTERS, *VARIANTS, *COMBINERS})
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
    combiners = {m: c for m, c in COMBINERS.items() if m in args.models}
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
        return 1
    return 0


# ── command line ──────────────────────────────────────────────────────────────


def _tickers(value: str) -> list[str]:
    return [t.strip().upper() for t in value.split(",") if t.strip()]


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
        "--live", action="store_true", help="allow the non-local database in .env"
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
        type=lambda v: [m.strip().lower() for m in v.split(",") if m.strip()],
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

    compare = commands.add_parser(
        "compare", help="the keep-or-drop rule, a candidate run against the incumbent"
    )
    compare.add_argument("--incumbent", type=Path, required=True, help="a run folder")
    compare.add_argument("--candidate", type=Path, required=True, help="a run folder")
    compare.add_argument(
        "--models",
        type=lambda v: [m.strip().lower() for m in v.split(",") if m.strip()],
        required=True,
    )

    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            if not is_local_database() and not args.live:
                print(
                    "Refusing: .env points at a non-local database; pass --live.",
                    file=sys.stderr,
                )
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
        return run_evaluation(args)
    except Exception as e:
        print(f"Failed: {type(e).__name__}: {masked(str(e))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
