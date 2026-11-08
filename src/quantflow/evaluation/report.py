"""Tables from the rolling-origin cells (docs/evaluation.md, sections 3 and 7).

Everything here is a pure function of the cells, plus the articles for news
coverage, so a reported number can always be traced back to cells.csv.

Comparisons use only the cells every compared model forecast: a failed cell
removes that ticker, origin and day from every model's numbers, so no model
is scored on an easier subset than another.
"""

import numpy as np
import pandas as pd

from quantflow.evaluation import metrics
from quantflow.features.news import dates_with_news

BASELINE = "naive"
# A model beats naive only when its primary test's Holm-adjusted p-value is
# under this and it did better, never on a per-ticker or per-horizon figure.
SIGNIFICANCE = 0.05
# Under this share of validation (ticker, date) rows with a real article,
# the news ablation reports "not enough data" (decision D15).
NEWS_COVERAGE_NEEDED = 0.5
KEY = ["ticker", "origin", "h"]


def scored(cells: pd.DataFrame) -> pd.DataFrame:
    """The cells of scored origins; warmup origins only fit the ensemble."""
    return cells[cells["role"] == "scored"]


def comparable(cells: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    """The ok cells of `models` (default: all) at the ticker, origin and day
    every one of them forecast."""
    models = models or sorted(cells["model"].unique())
    subset = cells[cells["model"].isin(models)]
    ok = subset[subset["status"] == "ok"]
    counts = ok.groupby(KEY)["model"].nunique()
    shared = counts[counts == len(models)].index
    return ok.set_index(KEY).loc[lambda df: df.index.isin(shared)].reset_index()


def _summary(group: pd.DataFrame) -> dict:
    ape = metrics.absolute_percentage_errors(group["actual"], group["yhat"])
    move = (group["yhat"] - group["origin_close"]).abs() / group["origin_close"] * 100
    return {
        "n": len(group),
        "mape": float(np.mean(ape)),
        "mae": metrics.mae(group["actual"], group["yhat"]),
        "rmse": metrics.rmse(group["actual"], group["yhat"]),
        "ape_q25": float(np.percentile(ape, 25)),
        "ape_q50": float(np.percentile(ape, 50)),
        "ape_q75": float(np.percentile(ape, 75)),
        "directional_accuracy": metrics.directional_accuracy(
            group["origin_close"], group["actual"], group["yhat"]
        ),
        "up_share": metrics.up_share(group["origin_close"], group["actual"]),
        "move_median": float(move.median()),
        "band_coverage": metrics.band_coverage(
            group["actual"], group["yhat_lower"], group["yhat_upper"]
        ),
        "band_width": metrics.band_width(
            group["yhat"], group["yhat_lower"], group["yhat_upper"]
        ),
        "interval_score": metrics.interval_score(
            group["actual"], group["yhat_lower"], group["yhat_upper"]
        ),
    }


def summary_table(cells: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """One row per group of comparable cells, with skill against naive's MAE
    on the same cells. `by` includes "model"; e.g. ["ticker", "model", "h"]."""
    rows = [
        dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True))
        | _summary(group)
        for key, group in cells.groupby(by, sort=True)
    ]
    table = pd.DataFrame(rows)
    if table.empty or BASELINE not in set(table["model"]):
        return table
    others = [c for c in by if c != "model"]
    naive = table[table["model"] == BASELINE].set_index(others)["mae"]
    naive_mae = (
        table.set_index(others).index.map(naive)
        if others
        else np.full(len(table), naive.iloc[0])
    )
    table["skill"] = [
        metrics.skill(model, float(base))
        for model, base in zip(table["mae"], naive_mae, strict=True)
    ]
    return table


def primary_tests(cells: pd.DataFrame) -> pd.DataFrame:
    """Each model's primary test against naive (docs/evaluation.md, section 3).

    Per origin, the model's absolute percentage error minus naive's,
    averaged over every ticker and day; the mean of those per-origin numbers
    is tested with a Newey-West standard error, and the p-values are
    Holm-adjusted across the models.
    """
    if BASELINE not in set(cells["model"]):
        return pd.DataFrame()
    ape = cells.assign(
        ape=metrics.absolute_percentage_errors(cells["actual"], cells["yhat"])
    )
    wide = ape.pivot_table(index=KEY, columns="model", values="ape")
    rows = {}
    for model in wide.columns:
        if model == BASELINE:
            continue
        per_origin = (wide[model] - wide[BASELINE]).groupby(level="origin").mean()
        result = metrics.mean_difference_test(per_origin.to_numpy(), lag=1)
        if result is not None:
            rows[model] = result
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame.from_dict(rows, orient="index")
    adjusted = metrics.holm(table["p"].to_dict())
    table["p_holm"] = table.index.map(adjusted)
    table["beats_naive"] = (table["mean"] < 0) & (table["p_holm"] < SIGNIFICANCE)
    return table.rename_axis("model").reset_index()


def headline(cells: pd.DataFrame) -> pd.DataFrame:
    """One row per model across tickers: mean MAPE, mean and median skill,
    tickers with positive skill, and the primary test."""
    per_ticker = summary_table(cells, ["ticker", "model"])
    if per_ticker.empty:
        return per_ticker
    grouped = per_ticker.groupby("model")
    table = pd.DataFrame(
        {
            "tickers": grouped["ticker"].nunique(),
            "mean_mape": grouped["mape"].mean(),
            "mean_skill": grouped["skill"].mean(),
            "median_skill": grouped["skill"].median(),
            "tickers_positive_skill": grouped["skill"].apply(
                lambda s: int((s > 0).sum())
            ),
        }
    ).reset_index()
    tests = primary_tests(cells)
    if not tests.empty:
        table = table.merge(
            tests[["model", "mean", "p", "p_holm", "beats_naive"]].rename(
                columns={"mean": "mean_ape_difference"}
            ),
            on="model",
            how="left",
        )
    return table.sort_values("mean_mape").reset_index(drop=True)


def failures(cells: pd.DataFrame) -> pd.DataFrame:
    """Per model: cells, failed cells, the failed share and an example."""
    grouped = cells.groupby("model")
    table = pd.DataFrame(
        {
            "cells": grouped.size(),
            "failed": grouped["status"].apply(lambda s: int((s == "failed").sum())),
        }
    )
    table["share"] = table["failed"] / table["cells"]
    first_error = cells[cells["status"] == "failed"].groupby("model")["error"].first()
    table["example"] = table.index.map(first_error).fillna("")
    return table.reset_index()


def news_coverage(
    articles: pd.DataFrame,
    trading_dates,
    tickers: list[str],
    windows: dict[str, list],
) -> pd.DataFrame:
    """The share of (ticker, date) rows with at least one real article.

    windows maps a name to its scored origin dates. For each window, two
    rows per ticker and one for all tickers: the origin dates themselves,
    and the training history up to the window's first origin (every
    trading date before it), since a model learns from the latter.
    """
    pairs = dates_with_news(articles, trading_dates)
    with_news = set(zip(pairs["ticker"], pairs["date"], strict=True))
    dates = pd.DatetimeIndex(trading_dates)
    rows = []
    for window, origins in windows.items():
        origins = pd.DatetimeIndex(origins)
        spans = {"origins": origins, "history": dates[dates < origins.min()]}
        for span, span_dates in spans.items():
            for ticker in [*tickers, "all"]:
                group = tickers if ticker == "all" else [ticker]
                total = len(span_dates) * len(group)
                hits = sum((t, d) in with_news for t in group for d in span_dates)
                rows.append(
                    {"window": window, "span": span, "ticker": ticker,
                     "rows": total, "with_news": hits,
                     "share": hits / total if total else np.nan}
                )  # fmt: skip
    return pd.DataFrame(rows)


def news_ablation_status(coverage: pd.DataFrame) -> str:
    """ "testable" if at least NEWS_COVERAGE_NEEDED of the validation origin
    rows have a real article, else "not enough data" (decision D15)."""
    row = coverage[
        (coverage["window"] == "validation")
        & (coverage["span"] == "origins")
        & (coverage["ticker"] == "all")
    ]
    if row.empty or not row["share"].iloc[0] >= NEWS_COVERAGE_NEEDED:
        return "not enough data"
    return "testable"


def markdown_table(df: pd.DataFrame, digits: int = 3) -> str:
    if df.empty:
        return "(none)\n"

    def cell(value) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        if isinstance(value, pd.Timestamp):
            return value.date().isoformat()
        return str(value)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |"]
    lines.append("|" + "---|" * len(df.columns))
    lines += [
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join(lines) + "\n"


def markdown(
    header: dict, cells: pd.DataFrame, coverage: pd.DataFrame | None = None
) -> str:
    """The run's report: what was run, then the tables, all from `cells`."""
    rows = scored(cells)
    compared = comparable(rows)
    parts = ["# Rolling-origin evaluation\n"]
    parts += [f"- {key}: {value}" for key, value in header.items()]
    parts.append(
        '\nThe headline\'s primary test is the only basis for "beats naive"; '
        "every per-ticker and per-horizon figure below is descriptive.\n"
    )
    parts += ["## Headline\n", markdown_table(headline(compared))]
    parts += [
        "## Per ticker\n",
        markdown_table(summary_table(compared, ["ticker", "model"])),
    ]
    parts += [
        "## Per horizon\n",
        markdown_table(summary_table(compared, ["model", "h"])),
    ]
    parts += ["## Failures\n", markdown_table(failures(rows))]
    if coverage is not None:
        parts += ["## News coverage\n", markdown_table(coverage)]
        parts.append(f"News ablation: {news_ablation_status(coverage)}.\n")
    return "\n".join(parts)
