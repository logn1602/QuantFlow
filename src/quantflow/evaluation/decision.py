"""The Phase 3 keep-or-drop rule (docs/evaluation.md, section 16), as code.

Written before any experiment ran, so every decision is computed the same
way from two runs' cells rather than read off a table:

  - one model's scored cells in the incumbent's run and the candidate's, on
    the validation dates; the two runs must cover the same cells;
  - dropped if the candidate fails more cells than the incumbent, or more
    than the harness's limit;
  - otherwise compared on the cells both forecast: kept only if its MAPE
    over all of them is strictly lower AND its MAPE is lower on at least
    TICKERS_NEEDED tickers. A tie drops.

Everything else here (by horizon, MAE, bands, the paired test) is reported,
not deciding. The test dates are never used to choose (decision D13), so a
comparison on them is refused.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantflow.evaluation import metrics
from quantflow.evaluation.report import KEY
from quantflow.evaluation.rolling_origin import MAX_FAILURE_SHARE

# Of the 8 tickers: a change must help most of them, not one or two.
TICKERS_NEEDED = 5
WINDOW = "validation"


@dataclass(frozen=True)
class Comparison:
    """One model, incumbent against candidate."""

    model: str
    keep: bool
    reason: str
    cells: int  # compared: ok in both runs
    mape_incumbent: float
    mape_candidate: float
    tickers: int
    tickers_better: int
    failed_incumbent: int
    failed_candidate: int
    per_ticker: pd.DataFrame
    per_horizon: pd.DataFrame
    # Per origin, the candidate's absolute percentage error minus the
    # incumbent's, averaged over tickers and days; Newey-West, lag 1.
    paired_test: dict | None


def _model_cells(cells: pd.DataFrame, model: str, side: str) -> pd.DataFrame:
    rows = cells[(cells["model"] == model) & (cells["role"] == "scored")]
    if rows.empty:
        raise ValueError(f"the {side} run has no scored {model} cells")
    windows = set(rows["window"])
    if windows != {WINDOW}:
        raise ValueError(
            f"the {side} run's {model} cells are on {sorted(windows)}; decisions "
            f"use the {WINDOW} dates only (decision D13)"
        )
    rows = rows.assign(origin=pd.to_datetime(rows["origin"]))
    if rows.duplicated(KEY).any():
        raise ValueError(f"the {side} run has {model} cells twice")
    return rows


def _summary(group: pd.DataFrame, suffix: str) -> dict:
    yhat, actual = group[f"yhat{suffix}"], group["actual"]
    return {
        f"mape{suffix}": metrics.mape(actual, yhat),
        f"mae{suffix}": metrics.mae(actual, yhat),
        f"coverage{suffix}": metrics.band_coverage(
            actual, group[f"yhat_lower{suffix}"], group[f"yhat_upper{suffix}"]
        ),
        f"interval_score{suffix}": metrics.interval_score(
            actual, group[f"yhat_lower{suffix}"], group[f"yhat_upper{suffix}"]
        ),
        f"directional{suffix}": metrics.directional_accuracy(
            group["origin_close"], actual, yhat
        ),
    }


def _table(both: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, group in both.groupby(by, sort=True):
        rows.append(
            {by: key, "n": len(group)}
            | _summary(group, "_incumbent")
            | _summary(group, "_candidate")
        )
    table = pd.DataFrame(rows)
    table["better"] = table["mape_candidate"] < table["mape_incumbent"]
    return table


def compare(
    incumbent: pd.DataFrame,
    candidate: pd.DataFrame,
    model: str,
    tickers_needed: int = TICKERS_NEEDED,
    failure_limit: float = MAX_FAILURE_SHARE,
) -> Comparison:
    """The rule applied to `model`, from each run's cells (cells.csv)."""
    old = _model_cells(incumbent, model, "incumbent")
    new = _model_cells(candidate, model, "candidate")
    old_keys = set(map(tuple, old[KEY].astype(str).to_numpy()))
    new_keys = set(map(tuple, new[KEY].astype(str).to_numpy()))
    if old_keys != new_keys:
        raise ValueError(
            f"the runs do not cover the same {model} cells "
            f"({len(old_keys)} against {len(new_keys)}, "
            f"{len(old_keys ^ new_keys)} differ)"
        )

    failed_old = int(old["status"].eq("failed").sum())
    failed_new = int(new["status"].eq("failed").sum())
    columns = ["actual", "origin_close", "yhat", "yhat_lower", "yhat_upper"]
    both = (
        old[old["status"] == "ok"][KEY + columns]
        .merge(
            new[new["status"] == "ok"][KEY + columns].drop(
                columns=["actual", "origin_close"]
            ),
            on=KEY,
            suffixes=("_incumbent", "_candidate"),
        )
        .sort_values(KEY, kind="stable")
        .reset_index(drop=True)
    )
    # The failure rules come first: they decide without any shared cell.
    total = len(new)
    drop_for_failures = None
    if failed_new / total > failure_limit:
        drop_for_failures = (
            f"drop: fails {failed_new} of {total} cells, over the "
            f"{failure_limit:.0%} limit"
        )
    elif failed_new > failed_old:
        drop_for_failures = (
            f"drop: fails {failed_new} cells, the incumbent {failed_old}"
        )
    if both.empty:
        if drop_for_failures is None:
            raise ValueError(
                f"no cells both runs forecast for {model}: the incumbent "
                f"failed {failed_old} of {len(old)}"
            )
        return Comparison(
            model=model, keep=False, reason=drop_for_failures, cells=0,
            mape_incumbent=np.nan, mape_candidate=np.nan, tickers=0,
            tickers_better=0, failed_incumbent=failed_old,
            failed_candidate=failed_new, per_ticker=pd.DataFrame(),
            per_horizon=pd.DataFrame(), paired_test=None,
        )  # fmt: skip

    per_ticker = _table(both, "ticker")
    per_horizon = _table(both, "h")
    mape_old = metrics.mape(both["actual"], both["yhat_incumbent"])
    mape_new = metrics.mape(both["actual"], both["yhat_candidate"])
    tickers = len(per_ticker)
    better = int(per_ticker["better"].sum())

    ape_old = metrics.absolute_percentage_errors(both["actual"], both["yhat_incumbent"])
    ape_new = metrics.absolute_percentage_errors(both["actual"], both["yhat_candidate"])
    per_origin = pd.Series(ape_new - ape_old).groupby(both["origin"]).mean()
    paired = metrics.mean_difference_test(per_origin.to_numpy(), lag=1)

    keep = False
    if drop_for_failures is not None:
        reason = drop_for_failures
    elif not mape_new < mape_old:
        reason = (
            f"drop: MAPE {mape_new:.4f}% is not below the incumbent's {mape_old:.4f}%"
        )
    elif better < tickers_needed:
        reason = (
            f"drop: MAPE lower on {better} of {tickers} tickers, "
            f"{tickers_needed} needed"
        )
    else:
        keep = True
        reason = (
            f"keep: MAPE {mape_new:.4f}% against {mape_old:.4f}%, lower on "
            f"{better} of {tickers} tickers"
        )
    return Comparison(
        model=model,
        keep=keep,
        reason=reason,
        cells=len(both),
        mape_incumbent=mape_old,
        mape_candidate=mape_new,
        tickers=tickers,
        tickers_better=better,
        failed_incumbent=failed_old,
        failed_candidate=failed_new,
        per_ticker=per_ticker,
        per_horizon=per_horizon,
        paired_test=paired,
    )


def summary_row(result: Comparison) -> dict:
    """One line per model for a decisions table."""
    test = result.paired_test or {}
    return {
        "model": result.model,
        "decision": "keep" if result.keep else "drop",
        "cells": result.cells,
        "mape_incumbent": result.mape_incumbent,
        "mape_candidate": result.mape_candidate,
        "tickers_better": f"{result.tickers_better} of {result.tickers}",
        "failed_incumbent": result.failed_incumbent,
        "failed_candidate": result.failed_candidate,
        "paired_mean_difference": test.get("mean", np.nan),
        "paired_p": test.get("p", np.nan),
        "reason": result.reason,
    }
