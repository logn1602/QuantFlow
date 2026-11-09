"""What the README and the dashboard show from the evaluation runs.

Built only from a run folder's own tables (headline.csv, per_ticker.csv,
per_horizon.csv) and its report header, so every published number traces
back to a run and its cells (docs/evaluation.md, section 17):

  - accuracy_table: per window, ticker and model, the MAPE, the skill
    against naive and the band coverage, with the run each came from. The
    dashboard reads it (quantflow/dashboard/published_accuracy.csv).
  - horizon_chart: the README's chart, MAPE by days ahead, each model in its
    own panel against naive ("no change").

pipelines/evaluate.py's publish command writes both from a validation run
and a test-date run.
"""

import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quantflow.evaluation.report import BASELINE
from quantflow.evaluation.splits import HORIZON

ALL_TICKERS = "ALL"

# Display names, in the order the chart's panels and the tables use when
# MAPE does not decide it.
LABELS = {
    BASELINE: "No change",
    "arima": "ARIMA",
    "drift": "Drift",
    "prophet": "Prophet",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "ensemble_stack": "Ensemble (live weights)",
    "ensemble_rolling": "Ensemble (rolling weights)",
}

COLUMNS = [
    "window", "ticker", "model", "cells", "mape", "skill", "band_coverage",
    "beats_naive", "p_holm", "run", "commit", "as_of", "origins",
    "first_origin", "last_origin",
]  # fmt: skip

_ORIGINS = re.compile(r"^(\d+), every \d+ trading days, (\S+) to (\S+)$")


@dataclass(frozen=True)
class RunTables:
    """One run folder's header and the tables publishing needs."""

    name: str
    header: dict
    headline: pd.DataFrame
    per_ticker: pd.DataFrame
    per_horizon: pd.DataFrame

    @property
    def window(self) -> str:
        return self.header.get("window", "")

    @property
    def origins(self) -> tuple[int, str, str]:
        """(count, first, last) from the header's "scored origins" line."""
        match = _ORIGINS.match(self.header.get("scored origins", ""))
        if not match:
            raise ValueError(
                f"run {self.name}: unreadable scored origins "
                f"{self.header.get('scored origins')!r}"
            )
        return int(match.group(1)), match.group(2), match.group(3)


def read_run(directory: Path, header: dict) -> RunTables:
    return RunTables(
        name=directory.name,
        header=header,
        headline=pd.read_csv(directory / "headline.csv"),
        per_ticker=pd.read_csv(directory / "per_ticker.csv"),
        per_horizon=pd.read_csv(directory / "per_horizon.csv"),
    )


def accuracy_table(runs: list[RunTables]) -> pd.DataFrame:
    """Per run (window), one row per ticker and model from per_ticker.csv,
    and one ALL row per model from headline.csv: its mean MAPE and skill
    over tickers and its primary test. ALL's band coverage is the share over
    every cell, from the per-ticker shares weighted by their cells."""
    frames = []
    for run in runs:
        origins, first, last = run.origins
        provenance = {
            "window": run.window,
            "run": run.name,
            "commit": run.header.get("commit", "")[:7],
            "as_of": run.header.get("as of", ""),
            "origins": origins,
            "first_origin": first,
            "last_origin": last,
        }
        tickers = run.per_ticker.assign(cells=run.per_ticker["n"])[
            ["ticker", "model", "cells", "mape", "skill", "band_coverage"]
        ]
        pooled = []
        for model, group in tickers.groupby("model", sort=False):
            covered = group.dropna(subset=["band_coverage"])
            pooled.append(
                {
                    "model": model,
                    "cells": int(group["cells"].sum()),
                    "band_coverage": (
                        float(np.average(covered["band_coverage"],
                                         weights=covered["cells"]))
                        if not covered.empty else np.nan
                    ),
                }
            )  # fmt: skip
        headline = run.headline.rename(columns={"mean_mape": "mape",
                                                "mean_skill": "skill"})  # fmt: skip
        everything = (
            pd.DataFrame(pooled)
            .merge(
                headline[["model", "mape", "skill", "beats_naive", "p_holm"]],
                on="model",
                how="left",
            )
            .assign(ticker=ALL_TICKERS)
        )
        frames.append(pd.concat([everything, tickers]).assign(**provenance))
    table = pd.concat(frames, ignore_index=True)
    # Full precision: rounding here and again where the figures are shown
    # would print 2.40% for a MAPE of 2.39497%, where the docs say 2.39.
    for column in ("mape", "skill", "band_coverage", "p_holm"):
        table[column] = table[column].astype(float)
    table["beats_naive"] = table["beats_naive"].map(_flag)
    order = {name: i for i, name in enumerate(LABELS)}
    table = table.sort_values(
        ["window", "ticker", "model"],
        key=lambda s: s.map(order) if s.name == "model" else s,
    )
    return table[COLUMNS].reset_index(drop=True)


def _flag(value) -> str:
    """The headline's beats_naive as "true", "false" or "" (no test, as for
    naive itself). Spelled out rather than bool(value), which reads the
    string "False" as true."""
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if text not in {"true", "false"}:
        raise ValueError(f"unreadable beats_naive value {value!r}")
    return text


def beats_naive_note(runs: list[RunTables]) -> str:
    """One sentence on which models beat naive by the primary test, per run."""
    winners = []
    for run in runs:
        headline = run.headline
        flagged = headline[headline["beats_naive"].map(_flag) == "true"]
        winners += [
            f"{LABELS.get(m, m)} ({run.window} dates)" for m in flagged["model"]
        ]
    if not winners:
        return 'No model beats "no change" on either set of dates.'
    return 'Beats "no change": ' + ", ".join(winners) + "."


# Colors from the data-viz reference palette (light surface): the models in
# series blue, the "no change" yardstick in neutral ink, text in text ink.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
MODEL_COLOR = "#2a78d6"
BASELINE_COLOR = "#52514e"
# The test dates dashed: an offset and an on/off pattern, in line widths.
TEST_DASH: tuple[float, tuple[float, float]] = (0.0, (3.0, 2.0))
LINE_STYLES: dict[str, object] = {"validation": "-", "test": TEST_DASH}
# One panel per model; the last cell holds the key.
ROWS, COLUMNS_OF_PANELS = 2, 4


def horizon_chart(runs: list[RunTables], path: Path) -> None:
    """MAPE by days ahead, one panel per model against naive, the validation
    run solid and the test-date run dashed, on one shared y-axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    by_window = {run.window: run for run in runs}
    validation = by_window["validation"]
    models = [
        m
        for m in validation.headline.sort_values("mean_mape")["model"]
        if m != BASELINE and all(m in set(run.per_horizon["model"]) for run in runs)
    ]
    room = ROWS * COLUMNS_OF_PANELS - 1
    if len(models) > room:
        raise ValueError(f"{len(models)} models; the chart has room for {room}")

    fig, axes = plt.subplots(
        ROWS, COLUMNS_OF_PANELS, figsize=(8.8, 5.6), sharex=True, sharey=True,
        facecolor=SURFACE,
    )  # fmt: skip
    top = max(float(run.per_horizon["mape"].max()) for run in runs)
    for ax, model in zip(axes.flat, models, strict=False):
        ax.set_facecolor(SURFACE)
        for window, style in LINE_STYLES.items():
            if window not in by_window:
                continue
            table = by_window[window].per_horizon.sort_values("h")
            for name, color in ((BASELINE, BASELINE_COLOR), (model, MODEL_COLOR)):
                line = table[table["model"] == name]
                ax.plot(line["h"], line["mape"], color=color, linewidth=1.5,
                        linestyle=style, solid_capstyle="round",
                        dash_capstyle="round", solid_joinstyle="round")  # fmt: skip
        ax.set_title(LABELS.get(model, model), loc="left", fontsize=9.5, color=TEXT)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    axes[0, 0].set_ylim(0, np.ceil(top))
    axes[0, 0].set_xticks(range(1, HORIZON + 1))
    for ax in axes[1, :]:
        ax.set_xlabel("Days ahead", fontsize=8.5, color=TEXT_SECONDARY)
    for ax in axes[:, 0]:
        ax.set_ylabel("MAPE %", fontsize=8.5, color=TEXT_SECONDARY)

    key = axes.flat[-1]
    key.axis("off")
    handles = [
        Line2D([], [], color=MODEL_COLOR, linewidth=1.5, linestyle="-",
               label="Model, validation dates"),
        Line2D([], [], color=MODEL_COLOR, linewidth=1.5, linestyle=TEST_DASH,
               label="Model, test dates"),
        Line2D([], [], color=BASELINE_COLOR, linewidth=1.5, linestyle="-",
               label='"No change", validation'),
        Line2D([], [], color=BASELINE_COLOR, linewidth=1.5,
               linestyle=TEST_DASH, label='"No change", test'),
    ]  # fmt: skip
    legend = key.legend(handles=handles, loc="upper left", frameon=False,
                        fontsize=8, labelcolor=TEXT, handlelength=2.6)  # fmt: skip
    legend.set_in_layout(False)
    note = textwrap.fill(beats_naive_note(runs), 30, break_long_words=False)
    key.text(0.0, 0.32, "Lower is better.\n" + note,
             transform=key.transAxes, fontsize=8, color=TEXT_SECONDARY,
             va="top", linespacing=1.4)  # fmt: skip

    count, first, last = validation.origins
    lines = [
        f'{HORIZON}-day forecasts: error by days ahead, each model against "no change"',
        f"MAPE over {validation.per_ticker['ticker'].nunique()} tickers, "
        f"data to {validation.header.get('as of', '?')}. "
        f"Validation: {count} start dates, {first} to {last}.",
    ]
    if "test" in by_window:
        count, first, last = by_window["test"].origins
        lines[1] += f" Test: {count} start dates, {first} to {last}."
    fig.text(0.01, 0.985, lines[0], fontsize=11.5, color=TEXT, va="top",
             fontweight="bold")  # fmt: skip
    fig.text(0.01, 0.935, lines[1], fontsize=8, color=TEXT_SECONDARY, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=1.6, w_pad=1.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path, dpi=160, facecolor=SURFACE, metadata={"Software": None},
        pil_kwargs={"optimize": True},
    )  # fmt: skip
    plt.close(fig)
