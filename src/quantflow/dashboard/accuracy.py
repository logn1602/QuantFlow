"""The accuracy the dashboard shows, from the published evaluation table.

published_accuracy.csv, next to this file, is written by
`python -m quantflow.pipelines.evaluate publish` from two committed runs of
the rolling-origin evaluation (docs/evaluation.md, section 17): the 7-day
forecast's error on the validation dates and on the test dates, per ticker
and for all tickers. The daily run cannot measure this itself: it fits each
model once and deletes the previous day's forecasts.

Kept apart from app.py, which reads the database when imported, so these
functions can be tested offline.
"""

from pathlib import Path

import pandas as pd

PUBLISHED = Path(__file__).with_name("published_accuracy.csv")
BASELINE = "naive"
ALL_TICKERS = "ALL"

# The forecasts the dashboard draws, in its order. Drift and the rolling
# ensemble are evaluated but never shown, so they are not listed here.
SHOWN = {
    BASELINE: "No change",
    "arima": "ARIMA",
    "prophet": "Prophet",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "ensemble_stack": "Ensemble",
}


def load_published(path: Path = PUBLISHED) -> pd.DataFrame | None:
    """The published table, or None when it is missing or unreadable, so the
    page can say so instead of failing."""
    try:
        table = pd.read_csv(path, dtype={"beats_naive": str})
    except (OSError, ValueError):
        return None
    needed = {"window", "ticker", "model", "mape", "skill", "band_coverage"}
    return table if needed <= set(table.columns) else None


def _percent(value) -> str:
    """A band coverage, already a percentage (evaluation.metrics)."""
    return "" if pd.isna(value) else f"{value:.0f}%"


def accuracy_rows(published: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """One row per shown model for `ticker`: MAPE on the validation and test
    dates, skill against no change (validation) and band coverage."""
    rows = published[published["ticker"] == ticker]

    def value(model: str, window: str, column: str):
        match = rows[(rows["model"] == model) & (rows["window"] == window)]
        return match[column].iloc[0] if not match.empty else float("nan")

    out = []
    for model, label in SHOWN.items():
        out.append(
            {
                "Model": label,
                "MAPE, validation": value(model, "validation", "mape"),
                "MAPE, test": value(model, "test", "mape"),
                "Skill vs no change": (
                    None if model == BASELINE else value(model, "validation", "skill")
                ),
                "Band held the close": (
                    "no band"
                    if model == BASELINE
                    else f"{_percent(value(model, 'validation', 'band_coverage'))}"
                    f" / {_percent(value(model, 'test', 'band_coverage'))}"
                ),
            }
        )
    table = pd.DataFrame(out)
    for column in ("MAPE, validation", "MAPE, test"):
        table[column] = table[column].map(lambda v: "" if pd.isna(v) else f"{v:.2f}%")
    table["Skill vs no change"] = table["Skill vs no change"].map(
        lambda v: "" if v is None or pd.isna(v) else f"{v:+.3f}"
    )
    return table


def band_caption(published: pd.DataFrame) -> str:
    """Each shown band's measured coverage over all tickers, validation / test."""
    rows = published[published["ticker"] == ALL_TICKERS]
    parts = []
    for model, label in SHOWN.items():
        if model == BASELINE:
            continue
        coverage = {
            window: rows[(rows["model"] == model) & (rows["window"] == window)][
                "band_coverage"
            ]
            for window in ("validation", "test")
        }
        if coverage["validation"].empty:
            continue
        test = coverage["test"].iloc[0] if not coverage["test"].empty else None
        parts.append(
            f"{label} {_percent(coverage['validation'].iloc[0])}"
            + (f" / {_percent(test)}" if test is not None else "")
        )
    return (
        "Shaded bands: the share of actual closes that fell inside each band, "
        "validation / test dates, all tickers: "
        + ", ".join(parts)
        + ". Measured, not a promised 95%."
    )


def baseline_note(published: pd.DataFrame) -> str:
    """Says every model is measured against no change, and which (if any)
    beats it by the evaluation's primary test, on either set of dates."""
    rows = published[
        (published["ticker"] == ALL_TICKERS) & published["model"].isin(SHOWN)
    ]
    winners = rows[rows["beats_naive"].astype(str).str.lower() == "true"]
    lead = (
        'Every model is measured against "no change" (the latest close, '
        "held for all 7 days)"
    )
    if winners.empty:
        return lead + "; none currently beats it."
    named = ", ".join(
        f"{SHOWN[m]} ({w} dates)"
        for m, w in zip(winners["model"], winners["window"], strict=True)
    )
    return f"{lead}. Beats it: {named}."


def provenance(published: pd.DataFrame) -> str:
    """Where the published figures come from, for a caption."""
    parts = []
    for window in ("validation", "test"):
        rows = published[published["window"] == window]
        if rows.empty:
            continue
        row = rows.iloc[0]
        parts.append(
            f"{window}: {row['origins']} start dates, "
            f"{row['first_origin']} to {row['last_origin']} (run {row['run']})"
        )
    as_of = published["as_of"].iloc[0] if "as_of" in published else "?"
    return (
        f"Rolling-origin evaluation of the 7-day forecast on data to {as_of}; "
        + "; ".join(parts)
        + ". Lower MAPE is better; skill above 0 would beat no change."
    )
