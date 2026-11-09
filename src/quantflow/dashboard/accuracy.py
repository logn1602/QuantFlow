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
# The table's schema is evaluation/publish.py's (COLUMNS, ALL_TICKERS) and
# naive is evaluation/report.py's BASELINE. They are restated here rather than
# imported so the page does not load the scoring code;
# tests/unit/test_dashboard_accuracy.py checks they agree.
BASELINE = "naive"
ALL_TICKERS = "ALL"
WINDOWS = ("validation", "test")
# Column headings of the accuracy table, which say what each figure covers.
SKILL = "Skill vs no change (validation)"
BAND = "Band held the close (validation / test)"
# Every column this module reads.
READ_COLUMNS = {
    "window", "ticker", "model", "mape", "skill", "band_coverage", "beats_naive",
    "run", "as_of", "origins", "first_origin", "last_origin",
}  # fmt: skip

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
    # An empty table would pass the column check and then fail the caption on
    # its first row, stopping every tab after the Forecasts tab.
    if table.empty or not READ_COLUMNS.issubset(table.columns):
        return None
    return table


def _cell(rows: pd.DataFrame, model: str, window: str, column: str):
    """One model's figure for one window, or NaN when the table lacks it."""
    match = rows[(rows["model"] == model) & (rows["window"] == window)]
    return match[column].iloc[0] if not match.empty else float("nan")


def _coverage(rows: pd.DataFrame, model: str) -> str:
    """A band's measured coverage, validation / test. The values are already
    percentages (evaluation.metrics.band_coverage); a missing one is "?"."""
    shares = [_cell(rows, model, window, "band_coverage") for window in WINDOWS]
    return " / ".join("?" if pd.isna(v) else f"{v:.0f}%" for v in shares)


def accuracy_rows(published: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """One row per shown model for `ticker`: MAPE on the validation and test
    dates, skill against no change (validation) and band coverage. No model
    is singled out; the one mark is "(lowest)" on the lowest MAPE in each
    window, often "no change"'s."""
    rows = published[published["ticker"] == ticker]
    out = []
    for model, label in SHOWN.items():
        out.append(
            {
                "Model": label,
                "MAPE, validation": _cell(rows, model, "validation", "mape"),
                "MAPE, test": _cell(rows, model, "test", "mape"),
                SKILL: (
                    None
                    if model == BASELINE
                    else _cell(rows, model, "validation", "skill")
                ),
                BAND: ("no band" if model == BASELINE else _coverage(rows, model)),
            }
        )
    table = pd.DataFrame(out)
    for column in ("MAPE, validation", "MAPE, test"):
        lowest = table[column].min()
        table[column] = table[column].map(
            lambda v, lowest=lowest: (
                "" if pd.isna(v) else f"{v:.2f}%" + (" (lowest)" if v == lowest else "")
            )
        )
    table[SKILL] = table[SKILL].map(
        lambda v: "" if v is None or pd.isna(v) else f"{v:+.3f}"
    )
    return table


def band_caption(published: pd.DataFrame) -> str:
    """Each shown band's measured coverage over all tickers, validation / test."""
    rows = published[published["ticker"] == ALL_TICKERS]
    parts = [
        f"{label} {_coverage(rows, model)}"
        for model, label in SHOWN.items()
        if model != BASELINE
    ]
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
    for window in WINDOWS:
        rows = published[published["window"] == window]
        if rows.empty:
            continue
        row = rows.iloc[0]
        parts.append(
            f"{window}: {row['origins']} start dates, "
            f"{row['first_origin']} to {row['last_origin']} (run {row['run']})"
        )
    as_of = published["as_of"].iloc[0]
    return (
        f"Rolling-origin evaluation of the 7-day forecast on data to {as_of}; "
        + "; ".join(parts)
        + ". Lower MAPE is better. Skill above 0 means a lower error than no "
        "change on this ticker's dates; beating no change rests on the "
        "evaluation's one test over all tickers, as the note above says."
    )


def covers(published: pd.DataFrame, ticker: str) -> bool:
    """Whether the published evaluation scored `ticker` at all."""
    return ticker in set(published["ticker"])


def no_change_forecast(prices: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """ "No change" over the forecasts' dates: the close the forecasts started
    from (the last one before their first date), held. The first row is that
    close on its own date, so a chart's line starts at an actual price; the
    rest are the forecast. Anchored there rather than at the newest close: on
    a day prices arrived but the models did not run, the newest close is
    already one of the forecasts' dates. Empty without prices or forecasts."""
    if prices.empty or forecasts.empty:
        return pd.DataFrame({"date": [], "close": []})
    future = sorted(pd.to_datetime(forecasts["forecast_date"]).unique())
    before = prices[pd.to_datetime(prices["date"]) < future[0]]
    if before.empty:
        return pd.DataFrame({"date": [], "close": []})
    start = before.sort_values("date").iloc[-1]
    return pd.DataFrame(
        {"date": [start["date"], *future], "close": float(start["close"])}
    )
