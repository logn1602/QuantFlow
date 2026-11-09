"""tests/unit/test_dashboard_accuracy.py

Covers dashboard/accuracy.py, what the Forecasts tab says about accuracy:
the 7-day error measured by the rolling-origin evaluation, every shown model
beside "no change", each band labelled with its measured coverage. Also
checks the committed table itself, and that it ships with the package, since
Streamlit Cloud installs the package rather than running from the checkout.
"""

import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from quantflow.dashboard import accuracy

ROOT = Path(__file__).resolve().parents[2]


def _published(winner=None):
    rows = []
    for window in ("validation", "test"):
        for ticker in ("ALL", "AAPL"):
            for i, model in enumerate(accuracy.SHOWN):
                rows.append(
                    {"window": window, "ticker": ticker, "model": model,
                     "mape": 2.88 + 0.1 * i, "skill": -0.01 * i,
                     "band_coverage": np.nan if model == "naive" else 40.0 + 10 * i,
                     "beats_naive": (
                         "" if model == "naive" or ticker != "ALL"
                         else str((model, window) == winner).lower()
                     ),
                     "run": "P4" if window == "validation" else "T3",
                     "as_of": "2026-11-06", "origins": 42 if window == "validation" else 25,
                     "first_origin": "2025-06-30", "last_origin": "2026-04-23"}
                )  # fmt: skip
    return pd.DataFrame(rows)


def test_a_missing_or_malformed_table_reads_as_none(tmp_path):
    assert accuracy.load_published(tmp_path / "missing.csv") is None
    (tmp_path / "bad.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    assert accuracy.load_published(tmp_path / "bad.csv") is None


def test_every_shown_model_is_listed_beside_no_change_for_the_ticker():
    rows = accuracy.accuracy_rows(_published(), "AAPL")
    assert rows["Model"].tolist() == list(accuracy.SHOWN.values())
    no_change = rows.iloc[0]
    assert no_change["Model"] == "No change"
    assert no_change["MAPE, validation"] == "2.88%"
    assert no_change[accuracy.SKILL] == ""
    assert no_change[accuracy.BAND] == "no band"
    arima = rows.iloc[1]
    assert arima[accuracy.SKILL] == "-0.010"
    assert arima[accuracy.BAND] == "50% / 50%"


def test_every_band_is_labelled_with_its_measured_coverage():
    caption = accuracy.band_caption(_published())
    for label in ("ARIMA 50% / 50%", "Prophet 60%", "Ensemble 90% / 90%"):
        assert label in caption
    assert "No change" not in caption
    assert "not a promised 95%" in caption


def test_the_note_says_none_beats_no_change_unless_the_test_says_one_does():
    note = accuracy.baseline_note(_published())
    assert note.startswith('Every model is measured against "no change"')
    assert note.endswith("none currently beats it.")
    note = accuracy.baseline_note(_published(winner=("xgboost", "test")))
    assert note.endswith("Beats it: XGBoost (test dates).")


def test_the_committed_table_covers_every_ticker_and_shown_model_in_both_windows():
    """Checked against the table itself, not config.TICKERS: CI and the daily
    workflow set their own ticker lists, and the table describes the
    evaluation's tickers whatever they are."""
    published = accuracy.load_published()
    assert published is not None
    assert set(published["window"]) == {"validation", "test"}
    assert published["as_of"].nunique() == 1
    tickers = {
        window: set(rows["ticker"]) for window, rows in published.groupby("window")
    }
    assert tickers["validation"] == tickers["test"]
    assert accuracy.ALL_TICKERS in tickers["validation"]
    assert len(tickers["validation"]) > 1
    for window, rows in published.groupby("window"):
        for ticker, group in rows.groupby("ticker"):
            assert set(accuracy.SHOWN) <= set(group["model"]), (window, ticker)
    shown = published[published["model"].isin(accuracy.SHOWN)]
    assert (shown["mape"] > 0).all() and shown["mape"].notna().all()
    # Band coverage is a percentage (evaluation.metrics.band_coverage), which the
    # captions print as it is; naive has no band.
    banded = shown[shown["model"] != accuracy.BASELINE]["band_coverage"]
    assert banded.notna().all() and banded.between(1, 100).all()
    assert shown[shown["model"] == accuracy.BASELINE]["band_coverage"].isna().all()


def test_the_schema_restated_here_matches_the_one_publish_writes():
    """accuracy.py restates the table's schema instead of importing the
    scoring code into the page; this keeps the two from drifting."""
    from quantflow.evaluation import publish, report

    assert accuracy.READ_COLUMNS.issubset(publish.COLUMNS)
    assert accuracy.ALL_TICKERS == publish.ALL_TICKERS
    assert accuracy.BASELINE == report.BASELINE
    assert set(accuracy.SHOWN) <= set(publish.LABELS)


def test_a_table_missing_a_column_the_page_reads_is_refused(tmp_path):
    path = tmp_path / "published.csv"
    _published().drop(columns=["run"]).to_csv(path, index=False)
    assert accuracy.load_published(path) is None
    _published().to_csv(path, index=False)
    assert accuracy.load_published(path) is not None


def test_a_missing_coverage_figure_shows_as_unknown_in_the_table_and_caption():
    published = _published()
    gone = (published["model"] == "arima") & (published["window"] == "test")
    published = published[~gone]
    rows = accuracy.accuracy_rows(published, "AAPL")
    assert rows.loc[rows["Model"] == "ARIMA", accuracy.BAND].item() == "50% / ?"
    assert "ARIMA 50% / ?," in accuracy.band_caption(published)


def test_no_change_holds_the_latest_close_over_the_forecast_dates():
    prices = pd.DataFrame(
        {"date": pd.to_datetime(["2026-11-05", "2026-11-06"]), "close": [101.0, 99.5]}
    )
    forecasts = pd.DataFrame(
        {"model": ["arima", "arima", "xgboost", "xgboost"],
         "forecast_date": pd.to_datetime(["2026-11-10", "2026-11-09"] * 2)}
    )  # fmt: skip
    line = accuracy.no_change_forecast(prices, forecasts)
    assert line["date"].tolist() == list(
        pd.to_datetime(["2026-11-06", "2026-11-09", "2026-11-10"])
    )
    assert line["close"].tolist() == [99.5, 99.5, 99.5]
    assert accuracy.no_change_forecast(prices, forecasts.iloc[0:0]).empty
    assert accuracy.no_change_forecast(prices.iloc[0:0], forecasts).empty


def test_no_change_starts_from_the_close_the_forecasts_started_from():
    """On a day prices arrived but the models did not run, the newest close
    is already one of the stored forecasts' dates; "no change" for those
    forecasts is the close before them."""
    prices = pd.DataFrame(
        {"date": pd.to_datetime(["2026-11-06", "2026-11-09"]), "close": [99.5, 104.0]}
    )
    forecasts = pd.DataFrame(
        {"model": ["arima", "arima"],
         "forecast_date": pd.to_datetime(["2026-11-09", "2026-11-10"])}
    )  # fmt: skip
    line = accuracy.no_change_forecast(prices, forecasts)
    assert line["date"].iloc[0] == pd.Timestamp("2026-11-06")
    assert line["close"].tolist() == [99.5, 99.5, 99.5]


def test_the_rows_keep_windows_and_tickers_apart():
    """Each column reads its own window, and the ticker's rows, never ALL's."""
    published = _published()
    test = published["window"] == "test"
    published.loc[test, "mape"] += 1.0
    published.loc[test, "skill"] -= 0.1
    published.loc[published["ticker"] == "ALL", "mape"] += 10.0
    arima = accuracy.accuracy_rows(published, "AAPL").set_index("Model").loc["ARIMA"]
    assert arima["MAPE, validation"] == "2.98%"
    assert arima["MAPE, test"] == "3.98%"
    assert arima[accuracy.SKILL] == "-0.010"


def test_provenance_names_each_window_and_its_run():
    text = accuracy.provenance(_published())
    assert "on data to 2026-11-06" in text
    assert "validation: 42 start dates, 2025-06-30 to 2026-04-23 (run P4)" in text
    assert "(run T3)" in text


def test_an_empty_table_reads_as_none_so_the_page_says_so(tmp_path):
    """A table with its header and no rows passed the column check, and the
    caption then failed on its first row, stopping every tab after it."""
    path = tmp_path / "empty.csv"
    _published().iloc[0:0].to_csv(path, index=False)
    assert accuracy.load_published(path) is None


def test_a_ticker_the_evaluation_did_not_score_is_said_to_be_missing():
    published = _published()
    assert accuracy.covers(published, "AAPL")
    assert not accuracy.covers(published, "SPCX")


def test_the_committed_table_ships_with_the_package():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    data = config["tool"]["setuptools"]["package-data"]
    assert accuracy.PUBLISHED.name in data["quantflow.dashboard"]
