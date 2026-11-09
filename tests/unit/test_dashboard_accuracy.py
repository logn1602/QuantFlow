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

from quantflow.config import TICKERS
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
    assert no_change["Skill vs no change"] == ""
    assert no_change["Band held the close"] == "no band"
    arima = rows.iloc[1]
    assert arima["Skill vs no change"] == "-0.010"
    assert arima["Band held the close"] == "50% / 50%"


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
    published = accuracy.load_published()
    assert published is not None
    assert set(published["window"]) == {"validation", "test"}
    assert published["as_of"].nunique() == 1
    for window, rows in published.groupby("window"):
        assert set(rows["ticker"]) == {*TICKERS, "ALL"}, window
        for ticker, group in rows.groupby("ticker"):
            assert set(accuracy.SHOWN) <= set(group["model"]), (window, ticker)
    shown = published[published["model"].isin(accuracy.SHOWN)]
    assert (shown["mape"] > 0).all() and shown["mape"].notna().all()
    # Band coverage is a percentage (evaluation.metrics.band_coverage), which the
    # captions print as it is; naive has no band.
    banded = shown[shown["model"] != accuracy.BASELINE]["band_coverage"]
    assert banded.notna().all() and banded.between(1, 100).all()
    assert shown[shown["model"] == accuracy.BASELINE]["band_coverage"].isna().all()


def test_the_committed_table_ships_with_the_package():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    data = config["tool"]["setuptools"]["package-data"]
    assert accuracy.PUBLISHED.name in data["quantflow.dashboard"]
