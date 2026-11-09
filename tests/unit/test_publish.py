"""tests/unit/test_publish.py

Covers evaluation/publish.py and the evaluate command's publish step: what
the README and the dashboard show is built only from a run folder's own
tables and header, from a validation run and a test-date run of the same
snapshot, both on committed code.
"""

import struct

import numpy as np
import pandas as pd
import pytest

from quantflow.evaluation import publish
from quantflow.pipelines import evaluate

DIGEST = "f0c3957b" + "0" * 56
MODELS = ["naive", "arima", "xgboost"]


def _header(window, origins="42, every 5 trading days, 2025-06-30 to 2026-04-23"):
    return {
        "window": window,
        "snapshot digest": DIGEST,
        "as of": "2026-11-06",
        "commit": "73a47b5f04ac548583fe29c4f810623cc995236b",
        "scored origins": origins,
    }


def _tables(xgboost_beats=False):
    per_ticker = pd.DataFrame(
        {"ticker": ["AAPL"] * 3 + ["MSFT"] * 3, "model": MODELS * 2,
         "n": [294, 294, 294, 294, 294, 294],
         "mape": [2.0, 2.1, 2.4, 3.0, 3.1, 3.2],
         "skill": [0.0, -0.05, -0.2, 0.0, -0.03, -0.06],
         "band_coverage": [np.nan, 90.0, 40.0, np.nan, 80.0, 50.0]}
    )  # fmt: skip
    headline = pd.DataFrame(
        {"model": MODELS, "mean_mape": [2.5, 2.6, 2.8],
         "mean_skill": [0.0, -0.04, -0.13],
         "beats_naive": [np.nan, False, xgboost_beats],
         "p_holm": [np.nan, 0.79, 0.0006]}
    )  # fmt: skip
    per_horizon = pd.DataFrame(
        [{"model": m, "h": h, "mape": base + 0.4 * h}
         for m, base in zip(MODELS, [1.0, 1.05, 1.3], strict=True)
         for h in range(1, 8)]
    )  # fmt: skip
    return headline, per_ticker, per_horizon


def _run(window, name, **kwargs):
    headline, per_ticker, per_horizon = _tables(**kwargs)
    return publish.RunTables(name, _header(window), headline, per_ticker, per_horizon)


def test_the_table_has_every_ticker_and_all_per_window_with_its_run():
    table = publish.accuracy_table([_run("validation", "P4"), _run("test", "T3")])
    assert len(table) == 2 * 3 * 3  # windows x (2 tickers + ALL) x models
    every = table[(table["ticker"] == "ALL") & (table["window"] == "validation")]
    arima = every[every["model"] == "arima"].iloc[0]
    assert arima["mape"] == 2.6  # the headline's mean over tickers
    assert arima["band_coverage"] == pytest.approx(85.0)  # weighted by cells
    assert arima["beats_naive"] == "false"
    assert arima["cells"] == 588
    naive = every[every["model"] == "naive"].iloc[0]
    assert np.isnan(naive["band_coverage"]) and naive["beats_naive"] == ""
    aapl = table[(table["ticker"] == "AAPL") & (table["model"] == "xgboost")]
    assert aapl["mape"].tolist() == [2.4, 2.4]
    assert set(table["run"]) == {"P4", "T3"} and set(table["commit"]) == {"73a47b5"}
    assert table[["origins", "first_origin", "last_origin"]].iloc[0].tolist() == [
        42, "2025-06-30", "2026-04-23"
    ]  # fmt: skip
    # Naive first, in the label order, so the table reads the same every time.
    assert every["model"].tolist() == ["naive", "arima", "xgboost"]


def test_an_unreadable_origins_line_stops_publishing():
    run = _run("validation", "P4")
    broken = publish.RunTables(
        run.name, {**run.header, "scored origins": "?"}, run.headline,
        run.per_ticker, run.per_horizon,
    )  # fmt: skip
    with pytest.raises(ValueError, match="scored origins"):
        publish.accuracy_table([broken])


def test_the_note_names_only_models_the_primary_test_says_beat_naive():
    runs = [_run("validation", "P4"), _run("test", "T3")]
    assert publish.beats_naive_note(runs) == (
        'No model beats "no change" on either set of dates.'
    )
    runs[1] = _run("test", "T3", xgboost_beats=True)
    assert publish.beats_naive_note(runs) == 'Beats "no change": XGBoost (test dates).'


def test_the_chart_is_a_png_drawn_from_both_runs(tmp_path):
    path = tmp_path / "images" / "chart.png"
    publish.horizon_chart([_run("validation", "P4"), _run("test", "T3")], path)
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (1408, 896)  # 8.8 x 5.6 inches at 160 dpi


def _run_folder(root, name, window, *, commit=None, digest=DIGEST):
    folder = root / name
    folder.mkdir()
    header = _header(window) | {"snapshot digest": digest}
    if commit:
        header["commit"] = commit
    lines = ["# Rolling-origin evaluation", ""]
    lines += [f"- {key}: {value}" for key, value in header.items()]
    lines += ["", "## Headline", ""]
    (folder / "report.md").write_text("\n".join(lines), encoding="utf-8")
    headline, per_ticker, per_horizon = _tables()
    headline.to_csv(folder / "headline.csv", index=False)
    per_ticker.to_csv(folder / "per_ticker.csv", index=False)
    per_horizon.to_csv(folder / "per_horizon.csv", index=False)
    return folder


def _publish(tmp_path, validation, test):
    return evaluate.main(
        ["publish", "--validation", str(validation), "--test", str(test),
         "--table", str(tmp_path / "out" / "published.csv"),
         "--chart", str(tmp_path / "out" / "chart.png")]
    )  # fmt: skip


def test_publish_writes_the_table_and_the_chart(tmp_path):
    validation = _run_folder(tmp_path, "P4", "validation")
    test = _run_folder(tmp_path, "T3", "test")
    assert _publish(tmp_path, validation, test) == 0
    table = pd.read_csv(tmp_path / "out" / "published.csv")
    assert set(table["window"]) == {"validation", "test"}
    assert b"\r\n" not in (tmp_path / "out" / "published.csv").read_bytes()
    assert (tmp_path / "out" / "chart.png").stat().st_size > 0


@pytest.mark.parametrize(
    "case", ["windows swapped", "uncommitted run", "different snapshots"]
)
def test_publish_refuses_runs_it_cannot_vouch_for(tmp_path, case):
    if case == "windows swapped":
        validation = _run_folder(tmp_path, "T3", "test")
        test = _run_folder(tmp_path, "P4", "validation")
    elif case == "uncommitted run":
        validation = _run_folder(
            tmp_path, "P4", "validation", commit="73a47b5 (uncommitted changes)"
        )
        test = _run_folder(tmp_path, "T3", "test")
    else:
        validation = _run_folder(tmp_path, "P4", "validation")
        test = _run_folder(tmp_path, "T3", "test", digest="0" * 64)
    assert _publish(tmp_path, validation, test) == 2
    assert not (tmp_path / "out").exists()
