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
        "tickers": "AAPL, MSFT",
        "models": ", ".join(MODELS),
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


def test_the_table_order_does_not_depend_on_the_input_order():
    """The committed table must come out the same however the runs and their
    rows arrive, so a republish changes only what changed."""
    runs = [_run("validation", "P4"), _run("test", "T3")]
    flipped = [
        publish.RunTables(r.name, r.header, r.headline.iloc[::-1],
                          r.per_ticker.iloc[::-1], r.per_horizon)
        for r in runs[::-1]
    ]  # fmt: skip
    pd.testing.assert_frame_equal(
        publish.accuracy_table(runs), publish.accuracy_table(flipped)
    )


def test_all_tickers_band_coverage_is_weighted_by_cells():
    run = _run("validation", "P4")
    per_ticker = run.per_ticker.copy()
    per_ticker.loc[per_ticker["ticker"] == "MSFT", "n"] = 98
    run = publish.RunTables(
        run.name, run.header, run.headline, per_ticker, run.per_horizon
    )
    table = publish.accuracy_table([run])
    arima = table.query("ticker == 'ALL' and model == 'arima'").iloc[0]
    assert arima["band_coverage"] == pytest.approx((90 * 294 + 80 * 98) / 392)
    assert arima["cells"] == 392


def test_the_table_keeps_full_precision_for_the_page_to_round_once():
    """Rounded here and again on the page, a MAPE of 2.39497 showed as 2.40%
    where the docs, rounding once, say 2.39."""
    run = _run("validation", "P4")
    run.per_ticker.loc[1, "mape"] = 2.39497
    table = publish.accuracy_table([run])
    cell = table[(table["ticker"] == "AAPL") & (table["model"] == "arima")]
    assert cell["mape"].item() == 2.39497


@pytest.mark.parametrize(
    ("stored", "flag"), [(False, "false"), ("False", "false"), ("True", "true")]
)
def test_beats_naive_is_read_as_written_not_as_a_truthy_string(stored, flag):
    run = _run("validation", "P4")
    run.headline["beats_naive"] = run.headline["beats_naive"].astype(object)
    run.headline.loc[1, "beats_naive"] = stored
    table = publish.accuracy_table([run])
    every = table[(table["ticker"] == "ALL") & (table["model"] == "arima")]
    assert every["beats_naive"].item() == flag


def test_an_unreadable_beats_naive_value_stops_publishing():
    run = _run("validation", "P4")
    run.headline["beats_naive"] = run.headline["beats_naive"].astype(object)
    run.headline.loc[1, "beats_naive"] = "maybe"
    with pytest.raises(ValueError, match="beats_naive"):
        publish.accuracy_table([run])


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


def _run_folder(root, name, window, *, commit=None, digest=DIGEST, **overrides):
    folder = root / name
    folder.mkdir()
    header = _header(window) | {"snapshot digest": digest} | overrides
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
    "case",
    ["windows swapped", "uncommitted run", "different snapshots",
     "different tickers", "different models"],
)  # fmt: skip
def test_publish_refuses_runs_it_cannot_vouch_for(tmp_path, case):
    """Runs on other tickers or models would put two incomparable "all
    tickers" figures side by side, or drop a model's chart panel."""
    test_overrides = {
        "different snapshots": {"digest": "0" * 64},
        "different tickers": {"tickers": "AAPL, MSFT, NVDA"},
        "different models": {"models": "naive, arima"},
    }.get(case, {})
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
        test = _run_folder(tmp_path, "T3", "test", **test_overrides)
    assert _publish(tmp_path, validation, test) == 2
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "dropped", ["- snapshot digest:", "- tickers:", "- models:", "- commit:"]
)
def test_publish_refuses_a_run_header_missing(tmp_path, dropped):
    """A line missing from both headers would compare equal (None), so a
    missing line is a refusal, not a match."""
    folders = [
        _run_folder(tmp_path, "P4", "validation"),
        _run_folder(tmp_path, "T3", "test"),
    ]
    for folder in folders:
        report = folder / "report.md"
        lines = report.read_text(encoding="utf-8").splitlines()
        kept = [line for line in lines if not line.startswith(dropped)]
        report.write_text("\n".join(kept), encoding="utf-8")
    assert _publish(tmp_path, *folders) == 2
    assert not (tmp_path / "out").exists()


def test_a_chart_that_fails_leaves_the_table_as_it_was(tmp_path, monkeypatch):
    """The chart is drawn before the table is written, so a failure cannot
    leave the dashboard and the README describing different runs."""
    validation = _run_folder(tmp_path, "P4", "validation")
    test = _run_folder(tmp_path, "T3", "test")

    def fail(runs, path):
        raise ValueError("no room for another panel")

    monkeypatch.setattr(publish, "horizon_chart", fail)
    assert _publish(tmp_path, validation, test) == 1
    assert not (tmp_path / "out" / "published.csv").exists()


def test_the_chart_states_the_runs_own_ticker_count(tmp_path, monkeypatch):
    texts = []
    import matplotlib.figure

    original = matplotlib.figure.Figure.text

    def record(self, x, y, s, *args, **kwargs):
        texts.append(s)
        return original(self, x, y, s, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "text", record)
    publish.horizon_chart(
        [_run("validation", "P4"), _run("test", "T3")], tmp_path / "chart.png"
    )
    assert any(text.startswith("MAPE over 2 tickers") for text in texts), texts
