"""tests/unit/test_snapshot.py

Covers the snapshot files: an exact round trip, refusal of an altered file
or an existing directory, and the check for dates built from 15-min bars.
"""

import json

import numpy as np
import pandas as pd
import pytest

from quantflow.evaluation import snapshot as snap

DATES = pd.bdate_range("2026-03-02", periods=5)


def _bars(start):
    close = start + np.arange(len(DATES)) * 1.234567891
    return pd.DataFrame(
        {"date": DATES, "open": close, "high": close + 1, "low": close - 1,
         "close": close, "volume": [1e6, np.nan, 2e6, 3e6, 4e6]}
    )  # fmt: skip


PRICES = {"AAPL": _bars(100.0), "MSFT": _bars(400.0)}
ARTICLES = pd.DataFrame(
    {
        "ticker": ["AAPL", "MSFT"],
        "source": ["rss", "newsapi"],
        "url_domain": ["finance.yahoo.com", np.nan],
        "published_at": pd.to_datetime(
            ["2026-03-02 14:00", "2026-03-03 09:30"], utc=True
        ),
        "inserted_at": pd.to_datetime(
            ["2026-03-02 22:41:07.123456", "2026-03-04 22:40:00.000001"], utc=True
        ),
        "sentiment": ["positive", "neutral"],
        "score_pos": [0.81, 0.33],
        "score_neg": [0.09, 0.33],
        "score_neu": [0.10, 0.34],
        "compound": [0.72, 0.0],
    }
)
STORED = pd.DataFrame(
    {"ticker": ["AAPL"], "date": [DATES[2].date()], "zscore": [2.1234]}
)


def _write(directory, **overrides):
    kwargs = {
        "as_of": "2026-03-06",
        "exported_at": pd.Timestamp("2026-03-07 10:00", tz="UTC"),
        "prices": PRICES,
        "articles": ARTICLES,
        "stored_indicators": STORED.drop(columns="zscore"),
        "stored_anomalies": STORED,
        "built_dates": {"AAPL": [], "MSFT": [DATES[1]]},
    }
    return snap.write_snapshot(directory, **(kwargs | overrides))


def test_a_snapshot_reads_back_exactly(tmp_path):
    directory = _write(tmp_path / "2026-03-06")
    read = snap.read_snapshot(directory)
    assert read.as_of == pd.Timestamp("2026-03-06") and read.tickers == ["AAPL", "MSFT"]
    for ticker, bars in PRICES.items():
        pd.testing.assert_frame_equal(read.prices[ticker], bars, check_dtype=False)
    pd.testing.assert_frame_equal(read.articles, ARTICLES, check_dtype=False)
    assert read.stored_anomalies["date"].iloc[0] == DATES[2]
    assert read.built_dates == {"AAPL": [], "MSFT": [DATES[1]]}
    assert list(read.trading_dates) == list(DATES)
    assert len(read.digest) == 64


def test_the_manifest_records_each_files_hash_and_rows(tmp_path):
    manifest = json.loads((_write(tmp_path / "s") / snap.MANIFEST).read_text("utf-8"))
    assert manifest["files"]["prices"]["rows"] == 10
    assert manifest["files"]["articles"]["rows"] == 2
    assert all(len(f["sha256"]) == 64 for f in manifest["files"].values())


def test_an_altered_file_is_refused(tmp_path):
    directory = _write(tmp_path / "s")
    prices = directory / snap.FILES["prices"]
    prices.write_text(prices.read_text("utf-8").replace("100.0", "100.5", 1), "utf-8")
    with pytest.raises(snap.SnapshotError, match="no longer matches"):
        snap.read_snapshot(directory)


def test_a_snapshot_is_never_overwritten(tmp_path):
    directory = _write(tmp_path / "s")
    with pytest.raises(snap.SnapshotError, match="never overwritten"):
        _write(directory)


def test_a_missing_manifest_is_refused(tmp_path):
    with pytest.raises(snap.SnapshotError, match="manifest"):
        snap.read_snapshot(tmp_path)


def test_a_built_date_at_an_origin_or_target_is_refused(tmp_path):
    read = snap.read_snapshot(_write(tmp_path / "s"))
    snap.check_built_dates(read, [DATES[0], DATES[2]])  # not used: fine
    with pytest.raises(snap.SnapshotError, match="2026-03-03"):
        snap.check_built_dates(read, DATES[1:3])


def test_tickers_with_different_dates_cannot_share_origins(tmp_path):
    prices = {"AAPL": _bars(100.0), "MSFT": _bars(400.0).iloc[:4]}
    read = snap.read_snapshot(_write(tmp_path / "s", prices=prices))
    with pytest.raises(snap.SnapshotError, match="different dates"):
        _ = read.trading_dates
