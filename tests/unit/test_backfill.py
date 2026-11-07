"""tests/unit/test_backfill.py

Covers quantflow.ingestion.backfill: the one-off fill of Yahoo's daily bars
for dates that have only 15-min bars.

What has to hold: a dry run writes nothing; a real run writes only dates that
have 15-min bars and no daily bar; a second run writes nothing; and every row
written carries the batch's inserted_at, so the printed undo query deletes
exactly those rows.
"""

import numpy as np
import pandas as pd
import pytest

from quantflow.ingestion import backfill, yfinance_fetcher

NOW = pd.Timestamp("2026-11-09 18:00", tz="UTC")  # a Monday, after the close
SEEDED = ["2026-04-08", "2026-04-09", "2026-04-10"]
INTRADAY_ONLY = ["2026-04-13", "2026-04-14", "2026-04-15"]
NO_BARS_AT_ALL = "2026-04-16"


@pytest.fixture
def stored(price_bars):
    """Seeded daily bars, then dates that have only a 15-min session. The
    16th has no bars at all (a missed refresh)."""
    groups = [price_bars.daily(day, 100.0) for day in SEEDED]
    groups += [price_bars.session(day, [101.0, 101.5]) for day in INTRADAY_ONLY]
    return price_bars.frame(*groups)


@pytest.fixture
def yahoo(monkeypatch):
    """Yahoo's daily bars for every weekday in the range, raw, naive-indexed."""
    calls = []
    days = pd.bdate_range("2026-04-13", "2026-04-16")
    close = 200.0 + np.arange(len(days), dtype=float)
    frame = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Adj Close": close * 0.99,
            "Volume": np.full(len(days), 5e6),
        },
        index=days,
    )

    def fake_download(ticker, **kwargs):
        calls.append(kwargs)
        return frame.copy()

    monkeypatch.setattr(yfinance_fetcher.yf, "download", fake_download)
    return calls


def test_a_dry_run_writes_nothing_and_reports_rows_per_ticker(
    fake_db, read_tables, stored, yahoo, capsys
):
    read_tables["raw_prices"] = stored
    planned = backfill.run(["AAPL"], dry_run=True, now=NOW)

    out = capsys.readouterr().out
    assert planned == {"AAPL": 3}
    assert fake_db.inserts("raw_prices") == []
    assert "AAPL" in out
    assert "3 rows to insert" in out
    assert "2026-04-13 to 2026-04-15" in out
    assert "1 trading date with no stored bars" in out


def test_a_run_writes_only_dates_that_have_15min_bars_and_no_daily_bar(
    fake_db, read_tables, stored, yahoo
):
    read_tables["raw_prices"] = stored
    written = backfill.run(["AAPL"], now=NOW)

    rows = fake_db.tables["raw_prices"]
    assert written == {"AAPL": 3}
    assert [r["ts"] for r in rows] == [
        pd.Timestamp(day, tz="UTC") for day in INTRADAY_ONLY
    ]
    assert [r["close"] for r in rows] == [200.0, 201.0, 202.0]  # raw, not Adj Close
    assert pd.Timestamp(NO_BARS_AT_ALL, tz="UTC") not in [r["ts"] for r in rows]
    assert yahoo[0]["auto_adjust"] is False


def test_every_row_carries_the_batch_timestamp_the_undo_query_uses(
    fake_db, read_tables, stored, yahoo, capsys
):
    read_tables["raw_prices"] = stored
    backfill.run(["AAPL"], now=NOW)

    stamps = {r["inserted_at"] for r in fake_db.tables["raw_prices"]}
    assert stamps == {NOW.to_pydatetime()}
    out = capsys.readouterr().out
    assert backfill.undo_sql(NOW.to_pydatetime()) in out


def test_running_again_writes_nothing(fake_db, read_tables, stored, yahoo, price_bars):
    read_tables["raw_prices"] = stored
    backfill.run(["AAPL"], now=NOW)

    # The table as the second run sees it: the first run's daily bars included.
    filled = [{k: r[k] for k in ("ts", "open", "high", "low", "close", "volume")}
              for r in fake_db.tables["raw_prices"]]  # fmt: skip
    read_tables["raw_prices"] = price_bars.frame(
        stored.drop(columns=["date", "ds", "y"]).to_dict("records"), filled
    )
    assert backfill.run(["AAPL"], now=NOW) == {"AAPL": 0}
    assert len(fake_db.inserts("raw_prices")) == 1


def test_the_undo_query_targets_exactly_one_batch():
    batch = pd.Timestamp("2026-11-09 18:00:00.123456", tz="UTC").to_pydatetime()
    sql = backfill.undo_sql(batch)

    literal = "TIMESTAMPTZ '2026-11-09 18:00:00.123456+00:00'"
    assert sql.count(f"inserted_at = {literal}") == 2  # the preview and the delete
    assert "DELETE FROM raw_prices" in sql
    assert "source = 'yfinance'" in sql
    assert "TIME '00:00'" in sql  # only midnight daily bars, which is all it writes


def test_a_failure_part_way_still_prints_the_undo_for_rows_already_written(
    fake_db, read_tables, stored, yahoo, monkeypatch, capsys
):
    """Each ticker's insert commits on its own, so a failure on the second
    ticker leaves the first one's rows in place. The undo query must still
    reach the operator."""
    from sqlalchemy.exc import DataError

    read_tables["raw_prices"] = stored
    downloads = []
    yahoo_download = yfinance_fetcher.yf.download  # the yahoo fixture's fake

    def failing_on_second(ticker, **kwargs):
        downloads.append(ticker)
        if len(downloads) == 2:
            fake_db.fail_next_insert = DataError("INSERT", {}, Exception("boom"))
        return yahoo_download(ticker, **kwargs)

    monkeypatch.setattr(yfinance_fetcher.yf, "download", failing_on_second)

    with pytest.raises(DataError):
        backfill.run(["AAPL", "MSFT"], now=NOW)

    out = capsys.readouterr().out
    assert len(fake_db.tables["raw_prices"]) == 3  # AAPL's rows were committed
    assert backfill.undo_sql(NOW.to_pydatetime()) in out


def test_a_date_with_a_stray_off_grid_row_still_gets_its_daily_bar(price_bars):
    """A stray 16:00 row used to make a date look like it had a daily bar, so
    the backfill skipped exactly the dates it exists to repair."""
    session = price_bars.session("2026-04-13", [101.0, 101.5])
    stray_ts = pd.Timestamp("2026-04-13 16:00", tz="America/New_York")
    stray = {**session[-1], "ts": stray_ts.tz_convert("UTC")}
    stored = price_bars.frame(session, stray)

    assert backfill.dates_needing_daily_bars(stored) == [pd.Timestamp("2026-04-13")]


def test_a_tz_aware_yahoo_index_still_matches_the_stored_dates(price_bars):
    """yfinance returns naive daily indexes today; an aware one must not turn
    every date into "Yahoo has no daily bar"."""
    stored = price_bars.frame(price_bars.session("2026-04-13", [101.0, 101.5]))
    index = pd.DatetimeIndex([pd.Timestamp("2026-04-13", tz="America/New_York")])
    yahoo = pd.DataFrame(
        {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [200.0],
         "Adj Close": [200.0], "Volume": [5e6]},
        index=index,
    )  # fmt: skip

    plan = backfill.plan_ticker("AAPL", stored, yahoo)
    assert [r["close"] for r in plan.rows] == [200.0]
    assert plan.no_yahoo_bar == []


def _unreachable(ticker, **kwargs):
    raise ConnectionError("yahoo unreachable")


def test_a_failed_download_is_reported_and_the_cli_exits_non_zero(
    fake_db, read_tables, stored, monkeypatch, capsys
):
    """download() turns a Yahoo failure into an empty frame. Read as "Yahoo
    has no bar for any of these dates", the backfill reported a clean run with
    nothing to do and exited 0."""
    read_tables["raw_prices"] = stored
    monkeypatch.setattr(yfinance_fetcher.yf, "download", _unreachable)

    assert backfill.main(["--ticker", "AAPL"]) == 1
    captured = capsys.readouterr()
    assert fake_db.inserts("raw_prices") == []
    assert "no daily bars from Yahoo" in captured.out
    assert "BACKFILL INCOMPLETE" in captured.err


def test_one_ticker_without_yahoo_data_does_not_stop_the_others(
    fake_db, read_tables, stored, yahoo, monkeypatch
):
    read_tables["raw_prices"] = stored
    yahoo_download = yfinance_fetcher.yf.download  # the yahoo fixture's fake

    def unreachable_for_aapl(ticker, **kwargs):
        if ticker == "AAPL":
            return _unreachable(ticker, **kwargs)
        return yahoo_download(ticker, **kwargs)

    monkeypatch.setattr(yfinance_fetcher.yf, "download", unreachable_for_aapl)
    with pytest.raises(backfill.BackfillError, match="AAPL"):
        backfill.run(["AAPL", "MSFT"], now=NOW)
    assert {r["ticker"] for r in fake_db.tables["raw_prices"]} == {"MSFT"}


def test_a_ticker_with_nothing_to_fill_does_not_download(
    fake_db, read_tables, price_bars, yahoo
):
    read_tables["raw_prices"] = price_bars.frame(price_bars.daily("2026-04-08", 1.0))
    assert backfill.run(["AAPL"], now=NOW) == {"AAPL": 0}
    assert yahoo == []
