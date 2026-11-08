"""tests/integration/test_writes_on_postgres.py

The write-path claims the unit tests can only fake, checked on real Postgres:

  - a multi-row INSERT ... ON CONFLICT DO NOTHING reports only the rows it
    wrote, across chunks, which is what every writer's count now relies on,
    and writes a key repeated inside one INSERT once;
  - the backfill's undo query deletes exactly the rows one batch wrote,
    including a trading day it filled from nothing;
  - replacing a ticker's indicators rolls back the delete if an insert fails,
    and two runs replacing them at once both succeed;
  - the daily-bar rule works on the types the real driver returns;
  - the model features line up with prices whatever the session time zone;
  - the news sums the live run reads equal the ones the evaluation adds up
    from the article rows, at every edge of the rules, and so do the features;
  - the ensemble reads only each base model's latest stored run;
  - save_anomalies counts only the rows Postgres wrote;
  - the read-only diagnostics in scripts/diagnostics/ run end to end.

Needs a disposable local Postgres, never the live database:

    uv run pytest -m integration --pg-url postgresql://user@127.0.0.1:5432/scratch

Each test gets its own schema, dropped after, holding the tables from
sql/schema.sql and sql/schema_sentiment.sql: the ones these tests write. The
other schema files are not needed, and schema_roles.sql creates cluster-wide
roles that a per-test schema cannot contain.
"""

import datetime as dt
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DataError

from quantflow.db import prices
from quantflow.db.batch import insert_rows
from quantflow.db.forecasts import load_base_forecasts
from quantflow.db.indicators import replace_indicators
from quantflow.features import anomalies
from quantflow.features.engineering import load_features
from quantflow.features.sentiment import save_sentiment
from quantflow.ingestion import backfill, yfinance_fetcher

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]


def _schema_engine(url, schema, **settings):
    """An engine whose sessions use the test schema. The lock timeout makes a
    stuck test fail rather than hang the suite."""
    options = {"search_path": schema, "lock_timeout": "15s"} | settings
    return create_engine(
        url,
        connect_args={"options": " ".join(f"-c{k}={v}" for k, v in options.items())},
    )


@pytest.fixture
def pg(request, install_engine, is_local_pg_url):
    url = request.config.getoption("--pg-url")
    if not url:
        pytest.skip("needs --pg-url pointing at a disposable local Postgres")
    if not is_local_pg_url(url):
        pytest.fail("--pg-url must be a local database, never the live one")

    schema = f"qf_test_{uuid.uuid4().hex[:8]}"
    admin = create_engine(url)
    with admin.begin() as conn:
        conn.exec_driver_sql(f"CREATE SCHEMA {schema}")
    engine = _schema_engine(url, schema)
    with engine.begin() as conn:
        for name in ("schema.sql", "schema_sentiment.sql"):
            conn.exec_driver_sql((ROOT / "sql" / name).read_text(encoding="utf-8"))
    install_engine(engine)

    yield engine

    engine.dispose()
    with admin.begin() as conn:
        conn.exec_driver_sql(f"DROP SCHEMA {schema} CASCADE")
    admin.dispose()


def _bars(days, ticker="AAPL"):
    return [
        {
            "ticker": ticker,
            "source": "yfinance",
            "ts": pd.Timestamp(day, tz="UTC"),
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1,
        }
        for day in days
    ]


def _count(engine, sql, **params):
    with engine.connect() as conn:
        return conn.execute(text(sql), params).scalar_one()


def test_on_conflict_skips_are_not_counted_across_chunks(pg):
    days = pd.bdate_range("2026-01-05", periods=10)
    assert prices.insert_bars(_bars(days[:4])) == 4

    with pg.begin() as conn:
        written = insert_rows(
            conn,
            "raw_prices",
            _bars(days),
            skip_conflicts=("ticker", "source", "ts"),
            chunk_rows=3,
        )
    assert written == 6
    assert _count(pg, "SELECT count(*) FROM raw_prices") == 10


def test_a_key_repeated_inside_one_insert_is_written_once(pg):
    row = _bars(["2026-01-05"])
    with pg.begin() as conn:
        written = insert_rows(
            conn, "raw_prices", row + row, skip_conflicts=("ticker", "source", "ts")
        )
    assert written == 1


def test_a_headline_two_feeds_return_at_once_is_written_and_counted_once(pg):
    """RSS and NewsAPI can return the same headline with the same timestamp,
    and save_sentiment sends both in one INSERT, relying on the rule above."""
    article = {
        "headline": "Apple beats estimates",
        "url": "https://example.com/a",
        "published_at": pd.Timestamp("2026-11-06 14:00", tz="UTC"),
        "source": "rss",
        "sentiment": "positive",
        "score_pos": 0.7,
        "score_neg": 0.2,
        "score_neu": 0.1,
        "compound": 0.5,
    }
    assert save_sentiment([article, article | {"source": "newsapi"}], "AAPL") == 1
    assert _count(pg, "SELECT count(*) FROM news_sentiment") == 1


def test_the_backfill_undo_deletes_exactly_the_rows_it_wrote(pg, monkeypatch, capsys):
    # A seeded daily bar, two dates with only 15-min bars, a daily bar some
    # other run wrote with DEFAULT NOW(), and the 16th, a trading day with no
    # bars at all.
    session = [
        pd.Timestamp(f"{day} 09:30", tz="America/New_York").tz_convert("UTC")
        for day in ("2026-04-13", "2026-04-14")
    ]
    rows = _bars(["2026-04-10"]) + [
        {**_bars(["2026-04-10"])[0], "ts": ts} for ts in session
    ]
    prices.insert_bars(rows + _bars(["2026-04-15"]))
    before = _count(pg, "SELECT count(*) FROM raw_prices")

    days = pd.bdate_range("2026-04-13", "2026-04-16")
    yahoo = pd.DataFrame(
        {c: np.full(4, 200.0) for c in ("Open", "High", "Low", "Close", "Adj Close")}
        | {"Volume": np.full(4, 5e6)},
        index=days,
    )
    monkeypatch.setattr(yfinance_fetcher.yf, "download", lambda t, **k: yahoo.copy())

    # 18:00 New York on the 16th, so its session is the last completed one.
    now = pd.Timestamp("2026-04-16 22:00:00.123456", tz="UTC")
    assert backfill.run(["AAPL"], now=now) == {"AAPL": 3}  # the 13th, 14th and 16th
    assert _count(pg, "SELECT count(*) FROM raw_prices") == before + 3

    undo = backfill.undo_sql(now.to_pydatetime())
    delete = undo[undo.index("DELETE FROM") :]
    with pg.begin() as conn:
        deleted = conn.execute(text(delete)).rowcount
    assert deleted == 3  # the empty day's row too, and not the other run's bar
    assert _count(pg, "SELECT count(*) FROM raw_prices") == before


def test_a_failed_indicator_replace_keeps_the_previous_rows(pg):
    ts = pd.Timestamp("2026-04-10", tz="UTC")
    columns = ("rsi_14", "macd", "macd_signal", "macd_hist", "bb_upper", "bb_middle")
    good = {"ticker": "AAPL", "ts": ts} | dict.fromkeys(columns, 1.0)
    good["bb_lower"] = 1.0
    assert replace_indicators("AAPL", [good]) == 1

    overflow = good | {"ts": ts + pd.Timedelta(days=1), "rsi_14": 1e9}  # NUMERIC(8,4)
    with pytest.raises(DataError):
        replace_indicators("AAPL", [good | {"rsi_14": 2.0}, overflow])

    assert _count(pg, "SELECT count(*) FROM technical_indicators") == 1
    assert _count(pg, "SELECT rsi_14 FROM technical_indicators") == 1.0


def test_two_runs_replacing_one_tickers_indicators_at_once_both_succeed(pg):
    """The second run's DELETE waits on the first and cannot see the rows the
    first commits; its INSERT must skip them rather than fail the refresh."""
    import threading
    import time

    ts = pd.Timestamp("2026-04-10", tz="UTC")
    columns = ("rsi_14", "macd", "macd_signal", "macd_hist", "bb_upper", "bb_middle")
    row = {"ticker": "AAPL", "ts": ts} | dict.fromkeys(columns, 1.0)
    row["bb_lower"] = 1.0
    assert replace_indicators("AAPL", [row]) == 1

    outcome = {}

    def second_run():
        try:
            outcome["written"] = replace_indicators("AAPL", [row | {"rsi_14": 3.0}])
        except Exception as e:
            outcome["error"] = e

    worker = threading.Thread(target=second_run)
    # Blocked by this backend in particular, not any lock wait on the server.
    blocked = (
        "SELECT count(*) FROM pg_stat_activity "
        "WHERE CAST(:pid AS integer) = ANY(pg_blocking_pids(pid))"
    )
    first = pg.connect()
    try:
        first_txn = first.begin()
        first_pid = first.exec_driver_sql("SELECT pg_backend_pid()").scalar_one()
        first.execute(
            text("DELETE FROM technical_indicators WHERE ticker = ANY(:t)"),
            {"t": ["AAPL"]},
        )
        insert_rows(first, "technical_indicators", [row | {"rsi_14": 2.0}])

        worker.start()
        deadline = time.monotonic() + 10
        while not _count(pg, blocked, pid=first_pid):  # the second DELETE waits
            if time.monotonic() > deadline:
                pytest.fail("the second run never blocked on the first")
            time.sleep(0.05)
        first_txn.commit()
    finally:
        first.close()  # rolls back if the commit was not reached
    worker.join(timeout=10)

    assert not worker.is_alive(), "the second run is still blocked"
    assert "error" not in outcome, outcome.get("error")
    assert outcome["written"] == 0
    assert _count(pg, "SELECT rsi_14 FROM technical_indicators") == 2.0


def test_daily_bars_on_the_types_the_driver_returns(pg):
    session = [
        {
            **_bars(["2026-04-13"])[0],
            "ts": pd.Timestamp(f"2026-04-13 {t}", tz="America/New_York"),
            "close": close,
        }
        for t, close in (("09:30", 100.0), ("15:45", 102.5))
    ]
    prices.insert_bars(_bars(["2026-04-10"]) + session)

    daily = prices.load_daily_close("AAPL")
    assert list(daily["ds"]) == [pd.Timestamp("2026-04-10"), pd.Timestamp("2026-04-13")]
    assert list(daily["y"]) == [1.0, 102.5]


def test_features_line_up_with_prices_in_a_new_york_session(pg, install_engine):
    """The features are built in process (decision D4): article times come
    back from the driver in the session's TimeZone and must still be cut in
    UTC (decision D7), and the indicator and anomaly tables, written for the
    dashboard, must not reach the features."""
    days = ["2026-04-09", "2026-04-10", "2026-04-13"]
    prices.insert_bars(_bars(days))
    columns = ("macd", "macd_signal", "macd_hist", "bb_upper", "bb_middle", "bb_lower")
    replace_indicators(
        "AAPL",
        [
            {"ticker": "AAPL", "ts": pd.Timestamp(day, tz="UTC"), "rsi_14": 50.0}
            | dict.fromkeys(columns, 1.0)
            for day in days
        ],
    )
    with pg.begin() as conn:
        insert_rows(
            conn,
            "anomalies",
            [{"ticker": "AAPL", "ts": pd.Timestamp(days[1], tz="UTC"),
              "close": 1.0, "zscore": 3.5, "flag": "HIGH"}],
        )  # fmt: skip
        insert_rows(
            conn,
            "news_sentiment",
            [{"ticker": "AAPL", "source": "rss", "headline": "morning news",
              "published_at": pd.Timestamp("2026-04-10 13:50", tz="UTC"),
              # 10:00 New York on the 10th: after the 9th's cutoff (12:00
              # UTC), so it counts on the 10th. Read as a naive New York
              # time it would land on the 9th.
              "inserted_at": pd.Timestamp("2026-04-10 14:00", tz="UTC"),
              "sentiment": "positive", "compound": 0.5}],
        )  # fmt: skip

    with pg.connect() as conn:
        schema = conn.exec_driver_sql("SELECT current_schema()").scalar_one()
    new_york = _schema_engine(pg.url, schema, timezone="America/New_York")
    install_engine(new_york)
    try:
        features = load_features("AAPL")
    finally:
        new_york.dispose()

    assert list(features["date"]) == [pd.Timestamp(day) for day in days]
    assert features["rsi_14"].isna().all()  # three bars: too short, not the table
    assert features["zscore"].isna().all()  # computed, not the stored 3.5
    assert features["article_count"].tolist()[1] == 1


def test_the_ensemble_reads_only_each_models_latest_run(pg):
    """forecasts keeps every run (run_at is part of its key), so after a run
    with --no-clear the ensemble read two predictions per date."""
    days = pd.bdate_range("2026-11-09", periods=3)
    early = dt.datetime(2026, 11, 6, 21, 0, tzinfo=dt.UTC)
    late = early + dt.timedelta(hours=1)

    def stored_run(model, run_at, close):
        return [
            {"ticker": "AAPL", "model": model, "forecast_date": day.date(),
             "predicted_close": close, "lower_bound": close - 1,
             "upper_bound": close + 1, "run_at": run_at}
            for day in days
        ]  # fmt: skip

    with pg.begin() as conn:
        insert_rows(
            conn,
            "forecasts",
            stored_run("arima", early, 100.0)
            + stored_run("arima", late, 101.0)
            + stored_run("prophet", early, 102.0),
        )

    df = load_base_forecasts("AAPL", ["arima", "prophet"])
    by_model = df.groupby("model")["predicted_close"].apply(
        lambda v: sorted(set(v.astype(float)))
    )
    assert by_model.to_dict() == {"arima": [101.0], "prophet": [102.0]}
    assert len(df) == 6


def test_saving_anomalies_counts_only_rows_written(pg):
    """On the schema's unique (ticker, ts), ON CONFLICT DO NOTHING skips a
    repeated row; save_anomalies used to count it anyway."""
    ts = pd.Timestamp("2026-04-10", tz="UTC")
    flags = pd.DataFrame(
        {"ticker": "AAPL", "close": 100.0, "zscore": 3.2, "flag": "HIGH"},
        index=pd.DatetimeIndex([ts, ts], name="ts"),
    )
    assert anomalies.save_anomalies(flags) == 1
    assert _count(pg, "SELECT count(*) FROM anomalies") == 1


def test_the_diagnostics_run_read_only_on_postgres(pg, monkeypatch, tmp_path):
    """scripts/diagnostics/run_1a_diagnostics.py, end to end: its read-only
    transaction is the first thing on the connection, all four queries run,
    and the results file has one table per query."""
    import importlib.util
    import sys

    script = ROOT / "scripts" / "diagnostics" / "run_1a_diagnostics.py"
    spec = importlib.util.spec_from_file_location("run_1a_diagnostics", script)
    assert spec and spec.loader
    diagnostics = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, diagnostics)
    spec.loader.exec_module(diagnostics)
    # The pg fixture's install_engine routes db.readonly's get_engine here.

    prices.insert_bars(_bars(["2026-04-09", "2026-04-10"]))
    out = tmp_path / "results.md"
    assert diagnostics.main(["--live", "--out", str(out)]) == 0

    text = out.read_text(encoding="utf-8")
    assert text.count("### ") == 4
    # Query 1 on this test's two daily bars: two dates, both with a daily bar.
    assert "| AAPL | 2026-04-09 | 2026-04-10 | 2 | 2 | 0 |" in text
    assert _count(pg, "SELECT count(*) FROM raw_prices") == 2  # nothing written


def test_the_evaluation_snapshot_is_exported_read_only(pg, tmp_path, capsys):
    """pipelines.evaluate export, end to end: daily bars built by the same
    rule as the loaders, a date built from 15-min bars listed, articles
    stored after the as-of date's news cutoff left out, the URL cut to its
    host, and nothing written."""
    from quantflow.evaluation.snapshot import read_snapshot
    from quantflow.pipelines import evaluate

    prices.insert_bars(_bars(["2026-04-07", "2026-04-08", "2026-04-09"]))
    session = pd.date_range(
        "2026-04-10 13:30", "2026-04-10 19:45", freq="15min", tz="UTC"
    )
    prices.insert_bars(
        [{"ticker": "AAPL", "source": "yfinance", "ts": ts, "open": 2.0, "high": 2.0,
          "low": 2.0, "close": 2.0, "volume": 1} for ts in session]
    )  # fmt: skip
    with pg.begin() as conn:
        for stored, url in (
            ("2026-04-10 22:40+00", "https://finance.yahoo.com/news/x?a=1"),
            ("2026-04-11 12:30+00", "https://example.com/late"),  # after the cutoff
        ):
            conn.execute(
                text("""
                    INSERT INTO news_sentiment (ticker, source, published_at,
                        headline, url, sentiment, score_pos, score_neg, score_neu,
                        compound, inserted_at)
                    VALUES ('AAPL', 'rss', '2026-04-10 20:00+00', :h, :url,
                        'positive', 0.8, 0.1, 0.1, 0.7, :stored)
                """),
                {"h": f"headline {stored}", "url": url, "stored": stored},
            )

    code = evaluate.main(
        ["export", "--as-of", "2026-04-10", "--tickers", "AAPL", "--out",
         str(tmp_path), "--live"]
    )  # fmt: skip
    assert code == 0, capsys.readouterr().err
    snapshot = read_snapshot(tmp_path / "2026-04-10")
    bars = snapshot.prices["AAPL"]
    assert len(bars) == 4 and bars["close"].iloc[-1] == 2.0  # last 15-min close
    assert snapshot.built_dates == {"AAPL": [pd.Timestamp("2026-04-10")]}
    assert snapshot.articles["url_domain"].tolist() == ["finance.yahoo.com"]
    assert _count(pg, "SELECT count(*) FROM news_sentiment") == 2  # nothing written


def _story(n, ticker, published, inserted=None, **columns):
    """A news_sentiment row, stored when published unless inserted is given."""
    row = {
        "ticker": ticker,
        "source": "rss",
        "headline": f"story {n}",
        "url": "https://finance.yahoo.com/news/x",
        "published_at": pd.Timestamp(published, tz="UTC"),
        "inserted_at": pd.Timestamp(inserted or published, tz="UTC"),
        "sentiment": "positive",
        "score_pos": 0.8,
        "score_neg": 0.1,
        "score_neu": 0.1,
        "compound": 0.7,
    }
    return row | columns


def test_the_news_sums_the_live_run_reads_are_the_evaluations(pg, install_engine):
    """The live run reads the news summed per day in SQL (load_news); the
    evaluation sums the article rows in Python (news_days). Same sums, at
    every edge the rules have, in a New York session, and so the same
    features, bit for bit."""
    from quantflow.db.features import ARTICLE_SELECT
    from quantflow.features import news
    from quantflow.features.engineering import build_features, load_news

    days = pd.bdate_range("2026-04-06", "2026-04-17")
    prices.insert_bars(_bars(days) + _bars(days, "MSFT"))
    cutoff = pd.Timestamp("2026-04-09 12:00")  # the 8th's
    us = pd.Timedelta("1us")
    stories = [
        _story(1, "AAPL", cutoff),  # on the cutoff: the 8th
        _story(2, "AAPL", cutoff + us, compound=-0.3333),  # just after: the 9th
        # NewsAPI: available a day after publishing, here exactly at a cutoff.
        _story(3, "AAPL", cutoff - pd.Timedelta(hours=24), source="newsapi"),
        # Stored two days late: counts from when it arrived.
        _story(4, "MSFT", "2026-04-08 15:00", "2026-04-10 23:00", source="newsapi",
               sentiment="negative", compound=-0.9012),
        _story(5, "MSFT", "2026-04-11 15:00"),  # Saturday: Monday's
        # A FinBERT placeholder, and one a hair off it.
        _story(6, "AAPL", "2026-04-13 15:00", sentiment="neutral", compound=0.0,
               score_pos=0.33, score_neg=0.33, score_neu=0.34),
        _story(7, "AAPL", "2026-04-13 16:00", sentiment="neutral", compound=0.0,
               score_pos=0.3301, score_neg=0.3299, score_neu=0.34),
        _story(8, "AAPL", "2026-04-13 17:00", compound=None),
        # The general feed under AAPL, at any host case or port; not MSFT's,
        # nor a look-alike host, nor a missing link.
        _story(9, "AAPL", "2026-04-14 15:00", url="https://feeds.marketwatch.com/a"),
        _story(10, "AAPL", "2026-04-14 15:00", url="https://MarketWatch.com:443/b"),
        _story(11, "AAPL", "2026-04-14 15:00", url="https://notmarketwatch.com/c"),
        _story(12, "MSFT", "2026-04-14 15:00", url="https://feeds.marketwatch.com/a"),
        _story(13, "AAPL", "2026-04-14 15:00", url=None, compound=0.1234),
        _story(14, "AAPL", "2026-04-14 15:00", url="ftp://feeds.marketwatch.com/d"),
    ]  # fmt: skip
    with pg.begin() as conn:
        insert_rows(conn, "news_sentiment", stories)
        rows = pd.read_sql(text(ARTICLE_SELECT), conn)
    assert len(rows) == len(stories)

    with pg.connect() as conn:
        schema = conn.exec_driver_sql("SELECT current_schema()").scalar_one()
    new_york = _schema_engine(pg.url, schema, timezone="America/New_York")
    install_engine(new_york)
    try:
        read = load_news()
        features = {t: load_features(t) for t in ("AAPL", "MSFT")}
    finally:
        new_york.dispose()

    summed = news.news_days(rows)
    pd.testing.assert_frame_equal(read, summed)
    assert read.loc[~read["counts"], "articles"].sum() == 4  # 6, 9, 10 and 14
    for ticker, live in features.items():
        built = build_features(
            prices.load_daily_bars(ticker),
            rows[rows["ticker"] == ticker],
            ticker,
            fetched_at=rows["inserted_at"],
        )
        pd.testing.assert_frame_equal(live, built)
