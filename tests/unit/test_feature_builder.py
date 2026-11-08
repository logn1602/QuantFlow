"""tests/unit/test_feature_builder.py

Covers build_features, the feature frame computed from daily bars and
articles, which both the live run (load_features) and the evaluation call.

Two contracts: load_features builds the frame in process and never reads the
indicator or anomaly tables (decision D4); and what it builds equals the
rows the real indicator and anomaly stages store for the dashboard, so the
features and the dashboard describe the same numbers.
"""

import numpy as np
import pandas as pd
import pytest

from quantflow.db.prices import daily_bars
from quantflow.features import anomalies, engineering, indicators

N_DAYS = 90
DAYS = pd.bdate_range("2026-01-02", periods=N_DAYS)


def _closes(seed=11):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, N_DAYS)))
    close[70] *= 1.12  # one jump, so both anomaly detectors have something to flag
    return close


@pytest.fixture
def stored_bars(price_bars):
    return price_bars.frame(
        [
            price_bars.daily(day.date(), float(close), volume=1e6 + 1e4 * i)
            for i, (day, close) in enumerate(zip(DAYS, _closes(), strict=True))
        ]
    )


def _article(ts, sentiment, compound, **overrides):
    """A stored article row, as load_articles returns it, stored when
    published."""
    scores = {"positive": (0.8, 0.1, 0.1), "negative": (0.1, 0.8, 0.1)}
    pos, neg, neu = scores.get(sentiment, (0.1, 0.1, 0.8))
    row = {
        "ticker": "AAPL",
        "source": "rss",
        "url_domain": "finance.yahoo.com",
        "published_at": pd.Timestamp(ts, tz="UTC"),
        "inserted_at": pd.Timestamp(ts, tz="UTC"),
        "sentiment": sentiment,
        "score_pos": pos,
        "score_neg": neg,
        "score_neu": neu,
        "compound": compound,
    }
    return row | overrides


ARTICLES = pd.DataFrame(
    [
        _article("2026-03-02 14:00", "positive", 0.5),
        _article("2026-03-02 15:30", "positive", 0.3),
        _article("2026-03-02 23:30", "negative", -0.4),
        # After midnight UTC, before the 2nd's cutoff at noon on the 3rd.
        _article("2026-03-03 00:30", "neutral", 0.0),
    ]
)

# The daily averages for ARTICLES, worked out by hand: Monday 2 March's news
# window runs to 12:00 UTC on Tuesday, so all four count on the 2nd.
STORED_SENTIMENT = pd.DataFrame(
    {
        "date": [pd.Timestamp("2026-03-02").date()],
        "sentiment_compound": [(0.5 + 0.3 - 0.4 + 0.0) / 4],
        "pos_count": [2],
        "neg_count": [1],
        "article_count": [4],
    }
)


def _as_selected(rows, columns):
    """Stored rows as the feature query returns them: the UTC date, then
    the selected columns, missing values as NaN."""
    frame = pd.DataFrame(rows)
    out = pd.DataFrame({"date": pd.to_datetime(frame["ts"], utc=True).dt.date})
    for column in columns:
        out[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    return out


def test_load_features_builds_in_process_from_the_bars_and_the_articles(
    read_tables, stored_bars
):
    """No canned frame for technical_indicators or anomalies: reading either
    would fail the test."""
    read_tables["raw_prices"] = stored_bars
    read_tables["news_sentiment"] = ARTICLES

    live = engineering.load_features("AAPL")
    built = engineering.build_features(
        daily_bars(stored_bars, "AAPL"), ARTICLES, "AAPL"
    )

    pd.testing.assert_frame_equal(live, built)
    assert built["zscore"].notna().any() and built["rsi_14"].notna().any()
    assert built["article_count"].notna().sum() == 1


def test_built_features_match_the_rows_the_stages_store(
    fake_db, read_tables, stored_bars
):
    read_tables["raw_prices"] = stored_bars
    indicators.run(["AAPL"])
    anomalies.run(["AAPL"])
    assert fake_db.tables["technical_indicators"] and fake_db.tables["anomalies"]

    stored = engineering.merge_feature_inputs(
        daily_bars(stored_bars, "AAPL"),
        _as_selected(
            fake_db.tables["technical_indicators"], engineering.INDICATOR_COLUMNS
        ),
        _as_selected(fake_db.tables["anomalies"], ["zscore"]),
        engineering.daily_sentiment(ARTICLES, DAYS),
    )
    built = engineering.build_features(
        daily_bars(stored_bars, "AAPL"), ARTICLES, "AAPL"
    )

    pd.testing.assert_frame_equal(built, stored, check_dtype=False)


BAD_ROWS = pd.DataFrame(
    [
        # Written while FinBERT could not load (removed in 29d1957).
        _article("2026-03-02 16:00", "neutral", 0.0, score_pos=0.33, score_neg=0.33,
                 score_neu=0.34),
        # MarketWatch's general-market feed, once credited to AAPL.
        _article("2026-03-04 16:00", "negative", -0.9,
                 url_domain="feeds.marketwatch.com"),
    ]
)  # fmt: skip


def test_bad_sentiment_rows_change_no_feature(stored_bars):
    bars = daily_bars(stored_bars, "AAPL")
    clean = engineering.build_features(bars, ARTICLES, "AAPL")
    with_bad = engineering.build_features(
        bars, pd.concat([ARTICLES, BAD_ROWS], ignore_index=True), "AAPL"
    )
    pd.testing.assert_frame_equal(with_bad, clean)


# ── Which trading date an article counts on (decision D7) ────────────────────


def _counts(stored_bars, *articles):
    built = engineering.build_features(
        daily_bars(stored_bars, "AAPL"), pd.DataFrame(articles), "AAPL"
    )
    counts = built.set_index("date")["article_count"]
    return {d.date().isoformat(): int(n) for d, n in counts[counts > 0].items()}


def test_a_weekend_article_counts_on_monday(stored_bars):
    saturday = _article("2026-03-07 15:00", "positive", 0.6)
    assert _counts(stored_bars, saturday) == {"2026-03-09": 1}


def test_a_newsapi_article_counts_from_a_day_after_it_was_published(stored_bars):
    newsapi = _article("2026-03-03 15:00", "positive", 0.6, source="newsapi",
                       inserted_at=pd.Timestamp("2026-03-03 15:05", tz="UTC"))  # fmt: skip
    assert _counts(stored_bars, newsapi) == {"2026-03-04": 1}


def test_an_article_stored_late_counts_from_when_it_arrived(stored_bars):
    late = _article("2026-03-03 15:00", "positive", 0.6,
                    inserted_at=pd.Timestamp("2026-03-06 23:00", tz="UTC"))  # fmt: skip
    assert _counts(stored_bars, late) == {"2026-03-06": 1}


def test_a_late_scheduled_run_stores_for_the_day_it_forecasts_from(stored_bars):
    """GitHub started the 22:30 UTC run as late as 06:45 UTC the next day."""
    fetched = _article("2026-03-03 21:00", "positive", 0.6,
                       inserted_at=pd.Timestamp("2026-03-04 06:50", tz="UTC"))  # fmt: skip
    assert _counts(stored_bars, fetched) == {"2026-03-03": 1}


def test_sentiment_is_averaged_over_each_dates_news_window():
    daily = engineering.daily_sentiment(ARTICLES, DAYS)
    expected = STORED_SENTIMENT.assign(date=pd.to_datetime(STORED_SENTIMENT["date"]))
    pd.testing.assert_frame_equal(daily, expected, check_dtype=False)


def test_no_articles_gives_missing_sentiment_on_every_date(stored_bars):
    built = engineering.build_features(
        daily_bars(stored_bars, "AAPL"), pd.DataFrame(), "AAPL"
    )
    assert len(built) == N_DAYS
    assert built[engineering.SENTIMENT_COLUMNS].isna().all().all()


def test_too_short_for_indicators_still_gives_every_column(price_bars):
    bars = price_bars.frame(
        [price_bars.daily(day.date(), 100.0 + i) for i, day in enumerate(DAYS[:20])]
    )
    built = engineering.build_features(daily_bars(bars, "AAPL"), ARTICLES, "AAPL")
    assert len(built) == 20
    assert {*engineering.INDICATOR_COLUMNS, "zscore"} <= set(built.columns)
    assert built[engineering.INDICATOR_COLUMNS].isna().all().all()


def test_no_prices_gives_an_empty_frame():
    assert engineering.build_features(pd.DataFrame(), ARTICLES, "AAPL").empty


# ── What depends on later data ────────────────────────────────────────────────
#
# Indicators are trailing, so a later price cannot move an earlier one. The
# anomaly scores are not: the IQR fences and the z-score stored on an
# IQR-flagged date are fitted on the whole history. That is why the
# evaluation rebuilds features from the history up to each date instead of
# reading the stored rows, and why this test pins the behaviour.


def _built(closes, price_bars):
    bars = price_bars.frame(
        [
            price_bars.daily(day.date(), float(close))
            for day, close in zip(DAYS, closes, strict=True)
        ]
    )
    return engineering.build_features(daily_bars(bars, "AAPL"), ARTICLES, "AAPL")


def test_a_later_price_never_moves_an_earlier_indicator(price_bars):
    closes = _closes()
    moved = closes.copy()
    moved[80:] *= 1.5
    before, after = _built(closes, price_bars), _built(moved, price_bars)
    columns = engineering.INDICATOR_COLUMNS
    pd.testing.assert_frame_equal(before.loc[:79, columns], after.loc[:79, columns])


def test_a_later_price_can_move_an_earlier_anomaly_score(price_bars):
    closes = _closes()
    calmer = closes.copy()
    calmer[80:] = closes[79]  # flat after day 80: narrower IQR fences
    before, after = _built(closes, price_bars), _built(calmer, price_bars)
    earlier = slice(0, 79)
    assert not before.loc[earlier, "zscore"].equals(after.loc[earlier, "zscore"])


def test_news_not_yet_available_on_any_date_gives_missing_sentiment(stored_bars):
    after_the_last_cutoff = _article("2026-06-01 15:00", "positive", 0.6)
    assert _counts(stored_bars, after_the_last_cutoff) == {}


# ── Missing news is not neutral (decision D6, finding F6) ─────────────────────

# One fetch a day, at 23:00 UTC, as the daily workflow stores its articles.
FETCHES = pd.DatetimeIndex(DAYS + pd.Timedelta(hours=23)).tz_localize("UTC")
NEWS_COLUMNS = [
    "sentiment_compound", "sentiment_3d", "sentiment_change", "pos_count",
    "neg_count", "article_count", "has_news",
]  # fmt: skip


def _news(stored_bars, articles, fetched_at=FETCHES):
    raw = engineering.build_features(
        daily_bars(stored_bars, "AAPL"), pd.DataFrame(articles), "AAPL",
        fetched_at=fetched_at,
    )  # fmt: skip
    out = engineering.engineer_features(raw, drop_unlabeled=False)
    return out.set_index("date")[NEWS_COLUMNS]


MARCH = [
    _article("2026-03-02 14:00", "positive", 0.5),
    _article("2026-03-02 15:30", "positive", 0.3),
    _article("2026-03-04 14:00", "negative", -0.4),
]


def test_after_the_last_article_a_day_reads_no_news_not_the_last_value(stored_bars):
    day = _news(stored_bars, MARCH).loc["2026-03-05"]
    assert day["has_news"] == 0 and day["article_count"] == 0
    assert day["pos_count"] == 0 and day["neg_count"] == 0
    assert np.isnan(day["sentiment_compound"])  # not -0.4 carried forward


def test_before_the_first_article_every_news_feature_is_missing(stored_bars):
    """Not collected yet is not "no news"."""
    before = _news(stored_bars, MARCH).loc[:"2026-02-27"]
    assert len(before) > 10 and before.isna().all().all()


def test_a_window_with_no_fetch_for_any_ticker_has_every_news_feature_missing(
    stored_bars,
):
    no_fetch_on_the_5th = FETCHES[
        FETCHES.normalize() != pd.Timestamp("2026-03-05", tz="UTC")
    ]
    news = _news(stored_bars, MARCH, fetched_at=no_fetch_on_the_5th)
    assert news.loc["2026-03-05"].isna().all()
    assert news.loc["2026-03-06", "article_count"] == 0  # fetched, nothing found


def test_the_three_day_sentiment_averages_every_article_in_the_three_windows(
    stored_bars,
):
    news = _news(stored_bars, MARCH)["sentiment_3d"]
    assert news["2026-03-02"] == pytest.approx((0.5 + 0.3) / 2)
    assert news["2026-03-03"] == pytest.approx((0.5 + 0.3) / 2)
    assert news["2026-03-04"] == pytest.approx((0.5 + 0.3 - 0.4) / 3)
    assert news["2026-03-06"] == pytest.approx(-0.4)
    assert np.isnan(news["2026-03-09"])  # the 5th, 6th and 9th had none


def test_the_sentiment_change_needs_news_on_both_days(stored_bars):
    two_days = [*MARCH, _article("2026-03-05 14:00", "positive", 0.2)]
    change = _news(stored_bars, two_days)["sentiment_change"]
    assert np.isnan(change["2026-03-03"]) and np.isnan(change["2026-03-04"])
    assert change["2026-03-05"] == pytest.approx(0.2 - (-0.4))


def test_has_news_is_a_feature():
    assert "has_news" in engineering.get_feature_cols()
