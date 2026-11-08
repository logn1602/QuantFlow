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


def _article(ts, sentiment, compound):
    return {
        "published_at": pd.Timestamp(ts, tz="UTC"),
        "sentiment": sentiment,
        "compound": compound,
    }


ARTICLES = pd.DataFrame(
    [
        _article("2026-03-02 14:00", "positive", 0.5),
        _article("2026-03-02 15:30", "positive", 0.3),
        # Late evening, still 2 March in UTC: the stored query groups by UTC date.
        _article("2026-03-02 23:30", "negative", -0.4),
        # Just after midnight UTC: 3 March.
        _article("2026-03-03 00:30", "neutral", 0.0),
    ]
)

# The daily averages for ARTICLES, worked out by hand.
STORED_SENTIMENT = pd.DataFrame(
    {
        "date": [pd.Timestamp("2026-03-02").date(), pd.Timestamp("2026-03-03").date()],
        "sentiment_compound": [(0.5 + 0.3 - 0.4) / 3, 0.0],
        "pos_count": [2, 0],
        "neg_count": [1, 0],
        "article_count": [3, 1],
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
    assert built["article_count"].notna().sum() == 2


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
        engineering.daily_sentiment(ARTICLES),
    )
    built = engineering.build_features(
        daily_bars(stored_bars, "AAPL"), ARTICLES, "AAPL"
    )

    pd.testing.assert_frame_equal(built, stored, check_dtype=False)


def test_sentiment_is_averaged_by_utc_publish_date():
    daily = engineering.daily_sentiment(ARTICLES)
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
