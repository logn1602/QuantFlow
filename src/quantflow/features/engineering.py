"""Model feature matrix.

Builds, from daily bars and the stored news articles, the frame of prices,
technical indicators, anomaly scores and news sentiment that the gradient
boosters train on, and the prediction target. The live run and the
evaluation call the same function, build_features (decision D4,
docs/evaluation.md).

This module is leakage-critical. engineer_features is backward-looking by
construction: lags use a positive shift and rolling windows are trailing.
tests/unit/test_engineering.py asserts that by perturbing future rows and
requiring no earlier feature value to move, and includes a probe that plants
a deliberate leak to prove the check fires.

Missing news is not neutral (decision D6): a date whose window was fetched
and held no article reads has_news 0, counts 0 and a missing compound, never
the last known value; before the ticker's first article, and on a date whose
window saw no fetch for any ticker, every news feature is missing.
"""

import numpy as np
import pandas as pd

from quantflow.db.features import load_news_days
from quantflow.db.prices import PRICE_COLUMNS, indexed_by_date, load_daily_bars
from quantflow.features.anomalies import (
    STORED_DECIMALS,
    detect_anomalies,
    point_in_time_scores,
)
from quantflow.features.indicators import compute_indicators, rounded_for_storage
from quantflow.features.news import (
    day_window_dates,
    news_day,
    news_days,
    query_rules,
    total,
)
from quantflow.utils.logger import get_logger

logger = get_logger(__name__)

INDICATOR_COLUMNS = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_middle",
    "bb_lower",
]
SENTIMENT_COLUMNS = ["sentiment_compound", "pos_count", "neg_count", "article_count"]
COUNT_COLUMNS = ["pos_count", "neg_count", "article_count"]


def load_news() -> pd.DataFrame:
    """Every ticker's stored news summed per news day (features.news
    .news_days' frame): one read serves a whole run, every ticker."""
    return load_news_days(**query_rules())


def load_features(ticker: str, news: pd.DataFrame | None = None) -> pd.DataFrame:
    """The live run's feature frame for a ticker: build_features on the
    stored daily bars and news.

    Prices come from load_daily_bars, the same one-row-per-trading-date
    series the statistical models read, so every model sees the same close.
    news is load_news' frame, read once by the caller for every ticker; it
    is read here if not given. The ticker's own days give its sentiment, and
    every ticker's the days a fetch ran (decision D6). Nothing is read from
    the technical_indicators or anomalies tables, which are written for the
    dashboard only.
    """
    prices = load_daily_bars(ticker)
    if prices.empty:
        logger.warning(f"No price data for {ticker}")
        return pd.DataFrame()
    if news is None:
        news = load_news()
    return features_from_news(
        prices,
        news[news["ticker"] == ticker],
        news["stored_day"].drop_duplicates(),
    )


def build_features(
    prices: pd.DataFrame,
    articles: pd.DataFrame,
    ticker: str,
    fetched_at=None,
) -> pd.DataFrame:
    """The feature frame for a history: one function for the live run and
    the evaluation.

    prices are daily bars (load_daily_bars' shape) and articles the stored
    sentiment rows (db.features.ARTICLE_SELECT's columns). Indicators are computed from
    the prices exactly as their stage does, at the precision its table
    keeps; anomaly scores are point-in-time (anomalies.point_in_time_scores,
    decision D14): each date's from the prices up to it, where the stage
    fits its fences on the whole history. Sentiment is averaged over each trading
    date's news window (daily_sentiment), leaving out the rows no feature
    should use (FinBERT placeholders, the general feed credited to AAPL:
    features.news.bad_rows). fetched_at holds the times any fetch stored
    articles, for any ticker (the ticker's own arrival times if omitted):
    a date whose window holds none has every news column missing (D6). It
    depends only on its arguments, and no row depends on a later one: given
    the history up to a date, it gives what the live run would have built on
    that date, and its training rows are built the way its newest row is.

    The articles are summed per news day first (features.news.news_days),
    the form the live run reads them in, and the frame is built from those
    sums (features_from_news).
    """
    if prices.empty:
        return pd.DataFrame()
    days = news_days(articles)
    if fetched_at is None:
        fetched_days = days.loc[days["counts"], "stored_day"]
    else:
        fetched_days = news_day(fetched_at)
    return features_from_news(prices, days, fetched_days)


def features_from_news(
    prices: pd.DataFrame, days: pd.DataFrame, fetched_days
) -> pd.DataFrame:
    """build_features from a news-days summary (features.news.news_days'
    frame) instead of article rows: what the live run reads.

    fetched_days are the news days on which a fetch stored articles, for any
    ticker.
    """
    if prices.empty:
        return pd.DataFrame()

    indexed = indexed_by_date(prices, PRICE_COLUMNS)

    computed = compute_indicators(indexed)
    if computed.empty:
        indicators = _empty(INDICATOR_COLUMNS)
    else:
        indicators = _dated(rounded_for_storage(computed))

    anomalies = _dated(point_in_time_scores(indexed["close"]).to_frame("zscore"))

    merged = merge_feature_inputs(
        prices, indicators, anomalies, window_sentiment(days, prices["date"])
    )
    return _known_news(merged, fetched_days)


def stage_anomaly_scores(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """The anomaly scores the anomaly stage stores for the dashboard (date,
    zscore, flagged dates only), fitted on the whole history and rounded as
    its table keeps them. No feature uses them; the evaluation's one-off
    check compares them with the stored rows."""
    indexed = indexed_by_date(prices, PRICE_COLUMNS)
    flags = detect_anomalies(indexed[["close", "volume"]], ticker)
    if flags.empty:
        return _empty(["zscore"])
    return _dated(flags[["zscore"]].astype(float).round(STORED_DECIMALS))


def _known_news(frame: pd.DataFrame, fetched_days) -> pd.DataFrame:
    """True counts where the news could be known, missing where it could
    not (decision D6).

    A date is known from the ticker's first article on, if a fetch for any
    ticker stored something in its window (or the date has an article):
    there, counts read 0 when the window held no article. Before the first
    article ("not collected" is not "no news"), and on a date whose window
    saw no fetch at all (nothing was looked for), every sentiment column is
    missing, counts included. fetched_days are the news days a fetch stored
    articles on.
    """
    dates = frame["date"]
    with_articles = frame["article_count"].notna()
    fetched_dates = set(day_window_dates(fetched_days, dates).dropna())
    fetched = dates.isin(fetched_dates) | with_articles
    if with_articles.any():
        collected = dates >= dates[with_articles].min()
    else:
        collected = pd.Series(False, index=frame.index)
    known = fetched & collected
    out = frame.copy()
    out[COUNT_COLUMNS] = out[COUNT_COLUMNS].fillna(0).where(known)
    out["sentiment_compound"] = out["sentiment_compound"].where(known)
    return out


def _empty(columns: list[str]) -> pd.DataFrame:
    """No rows, but typed columns, so merging it gives float columns of
    missing values rather than object ones."""
    return pd.DataFrame(
        {"date": pd.Series(dtype="datetime64[ns]")}
        | {column: pd.Series(dtype=float) for column in columns}
    )


def _dated(frame: pd.DataFrame) -> pd.DataFrame:
    """A frame indexed by 00:00 UTC timestamps, with the date as a column,
    as the stored queries return it."""
    dates = pd.DatetimeIndex(frame.index).tz_convert("UTC").tz_localize(None)
    return frame.reset_index(drop=True).assign(date=dates.normalize())


def daily_sentiment(articles: pd.DataFrame, trading_dates) -> pd.DataFrame:
    """Articles averaged over each trading date's news window: mean
    compound, positive and negative counts, and every article counted.

    An article counts on the trading date whose window holds the moment it
    became available to a run (decision D7, features.news): the later of
    its publish time plus its source's delay and its arrival in the
    database, with each window closing at 12:00 UTC the next day. So weekend
    news lands on Monday, NewsAPI's a day after publication, and an article
    fetched late from when it arrived, in training exactly as live. One that
    became available after the last date's cutoff counts nowhere yet. The
    rows no feature uses (features.news.bad_rows) are left out.
    """
    return window_sentiment(news_days(articles), trading_dates)


def window_sentiment(days: pd.DataFrame, trading_dates) -> pd.DataFrame:
    """daily_sentiment from a news-days summary: each trading date's window
    is a run of whole news days, so its counts and compound sum are theirs
    added up.

    The mean is the exact sum of the compounds, rounded once to a float,
    over their count. It used to be pandas' mean of the rows as floats; the
    two agree to the last bit or differ in it (at most 1.1e-16 on the
    2026-11-06 snapshot), and only an exact sum is the same whatever order
    the rows come in and whether they are added up per day or per window.
    """
    counted = days[days["counts"].astype(bool)]
    if counted.empty:
        return _empty(SENTIMENT_COLUMNS)
    dated = counted.assign(
        date=day_window_dates(counted["available_day"], trading_dates).to_numpy()
    ).dropna(subset=["date"])
    if dated.empty:
        return _empty(SENTIMENT_COLUMNS)
    by_date = dated.groupby("date", sort=True)
    sums = by_date[["articles", "positive", "negative", "compounds"]].sum()
    compound_sum = by_date["compound_sum"].agg(total)
    mean = [
        float(s) / n if n else np.nan
        for s, n in zip(compound_sum, sums["compounds"], strict=True)
    ]
    return pd.DataFrame(
        {
            "sentiment_compound": np.asarray(mean, dtype=float),
            "pos_count": sums["positive"],
            "neg_count": sums["negative"],
            "article_count": sums["articles"],
        },
        index=sums.index,
    ).reset_index()


def merge_feature_inputs(
    prices: pd.DataFrame,
    indicators: pd.DataFrame,
    anomalies: pd.DataFrame,
    sentiment: pd.DataFrame,
) -> pd.DataFrame:
    """Prices with each date's indicators, anomaly score and sentiment.

    One rule whether the rows were computed or read from the tables (the
    evaluation's one-off check compares the two): a date keeps its first
    indicator and anomaly row, and a date with none of a signal gets missing
    values, which engineer_features then fills.
    """
    indicators = indicators.assign(date=pd.to_datetime(indicators["date"]))
    indicators = indicators.drop_duplicates("date")

    anomalies = anomalies.assign(date=pd.to_datetime(anomalies["date"]))
    anomalies = anomalies.drop_duplicates("date")

    sentiment = sentiment.assign(date=pd.to_datetime(sentiment["date"]))

    df = prices.merge(indicators, on="date", how="left")
    df = df.merge(anomalies, on="date", how="left")
    df = df.merge(sentiment, on="date", how="left")

    return df


def bollinger_features(
    close: pd.Series, upper: pd.Series, lower: pd.Series, middle: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """bb_position, where the close sits in the band (0 on the lower band, 1
    on the upper, beyond them outside it, never clipped), and bb_width, the
    band's width over its middle, a volatility measure. Missing where either
    would divide by zero. One definition for training and the boosters'
    rollout (boosting.generate_forecast)."""
    width = upper - lower
    position = (close - lower) / width.replace(0, np.nan)
    return position, width / middle.replace(0, np.nan)


def engineer_features(df: pd.DataFrame, drop_unlabeled: bool = True) -> pd.DataFrame:
    """
    Build the full feature matrix from raw merged data.
    All features are derived from past data only: no future leakage.

    drop_unlabeled=False keeps the newest row, whose target (the next day's
    close) is unknown. Training and scoring must not see it; the live
    forecast needs exactly it, because its features describe the latest
    trading day and its target is the first day to forecast.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # ── Price-based features ─────────────────────────────────────────────────
    # Lagged closes
    for lag in [1, 2, 3, 5, 10]:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)

    # Daily return
    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)

    # Rolling statistics
    for window in [5, 10, 20]:
        df[f"rolling_mean_{window}"] = df["close"].rolling(window).mean()
        df[f"rolling_std_{window}"] = df["close"].rolling(window).std()

    # High-Low range
    df["hl_range"] = df["high"] - df["low"]

    # Volume change
    df["volume_change"] = df["volume"].pct_change(1)

    # ── Bollinger Band features ──────────────────────────────────────────────
    df["bb_position"], df["bb_width"] = bollinger_features(
        df["close"], df["bb_upper"], df["bb_lower"], df["bb_middle"]
    )

    # ── Anomaly features ─────────────────────────────────────────────────────
    # zscore is 0 on a date neither detector flags and missing in their warmup.
    df["is_anomaly"] = (
        (df["zscore"].abs() >= 2.0).astype(float).where(df["zscore"].notna())
    )

    # ── Sentiment features (decision D6) ─────────────────────────────────────
    # A missing article_count means the news could not be known that day
    # (not collected yet, or no fetch): every news feature is missing there.
    # A count of 0 means fetched and nothing found: has_news 0, counts 0 and
    # no compound. Nothing is carried forward from an earlier day.
    counts = df["article_count"]
    known = counts.notna()
    with_news = counts > 0
    compound = df["sentiment_compound"].where(with_news)
    df["has_news"] = with_news.astype(float).where(known)
    df["sentiment_compound"] = compound

    # The mean compound of every article in the last three dates' windows.
    compound_sum = (compound * counts).where(with_news, 0.0).where(known)
    total = compound_sum.rolling(3, min_periods=1).sum()
    articles = counts.rolling(3, min_periods=1).sum()
    df["sentiment_3d"] = (total / articles.where(articles > 0)).where(known)

    # Sentiment momentum: missing unless both days had news.
    df["sentiment_change"] = compound - compound.shift(1)

    # ── Target variables ─────────────────────────────────────────────────────
    df["target_price"] = df["close"].shift(-1)  # next day's price
    df["target_direction"] = (df["target_price"] > df["close"]).astype(
        int
    )  # 1=up, 0=down

    # Drop rows with insufficient history, and rows with no known target
    # unless the caller asked to keep the newest one.
    df = df.dropna(subset=["close_lag_10", "rolling_mean_20"])
    if drop_unlabeled:
        df = df.dropna(subset=["target_price"])

    return df


def get_feature_cols() -> list[str]:
    """Return the list of feature column names used for training."""
    return [
        # Price lags
        "close_lag_1",
        "close_lag_2",
        "close_lag_3",
        "close_lag_5",
        "close_lag_10",
        # Returns
        "return_1d",
        "return_5d",
        # Rolling stats
        "rolling_mean_5",
        "rolling_mean_10",
        "rolling_mean_20",
        "rolling_std_5",
        "rolling_std_10",
        "rolling_std_20",
        # OHLCV
        "open",
        "high",
        "low",
        "volume",
        "hl_range",
        "volume_change",
        # Technical indicators
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_position",
        "bb_width",
        # Anomaly
        "zscore",
        "is_anomaly",
        # Sentiment
        "sentiment_compound",
        "sentiment_3d",
        "sentiment_change",
        "pos_count",
        "neg_count",
        "article_count",
        "has_news",
    ]
