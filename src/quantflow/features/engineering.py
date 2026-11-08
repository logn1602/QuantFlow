"""Model feature matrix.

Builds, from daily bars and the stored news articles, the frame of prices,
technical indicators, anomaly scores and news sentiment that the gradient
boosters train on, and the prediction target. The live run and the
evaluation call the same function, build_features (decision D4,
docs/evaluation.md).

This module is leakage-critical. Every transform here is backward-looking by
construction: lags use a positive shift, rolling windows are trailing, and
sentiment is forward-filled so a day with no news inherits the last known
score rather than seeing a future one. tests/unit/test_engineering.py asserts
that by perturbing future rows and requiring no earlier feature value to move,
and includes a probe that plants a deliberate leak to prove the check fires.
"""

import numpy as np
import pandas as pd

from quantflow.db.features import load_articles
from quantflow.db.prices import PRICE_COLUMNS, indexed_by_date, load_daily_bars
from quantflow.features.anomalies import STORED_DECIMALS, detect_anomalies
from quantflow.features.indicators import compute_indicators, rounded_for_storage
from quantflow.features.news import bad_rows
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


def load_features(ticker: str) -> pd.DataFrame:
    """The live run's feature frame for a ticker: build_features on the
    stored daily bars and articles.

    Prices come from load_daily_bars, the same one-row-per-trading-date
    series the statistical models read, so every model sees the same close.
    Nothing is read from the technical_indicators or anomalies tables, which
    are written for the dashboard only.
    """
    prices = load_daily_bars(ticker)
    if prices.empty:
        logger.warning(f"No price data for {ticker}")
        return pd.DataFrame()
    return build_features(prices, load_articles(ticker), ticker)


def build_features(
    prices: pd.DataFrame, articles: pd.DataFrame, ticker: str
) -> pd.DataFrame:
    """The feature frame for a history: one function for the live run and
    the evaluation.

    prices are daily bars (load_daily_bars' shape) and articles the stored
    sentiment rows (load_articles' columns). Indicators and anomaly scores
    are computed from the prices exactly as their stages do, at the
    precision the tables keep, and sentiment is averaged by UTC publish
    date, leaving out the rows no feature should use (FinBERT placeholders,
    the general feed credited to AAPL: features.news.bad_rows). It depends
    only on its arguments: given the history up to
    a date, it gives what the live run would have built on that date. The
    stored anomaly rows cannot do that, because the IQR fences behind them
    are fitted on the whole history.
    """
    if prices.empty:
        return pd.DataFrame()

    indexed = indexed_by_date(prices, PRICE_COLUMNS)

    computed = compute_indicators(indexed)
    if computed.empty:
        indicators = _empty(INDICATOR_COLUMNS)
    else:
        indicators = _dated(rounded_for_storage(computed))

    flags = detect_anomalies(indexed[["close", "volume"]], ticker)
    if flags.empty:
        anomalies = _empty(["zscore"])
    else:
        anomalies = _dated(flags[["zscore"]].astype(float).round(STORED_DECIMALS))

    if not articles.empty:
        articles = articles[~bad_rows(articles)]
    return merge_feature_inputs(
        prices, indicators, anomalies, daily_sentiment(articles)
    )


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


def daily_sentiment(articles: pd.DataFrame) -> pd.DataFrame:
    """Articles averaged by UTC publish date: mean compound, positive and
    negative counts, and every article counted."""
    if articles.empty:
        return _empty(SENTIMENT_COLUMNS)
    published = pd.to_datetime(articles["published_at"], utc=True)
    by_date = articles.assign(
        date=published.dt.tz_localize(None).dt.normalize(),
        compound=pd.to_numeric(articles["compound"]),
        positive=articles["sentiment"].eq("positive"),
        negative=articles["sentiment"].eq("negative"),
    ).groupby("date", sort=True)
    return pd.DataFrame(
        {
            "sentiment_compound": by_date["compound"].mean(),
            "pos_count": by_date["positive"].sum(),
            "neg_count": by_date["negative"].sum(),
            "article_count": by_date.size(),
        }
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
    # BB position: where is price relative to the bands? (0=lower, 1=upper)
    bb_range = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / bb_range.replace(0, np.nan)

    # BB width: measures volatility
    df["bb_width"] = bb_range / df["bb_middle"].replace(0, np.nan)

    # ── Anomaly features ─────────────────────────────────────────────────────
    df["zscore"] = df["zscore"].fillna(0)
    df["is_anomaly"] = (df["zscore"].abs() >= 2.0).astype(int)

    # ── Sentiment features ───────────────────────────────────────────────────
    # ffill first so days with no news inherit the last known sentiment,
    # rather than defaulting to 0 (neutral) which dilutes the signal.
    df["sentiment_compound"] = df["sentiment_compound"].ffill().fillna(0)
    df["pos_count"] = df["pos_count"].ffill().fillna(0)
    df["neg_count"] = df["neg_count"].ffill().fillna(0)
    df["article_count"] = df["article_count"].ffill().fillna(0)

    # Rolling sentiment (3-day average)
    df["sentiment_3d"] = df["sentiment_compound"].rolling(3).mean().fillna(0)

    # Sentiment momentum
    df["sentiment_change"] = df["sentiment_compound"].diff().fillna(0)

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
    ]
