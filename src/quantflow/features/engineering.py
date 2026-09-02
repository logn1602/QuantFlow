"""Model feature matrix.

Merges prices, technical indicators, anomaly scores and news sentiment into the
33-column frame the gradient boosters train on, and builds the prediction
target.

This module is leakage-critical. Every transform here is backward-looking by
construction: lags use a positive shift, rolling windows are trailing, and
sentiment is forward-filled so a day with no news inherits the last known
score rather than seeing a future one. tests/unit/test_engineering.py asserts
that by perturbing future rows and requiring no earlier feature value to move,
and includes a probe that plants a deliberate leak to prove the check fires.
"""

import numpy as np
import pandas as pd
from sqlalchemy import text

from quantflow.db.connection import get_engine
from quantflow.utils.logger import get_logger

logger = get_logger(__name__)


def load_features(ticker: str) -> pd.DataFrame:
    """
    Load and merge all available signals for a ticker:
    prices + technical indicators + anomalies + sentiment
    """
    engine = get_engine()

    with engine.connect() as conn:
        # Base prices
        prices = pd.read_sql(
            text("""
            SELECT ts::date AS date, open, high, low, close, volume
            FROM raw_prices
            WHERE ticker = :t AND source = 'yfinance'
            ORDER BY ts ASC
        """),
            conn,
            params={"t": ticker},
        )

        if prices.empty:
            logger.warning(f"No price data for {ticker}")
            return pd.DataFrame()

        # Technical indicators
        indicators = pd.read_sql(
            text("""
            SELECT ts::date AS date, rsi_14, macd, macd_signal, macd_hist,
                   bb_upper, bb_middle, bb_lower
            FROM technical_indicators
            WHERE ticker = :t
            ORDER BY ts ASC
        """),
            conn,
            params={"t": ticker},
        )

        # Anomalies
        anomalies = pd.read_sql(
            text("""
            SELECT ts::date AS date, zscore
            FROM anomalies
            WHERE ticker = :t
            ORDER BY ts ASC
        """),
            conn,
            params={"t": ticker},
        )

        # Sentiment — daily average
        sentiment = pd.read_sql(
            text("""
            SELECT
                published_at::date                  AS date,
                AVG(compound)                       AS sentiment_compound,
                COUNT(*) FILTER (WHERE sentiment='positive') AS pos_count,
                COUNT(*) FILTER (WHERE sentiment='negative') AS neg_count,
                COUNT(*)                            AS article_count
            FROM news_sentiment
            WHERE ticker = :t
            GROUP BY published_at::date
            ORDER BY date ASC
        """),
            conn,
            params={"t": ticker},
        )

    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.drop_duplicates("date").sort_values("date").reset_index(drop=True)

    indicators["date"] = pd.to_datetime(indicators["date"])
    indicators = indicators.drop_duplicates("date")

    anomalies["date"] = pd.to_datetime(anomalies["date"])
    anomalies = anomalies.drop_duplicates("date")

    sentiment["date"] = pd.to_datetime(sentiment["date"])

    # Merge everything
    df = prices.merge(indicators, on="date", how="left")
    df = df.merge(anomalies, on="date", how="left")
    df = df.merge(sentiment, on="date", how="left")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix from raw merged data.
    All features are derived from past data only — no future leakage.
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

    # Drop rows with NaN targets or insufficient history
    df = df.dropna(subset=["target_price", "close_lag_10", "rolling_mean_20"])

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
