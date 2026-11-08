"""Reads of the derived rows the feature matrix merges onto prices.

Every date here is the UTC calendar date, the same one load_daily_bars gives
each price row. A bare ts::date uses the session's TimeZone instead: UTC on
Supabase, but on a server set to New York it moved every indicator, anomaly
score and sentiment average to the previous day's price row.
"""

import pandas as pd
from sqlalchemy import text

from quantflow.db.connection import get_engine

_INDICATORS = text("""
    SELECT (ts AT TIME ZONE 'UTC')::date AS date,
           rsi_14, macd, macd_signal, macd_hist,
           bb_upper, bb_middle, bb_lower
    FROM technical_indicators
    WHERE ticker = :t
    ORDER BY ts ASC
""")

_ANOMALIES = text("""
    SELECT (ts AT TIME ZONE 'UTC')::date AS date, zscore
    FROM anomalies
    WHERE ticker = :t
    ORDER BY ts ASC
""")

# The daily average by UTC publish date. Which articles a trading date may
# use is decided in Phase 2 (docs/evaluation.md, section 4); until then this
# is the rule the live run uses.
_DAILY_SENTIMENT = text("""
    SELECT
        (published_at AT TIME ZONE 'UTC')::date AS date,
        AVG(compound)                       AS sentiment_compound,
        COUNT(*) FILTER (WHERE sentiment='positive') AS pos_count,
        COUNT(*) FILTER (WHERE sentiment='negative') AS neg_count,
        COUNT(*)                            AS article_count
    FROM news_sentiment
    WHERE ticker = :t
    GROUP BY (published_at AT TIME ZONE 'UTC')::date
    ORDER BY date ASC
""")


def load_stored_feature_inputs(ticker: str) -> dict[str, pd.DataFrame]:
    """The ticker's stored indicator rows, anomaly scores and daily sentiment,
    read on one connection: {"indicators", "anomalies", "sentiment"}."""
    params = {"t": ticker}
    with get_engine().connect() as conn:
        return {
            "indicators": pd.read_sql(_INDICATORS, conn, params=params),
            "anomalies": pd.read_sql(_ANOMALIES, conn, params=params),
            "sentiment": pd.read_sql(_DAILY_SENTIMENT, conn, params=params),
        }
