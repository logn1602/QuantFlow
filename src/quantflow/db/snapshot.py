"""The reads behind an evaluation snapshot (docs/evaluation.md, section 6).

Run on a connection from db.readonly.read_only_connection, so every table is
read from one snapshot of the database and nothing can be written. Prices
are read raw, on that connection, because load_daily_bars opens its own;
the caller builds daily bars from them with the same daily_bars() rule.
"""

import pandas as pd
from sqlalchemy import text

from quantflow.db.features import ARTICLE_SELECT
from quantflow.utils.trading_calendar import news_cutoff

_RAW_BARS = text("""
    SELECT ticker, ts, open, high, low, close, volume
    FROM raw_prices
    WHERE source = 'yfinance' AND ticker = ANY(:tickers) AND ts < :until
    ORDER BY ticker, ts
""")

# The live run's article columns (URL host only, no headlines or links).
_ARTICLES = text(
    ARTICLE_SELECT
    + "WHERE ticker = ANY(:tickers) AND inserted_at <= :stored_by "
    + "ORDER BY ticker, inserted_at"
)

_STORED_INDICATORS = text("""
    SELECT ticker, (ts AT TIME ZONE 'UTC')::date AS date,
           rsi_14, macd, macd_signal, macd_hist, bb_upper, bb_middle, bb_lower
    FROM technical_indicators
    WHERE ticker = ANY(:tickers)
    ORDER BY ticker, ts
""")

_STORED_ANOMALIES = text("""
    SELECT ticker, (ts AT TIME ZONE 'UTC')::date AS date, zscore
    FROM anomalies
    WHERE ticker = ANY(:tickers)
    ORDER BY ticker, ts
""")


def read_snapshot_tables(conn, tickers: list[str], as_of) -> dict[str, pd.DataFrame]:
    """Everything a snapshot holds, as of the trading date `as_of`.

    Raw bars dated on or before as_of; articles stored by as_of's news
    cutoff, which is what a run that day could have used; and the stored
    indicator and anomaly rows, only to check once that rebuilding them
    gives what the live run reads.
    """
    as_of = pd.Timestamp(as_of).normalize()
    until = (as_of + pd.Timedelta(days=1)).tz_localize("UTC")
    params = {"tickers": list(tickers)}
    return {
        "raw_bars": pd.read_sql(_RAW_BARS, conn, params=params | {"until": until}),
        "articles": pd.read_sql(
            _ARTICLES, conn, params=params | {"stored_by": news_cutoff(as_of)}
        ),
        "stored_indicators": pd.read_sql(_STORED_INDICATORS, conn, params=params),
        "stored_anomalies": pd.read_sql(_STORED_ANOMALIES, conn, params=params),
    }
