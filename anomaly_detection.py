"""
anomaly_detection.py
--------------------
Phase 3 — Anomaly Detection Engine

Detects unusual price movements using two methods:
  1. Z-Score       : How many standard deviations from the rolling mean
  2. IQR Method    : Flags prices outside 1.5x the interquartile range

Flags are written to the anomalies table with:
  - The date and closing price
  - The Z-score at that point
  - Flag type: 'HIGH' (spike up) or 'LOW' (spike down)

Usage:
    python anomaly_detection.py               # run for all tickers
    python anomaly_detection.py --ticker TSLA # run for one ticker
    python anomaly_detection.py --show TSLA   # print anomalies for a ticker
    python anomaly_detection.py --summary     # show anomaly counts per ticker
"""

import sys
import os
import argparse

import pandas as pd
import numpy as np
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(__file__))
from config import TICKERS
from db.connection import get_engine
from utils.logger import get_logger

logger = get_logger("anomaly_detection")

# ── Config ────────────────────────────────────────────────────────────────────
ZSCORE_WINDOW    = 30    # rolling window for mean/std calculation (trading days)
ZSCORE_THRESHOLD = 2.0   # flag if |z-score| exceeds this value
IQR_MULTIPLIER   = 1.5   # standard IQR fence multiplier


# ── Data loading ──────────────────────────────────────────────────────────────

def load_prices(ticker: str, source: str = "yfinance") -> pd.DataFrame:
    """Load daily close prices for a ticker, sorted oldest to newest."""
    engine = get_engine()
    query = text("""
        SELECT ts, close, volume
        FROM raw_prices
        WHERE ticker = :ticker
          AND source = :source
        ORDER BY ts ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker, "source": source})

    if df.empty:
        logger.warning(f"No price data found for {ticker}")
        return df

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts")
    df["close"]  = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df


# ── Anomaly detection ─────────────────────────────────────────────────────────

def detect_zscore_anomalies(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Rolling Z-Score anomaly detection on daily close prices.

    Z = (price - rolling_mean) / rolling_std

    A Z-score > 2.0 means the price is more than 2 standard deviations
    above the recent average — statistically unusual (~5% of trading days).

    Returns DataFrame of flagged rows only.
    """
    if len(df) < ZSCORE_WINDOW:
        logger.warning(f"{ticker}: Not enough rows for Z-score (need {ZSCORE_WINDOW})")
        return pd.DataFrame()

    result = df.copy()

    # Rolling mean and std over ZSCORE_WINDOW days
    result["rolling_mean"] = result["close"].rolling(window=ZSCORE_WINDOW).mean()
    result["rolling_std"]  = result["close"].rolling(window=ZSCORE_WINDOW).std()

    # Compute Z-score
    result["zscore"] = (
        (result["close"] - result["rolling_mean"]) / result["rolling_std"]
    )

    # Drop warm-up rows where rolling stats aren't ready
    result = result.dropna(subset=["zscore"])

    # Flag anomalies
    anomalies = result[result["zscore"].abs() >= ZSCORE_THRESHOLD].copy()
    anomalies["flag"] = anomalies["zscore"].apply(
        lambda z: "HIGH" if z > 0 else "LOW"
    )
    anomalies["ticker"] = ticker

    return anomalies[["ticker", "close", "zscore", "flag"]]


def detect_iqr_anomalies(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    IQR-based anomaly detection on daily returns (% change).

    Catches sudden large moves regardless of the price level —
    better for catching single-day crash/spike events.

    Returns DataFrame of flagged rows only.
    """
    result = df.copy()

    # Daily return as percentage
    result["daily_return"] = result["close"].pct_change() * 100
    result = result.dropna(subset=["daily_return"])

    # IQR fences
    Q1 = result["daily_return"].quantile(0.25)
    Q3 = result["daily_return"].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = Q1 - IQR_MULTIPLIER * IQR
    upper_fence = Q3 + IQR_MULTIPLIER * IQR

    anomalies = result[
        (result["daily_return"] < lower_fence) |
        (result["daily_return"] > upper_fence)
    ].copy()

    anomalies["flag"] = anomalies["daily_return"].apply(
        lambda r: "HIGH" if r > 0 else "LOW"
    )
    anomalies["zscore"] = (
        (anomalies["daily_return"] - result["daily_return"].mean()) /
        result["daily_return"].std()
    )
    anomalies["ticker"] = ticker

    return anomalies[["ticker", "close", "zscore", "flag"]]


# ── Database write ────────────────────────────────────────────────────────────

def save_anomalies(df: pd.DataFrame) -> int:
    """
    Insert anomaly rows into the anomalies table.
    Returns number of rows inserted.
    """
    if df.empty:
        return 0

    engine = get_engine()
    inserted = 0

    with engine.begin() as conn:
        for ts, row in df.iterrows():
            try:
                conn.execute(
                    text("""
                        INSERT INTO anomalies
                            (ticker, ts, close, zscore, flag)
                        VALUES
                            (:ticker, :ts, :close, :zscore, :flag)
                        ON CONFLICT DO NOTHING
                    """),
                    {
                        "ticker": row["ticker"],
                        "ts":     ts,
                        "close":  round(float(row["close"]),  4),
                        "zscore": round(float(row["zscore"]), 4),
                        "flag":   row["flag"],
                    }
                )
                inserted += 1
            except Exception as e:
                logger.warning(f"Row skipped: {e}")

    return inserted


# ── Display helpers ───────────────────────────────────────────────────────────

def show_anomalies(ticker: str, n: int = 15):
    """Print the most recent anomaly flags for a ticker."""
    engine = get_engine()
    query = text("""
        SELECT
            DATE(ts)               AS date,
            ROUND(close::numeric, 2)  AS close,
            ROUND(zscore::numeric, 3) AS zscore,
            flag
        FROM anomalies
        WHERE ticker = :ticker
        ORDER BY ts DESC
        LIMIT :n
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker, "n": n})

    if df.empty:
        print(f"No anomalies found for {ticker}.")
        return

    # Color coding in terminal
    high_count = len(df[df["flag"] == "HIGH"])
    low_count  = len(df[df["flag"] == "LOW"])

    print(f"\n{'='*60}")
    print(f"  Anomaly flags for {ticker} (latest {n})")
    print(f"  HIGH spikes: {high_count}  |  LOW crashes: {low_count}")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print()


def show_summary():
    """Print anomaly counts per ticker."""
    engine = get_engine()
    query = text("""
        SELECT
            ticker,
            COUNT(*) FILTER (WHERE flag = 'HIGH') AS spikes,
            COUNT(*) FILTER (WHERE flag = 'LOW')  AS crashes,
            COUNT(*)                               AS total,
            ROUND(MAX(ABS(zscore))::numeric, 2)   AS max_zscore,
            DATE(MAX(ts))                          AS last_anomaly
        FROM anomalies
        GROUP BY ticker
        ORDER BY total DESC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        print("No anomalies in database yet. Run: python anomaly_detection.py")
        return

    print(f"\n{'='*65}")
    print("  Anomaly Summary — all tickers")
    print(f"{'='*65}")
    print(df.to_string(index=False))
    print()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(tickers: list[str] = None) -> dict:
    """
    Full pipeline: detect anomalies using both methods, merge, save.
    Returns dict of {ticker: rows_inserted}.
    """
    tickers = tickers or TICKERS
    results = {}

    for ticker in tickers:
        logger.info(f"Detecting anomalies for {ticker}...")

        df = load_prices(ticker)
        if df.empty:
            results[ticker] = 0
            continue

        # Run both detection methods
        zscore_anomalies = detect_zscore_anomalies(df, ticker)
        iqr_anomalies    = detect_iqr_anomalies(df, ticker)

        # Merge and deduplicate by timestamp
        combined = pd.concat([zscore_anomalies, iqr_anomalies])
        combined = combined[~combined.index.duplicated(keep="first")]
        combined = combined.sort_index()

        n = save_anomalies(combined)
        results[ticker] = n
        logger.info(f"  {ticker}: {n} anomalies flagged")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly detection engine")
    parser.add_argument("--ticker",  type=str, help="Single ticker (e.g. TSLA)")
    parser.add_argument("--show",    type=str, help="Print anomalies for a ticker")
    parser.add_argument("--summary", action="store_true", help="Show anomaly counts per ticker")
    parser.add_argument("--rows",    type=int, default=15, help="Rows to display (default: 15)")
    args = parser.parse_args()

    if args.summary:
        show_summary()
    elif args.show:
        show_anomalies(args.show.upper(), n=args.rows)
    elif args.ticker:
        results = run([args.ticker.upper()])
        print(f"\nDone: {results}")
    else:
        logger.info("Running anomaly detection for all tickers...")
        results = run()
        total = sum(results.values())
        logger.info(f"Complete. Total anomalies flagged: {total}")
        for ticker, n in results.items():
            print(f"  {ticker}: {n} anomalies")