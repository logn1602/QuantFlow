"""
indicators.py
-------------
Phase 2 — Technical Indicator Engine

Reads raw OHLCV data from raw_prices, computes:
  - RSI (14)         : Relative Strength Index — momentum indicator
  - MACD             : Moving Average Convergence Divergence — trend indicator
  - MACD Signal      : 9-day EMA of MACD
  - MACD Histogram   : MACD - Signal (shows momentum shifts)
  - Bollinger Upper  : 20-day SMA + 2 std devs
  - Bollinger Middle : 20-day SMA
  - Bollinger Lower  : 20-day SMA - 2 std devs

Writes results to technical_indicators table.

Usage:
    python indicators.py                  # compute for all tickers
    python indicators.py --ticker AAPL    # compute for one ticker
    python indicators.py --show AAPL      # print latest indicators for a ticker
"""

import sys
import os
import argparse

import pandas as pd
import ta
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(__file__))
from config import TICKERS
from db.connection import get_engine
from utils.logger import get_logger

logger = get_logger("indicators")


# ── Data loading ─────────────────────────────────────────────────────────────

def load_prices(ticker: str, source: str = "yfinance") -> pd.DataFrame:
    """
    Load raw OHLCV data for a ticker from the database.
    Uses yfinance by default (more history = better indicators).
    Returns DataFrame sorted oldest → newest.
    """
    engine = get_engine()
    query = text("""
        SELECT ts, open, high, low, close, volume
        FROM raw_prices
        WHERE ticker = :ticker
          AND source = :source
        ORDER BY ts ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker, "source": source})

    if df.empty:
        logger.warning(f"No price data found for {ticker} ({source})")
        return df

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df


# ── Indicator computation ─────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RSI, MACD, and Bollinger Bands on a price DataFrame.
    Requires at least 26 rows (MACD window) to produce meaningful values.
    Returns DataFrame with indicator columns added.
    """
    if len(df) < 26:
        logger.warning(f"Not enough rows ({len(df)}) to compute indicators. Need at least 26.")
        return pd.DataFrame()

    result = df.copy()

    # ── RSI (14-period) ──────────────────────────────────────────────────────
    # > 70 = overbought (potential sell signal)
    # < 30 = oversold  (potential buy signal)
    result["rsi_14"] = ta.momentum.RSIIndicator(
        close=df["close"], window=14
    ).rsi()

    # ── MACD (12, 26, 9) ────────────────────────────────────────────────────
    # MACD crossing above signal = bullish
    # MACD crossing below signal = bearish
    macd = ta.trend.MACD(
        close=df["close"],
        window_slow=26,
        window_fast=12,
        window_sign=9,
    )
    result["macd"]        = macd.macd()
    result["macd_signal"] = macd.macd_signal()
    result["macd_hist"]   = macd.macd_diff()

    # ── Bollinger Bands (20-period, 2 std devs) ─────────────────────────────
    # Price near upper band = overbought
    # Price near lower band = oversold
    bb = ta.volatility.BollingerBands(
        close=df["close"], window=20, window_dev=2
    )
    result["bb_upper"]  = bb.bollinger_hband()
    result["bb_middle"] = bb.bollinger_mavg()
    result["bb_lower"]  = bb.bollinger_lband()

    # Drop rows where all indicators are NaN (warm-up period)
    indicator_cols = ["rsi_14", "macd", "bb_upper"]
    result = result.dropna(subset=indicator_cols)

    return result


# ── Database write ────────────────────────────────────────────────────────────

def save_indicators(df: pd.DataFrame, ticker: str) -> int:
    """
    Upsert indicator rows into technical_indicators table.
    Skips duplicates (same ticker + ts).
    Returns number of rows inserted.
    """
    if df.empty:
        return 0

    engine = get_engine()
    inserted = 0

    cols = ["rsi_14", "macd", "macd_signal", "macd_hist", "bb_upper", "bb_middle", "bb_lower"]

    with engine.begin() as conn:
        for ts, row in df.iterrows():
            try:
                conn.execute(
                    text("""
                        INSERT INTO technical_indicators
                            (ticker, ts, rsi_14, macd, macd_signal, macd_hist,
                             bb_upper, bb_middle, bb_lower)
                        VALUES
                            (:ticker, :ts, :rsi_14, :macd, :macd_signal, :macd_hist,
                             :bb_upper, :bb_middle, :bb_lower)
                        ON CONFLICT (ticker, ts) DO UPDATE SET
                            rsi_14      = EXCLUDED.rsi_14,
                            macd        = EXCLUDED.macd,
                            macd_signal = EXCLUDED.macd_signal,
                            macd_hist   = EXCLUDED.macd_hist,
                            bb_upper    = EXCLUDED.bb_upper,
                            bb_middle   = EXCLUDED.bb_middle,
                            bb_lower    = EXCLUDED.bb_lower
                    """),
                    {
                        "ticker":      ticker,
                        "ts":          ts,
                        "rsi_14":      round(float(row["rsi_14"]),   4) if pd.notna(row["rsi_14"])   else None,
                        "macd":        round(float(row["macd"]),      6) if pd.notna(row["macd"])      else None,
                        "macd_signal": round(float(row["macd_signal"]),6) if pd.notna(row["macd_signal"]) else None,
                        "macd_hist":   round(float(row["macd_hist"]), 6) if pd.notna(row["macd_hist"]) else None,
                        "bb_upper":    round(float(row["bb_upper"]),  4) if pd.notna(row["bb_upper"])  else None,
                        "bb_middle":   round(float(row["bb_middle"]), 4) if pd.notna(row["bb_middle"]) else None,
                        "bb_lower":    round(float(row["bb_lower"]),  4) if pd.notna(row["bb_lower"])  else None,
                    }
                )
                inserted += 1
            except Exception as e:
                logger.warning(f"Row skipped for {ticker} @ {ts}: {e}")

    return inserted


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(tickers: list[str] = None) -> dict:
    """
    Full pipeline: load → compute → save for each ticker.
    Returns dict of {ticker: rows_inserted}.
    """
    tickers = tickers or TICKERS
    results = {}

    for ticker in tickers:
        logger.info(f"Computing indicators for {ticker}...")

        df = load_prices(ticker)
        if df.empty:
            results[ticker] = 0
            continue

        indicators = compute_indicators(df)
        if indicators.empty:
            results[ticker] = 0
            continue

        n = save_indicators(indicators, ticker)
        results[ticker] = n
        logger.info(f"  {ticker}: {n} indicator rows saved")

    return results


def show_latest(ticker: str, n: int = 10):
    """Print the latest n indicator rows for a ticker."""
    engine = get_engine()
    query = text("""
        SELECT
            DATE(ts)       AS date,
            ROUND(rsi_14::numeric, 2)      AS rsi,
            ROUND(macd::numeric, 4)        AS macd,
            ROUND(macd_signal::numeric, 4) AS signal,
            ROUND(macd_hist::numeric, 4)   AS histogram,
            ROUND(bb_upper::numeric, 2)    AS bb_upper,
            ROUND(bb_middle::numeric, 2)   AS bb_mid,
            ROUND(bb_lower::numeric, 2)    AS bb_lower
        FROM technical_indicators
        WHERE ticker = :ticker
        ORDER BY ts DESC
        LIMIT :n
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker, "n": n})

    if df.empty:
        print(f"No indicators found for {ticker}. Run: python indicators.py --ticker {ticker}")
        return

    print(f"\n{'='*80}")
    print(f"  Latest {n} indicator rows for {ticker}")
    print(f"{'='*80}")
    print(df.to_string(index=False))

    # Quick signal interpretation
    latest = df.iloc[0]
    print(f"\n--- Signal summary for {ticker} (latest row) ---")
    rsi = latest["rsi"]
    if rsi > 70:
        print(f"  RSI {rsi} → OVERBOUGHT (potential pullback)")
    elif rsi < 30:
        print(f"  RSI {rsi} → OVERSOLD (potential bounce)")
    else:
        print(f"  RSI {rsi} → Neutral")

    hist = latest["histogram"]
    if hist > 0:
        print(f"  MACD Histogram {hist} → Bullish momentum")
    else:
        print(f"  MACD Histogram {hist} → Bearish momentum")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute technical indicators")
    parser.add_argument("--ticker", type=str, help="Single ticker to compute (e.g. AAPL)")
    parser.add_argument("--show",   type=str, help="Print latest indicators for a ticker")
    parser.add_argument("--rows",   type=int, default=10, help="Number of rows to show (default: 10)")
    args = parser.parse_args()

    if args.show:
        show_latest(args.show.upper(), n=args.rows)
    elif args.ticker:
        results = run([args.ticker.upper()])
        print(f"\nDone: {results}")
    else:
        logger.info("Running indicators for all tickers...")
        results = run()
        total = sum(results.values())
        logger.info(f"Complete. Total rows saved: {total}")
        for ticker, n in results.items():
            print(f"  {ticker}: {n} rows")