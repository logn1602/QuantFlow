"""
indicators.py
-------------
Phase 2 — Technical Indicator Engine

Reads raw OHLCV data from raw_prices, computes:
  - RSI (14)         : Relative Strength Index — momentum indicator
  - MACD             : Moving Average Convergence Divergence — trend indicator
  - MACD Signal      : 9-day EMA of MACD
  - MACD Histogram   : MACD - Signal (shows momentum shifts)
  - Bollinger Upper  : 20-day SMA + 2 std devs (population, ddof 0)
  - Bollinger Middle : 20-day SMA
  - Bollinger Lower  : 20-day SMA - 2 std devs

Writes results to technical_indicators table.

Usage:
    python indicators.py                  # compute for all tickers
    python indicators.py --ticker AAPL    # compute for one ticker
    python indicators.py --show AAPL      # print latest indicators for a ticker
"""

import argparse

import numpy as np
import pandas as pd
import ta
from sqlalchemy import text

from quantflow.config import TICKERS
from quantflow.db.batch import frame_to_rows
from quantflow.db.connection import get_engine
from quantflow.db.indicators import replace_indicators
from quantflow.db.prices import load_daily_ohlcv as load_prices
from quantflow.utils.logger import get_logger

logger = get_logger("indicators")

# Stored precision per column; matches the NUMERIC scales in sql/schema.sql.
_ROUNDING = {
    "rsi_14": 4,
    "macd": 6,
    "macd_signal": 6,
    "macd_hist": 6,
    "bb_upper": 4,
    "bb_middle": 4,
    "bb_lower": 4,
}

# Bollinger bands: the mean of the last BB_WINDOW closes, plus and minus
# BB_DEVIATIONS standard deviations.
BB_WINDOW = 20
BB_DEVIATIONS = 2


# ── Data loading ─────────────────────────────────────────────────────────────


# ── Indicator computation ─────────────────────────────────────────────────────


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RSI, MACD, and Bollinger Bands on a price DataFrame.
    Requires at least 26 rows (MACD window) to produce meaningful values.
    Returns DataFrame with indicator columns added.
    """
    if len(df) < 26:
        logger.warning(
            f"Not enough rows ({len(df)}) to compute indicators. Need at least 26."
        )
        return pd.DataFrame()

    result = df.copy()

    # ── RSI (14-period) ──────────────────────────────────────────────────────
    # > 70 = overbought (potential sell signal)
    # < 30 = oversold  (potential buy signal)
    result["rsi_14"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()

    # ── MACD (12, 26, 9) ────────────────────────────────────────────────────
    # MACD crossing above signal = bullish
    # MACD crossing below signal = bearish
    macd = ta.trend.MACD(
        close=df["close"],
        window_slow=26,
        window_fast=12,
        window_sign=9,
    )
    result["macd"] = macd.macd()
    result["macd_signal"] = macd.macd_signal()
    result["macd_hist"] = macd.macd_diff()

    # ── Bollinger Bands (20-period, 2 std devs) ─────────────────────────────
    # Price near upper band = overbought
    # Price near lower band = oversold
    bb = ta.volatility.BollingerBands(
        close=df["close"], window=BB_WINDOW, window_dev=BB_DEVIATIONS
    )
    result["bb_upper"] = bb.bollinger_hband()
    result["bb_middle"] = bb.bollinger_mavg()
    result["bb_lower"] = bb.bollinger_lband()

    # Drop rows where all indicators are NaN (warm-up period)
    indicator_cols = ["rsi_14", "macd", "bb_upper"]
    result = result.dropna(subset=indicator_cols)

    return result


# ── Database write ────────────────────────────────────────────────────────────


def rounded_for_storage(df: pd.DataFrame) -> pd.DataFrame:
    """The indicator columns at the precision technical_indicators keeps,
    which is what every reader of the table gets back."""
    return df[list(_ROUNDING)].astype(float).round(_ROUNDING)


def stored_bollinger_bands(closes) -> dict[str, float]:
    """The newest close's Bollinger bands as compute_indicators computes
    them and rounded_for_storage rounds them, from the closes alone: ta
    uses the population standard deviation (ddof 0) of the last BB_WINDOW
    closes. For the boosters' rollout, which must rebuild the features
    training saw from closes it predicted. Missing with fewer closes."""
    window = np.asarray(closes, dtype=float)[-BB_WINDOW:]
    if len(window) < BB_WINDOW:
        return dict.fromkeys(("bb_upper", "bb_middle", "bb_lower"), np.nan)
    middle = float(window.mean())
    deviation = float(window.std(ddof=0))
    bands = {
        "bb_upper": middle + BB_DEVIATIONS * deviation,
        "bb_middle": middle,
        "bb_lower": middle - BB_DEVIATIONS * deviation,
    }
    return {
        name: float(np.round(value, _ROUNDING[name])) for name, value in bands.items()
    }


def save_indicators(df: pd.DataFrame, ticker: str) -> int:
    """
    Replace the ticker's rows in technical_indicators with df, which is
    indexed by each trading date's 00:00 UTC timestamp.
    Returns rows written. A write error raises and leaves the previous rows
    in place; see db.indicators.replace_indicators.
    """
    if df.empty:
        return 0

    columns = list(_ROUNDING)
    out = rounded_for_storage(df)
    out.insert(0, "ts", out.index)
    out.insert(0, "ticker", ticker)
    rows = frame_to_rows(out.reset_index(drop=True), ["ticker", "ts", *columns])
    return replace_indicators(ticker, rows)


# ── Main pipeline ─────────────────────────────────────────────────────────────


def run(tickers: list[str] | None = None) -> dict:
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
        print(
            f"No indicators found for {ticker}. Run: python indicators.py --ticker {ticker}"
        )
        return

    print(f"\n{'=' * 80}")
    print(f"  Latest {n} indicator rows for {ticker}")
    print(f"{'=' * 80}")
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
    parser.add_argument(
        "--ticker", type=str, help="Single ticker to compute (e.g. AAPL)"
    )
    parser.add_argument("--show", type=str, help="Print latest indicators for a ticker")
    parser.add_argument(
        "--rows", type=int, default=10, help="Number of rows to show (default: 10)"
    )
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
