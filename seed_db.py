"""
seed_db.py
----------
Run this ONCE to seed the database with 2 years of historical daily prices.
After seeding, use the scheduler for ongoing intraday updates.

Usage:
    python seed_db.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from db.connection import test_connection
from ingestion.yfinance_fetcher import fetch_historical
from ingestion.alpha_vantage_fetcher import fetch_daily
from config import TICKERS
from utils.logger import get_logger

logger = get_logger("seed_db")


def main():
    logger.info("Starting database seed...")
    logger.info(f"Tickers: {TICKERS}")

    if not test_connection():
        logger.error("Database not reachable. Check your .env file.")
        sys.exit(1)

    # --- yFinance: 2 years of daily data (free, no limits) ---
    logger.info("=== Seeding from yFinance (2 years daily) ===")
    yf_results = fetch_historical(TICKERS, period="2y")
    for ticker, n in yf_results.items():
        logger.info(f"  {ticker}: {n} rows")

    # --- Alpha Vantage: last 100 trading days ---
    logger.info("=== Seeding from Alpha Vantage (last 100 days daily) ===")
    logger.info("Note: Free tier is 25 req/day — this will take a few minutes...")
    av_results = fetch_daily(TICKERS)
    for ticker, n in av_results.items():
        logger.info(f"  {ticker}: {n} rows")

    total_yf = sum(yf_results.values())
    total_av = sum(av_results.values())
    logger.info(f"Seed complete. yFinance: {total_yf} rows | Alpha Vantage: {total_av} rows")
    logger.info("You can now run: python scheduler/job_runner.py")


if __name__ == "__main__":
    main()
