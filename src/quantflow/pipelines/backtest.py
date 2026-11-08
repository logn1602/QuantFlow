"""
pipelines/backtest.py
---------------------
Runs the strategy backtest for each ticker: builds the ensemble's holdout
stack, simulates the long/flat strategy on its out-of-fold predictions
(quantflow.evaluation.backtest), stores the result for the dashboard and logs
it to MLflow.

It lives here rather than in evaluation because it needs the ensemble, and
evaluation never imports models.

Usage:
    python -m quantflow.pipelines.backtest                # run all tickers
    python -m quantflow.pipelines.backtest --ticker AAPL  # single ticker
    python -m quantflow.pipelines.backtest --show         # print saved results
"""

import argparse
import json
import sys
from datetime import datetime

import pandas as pd

from quantflow.config import TICKERS
from quantflow.db.backtest import load_backtest_results, replace_backtest_result
from quantflow.db.prices import load_daily_close as load_prices
from quantflow.evaluation.backtest import (
    INITIAL_CAPITAL,
    backtest_out_of_fold,
    log_to_mlflow,
)
from quantflow.features.engineering import load_news
from quantflow.models import ensemble
from quantflow.utils.logger import get_logger

logger = get_logger("backtest")


def _result_row(ticker: str, metrics: dict, sim: dict, eval_days: int) -> dict:
    """One backtest_results row. holdout_days stores the true evaluation
    length (the out-of-fold window), not the full 30-day holdout."""
    return {
        "ticker": ticker,
        "run_at": datetime.now(),
        "holdout_days": eval_days,
        "initial_capital": INITIAL_CAPITAL,
        "final_value": metrics["final_value"],
        "total_return": metrics["total_return"],
        "annualized_return": metrics["annualised_return"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "max_drawdown": metrics["max_drawdown"],
        "win_rate": metrics["win_rate"],
        "num_trades": metrics["num_trades"],
        "benchmark_return": metrics["benchmark_return"],
        "alpha": metrics["alpha"],
        "daily_values": json.dumps(sim["daily_values"]),
        "benchmark_values": json.dumps(sim["benchmark_values"]),
    }


def run_backtest(ticker: str, news: pd.DataFrame | None = None) -> dict:
    """
    End-to-end backtest for one ticker:
      1. Collect 30-day holdout predictions from all 4 base models
      2. Generate out-of-fold ensemble predictions (expanding-window NNLS)
      3. Simulate long/flat trading strategy on those predictions only
      4. Compute + save metrics

    The out-of-fold loop runs once. It used to run twice: tune_and_train_meta
    was called first for its log lines, repeating the whole loop, and its
    results were thrown away. The ensemble step logs the weights already.
    """
    logger.info(f"{'=' * 50}")
    logger.info(f"Backtesting: {ticker}")
    logger.info(f"{'=' * 50}")

    logger.info("Collecting holdout predictions from 4 base models...")
    stacked_df = ensemble.collect_holdout_stacks(ticker, news)
    if stacked_df.empty:
        logger.error(f"{ticker}: backtest skipped: insufficient data")
        return {}

    logger.info("Out-of-fold ensemble predictions (NNLS)...")
    result = backtest_out_of_fold(
        stacked_df,
        load_prices(ticker)["y"].to_numpy(),
        ensemble.out_of_fold_meta_predictions,
        ticker,
    )
    if result is None:
        return {}

    metrics = result["metrics"]
    logger.info(
        f"  Return: {metrics['total_return']:+.2f}% | "
        f"Benchmark: {metrics['benchmark_return']:+.2f}% | "
        f"Alpha: {metrics['alpha']:+.2f}%"
    )
    logger.info(
        f"  Sharpe: {metrics['sharpe_ratio']:.3f} | "
        f"Max DD: {metrics['max_drawdown']:.2f}% | "
        f"Win rate: {metrics['win_rate']:.1f}% | "
        f"Trades: {metrics['num_trades']}"
    )

    replace_backtest_result(
        _result_row(ticker, metrics, result["sim"], result["eval_days"])
    )
    log_to_mlflow(ticker, metrics, result["eval_days"])
    return metrics


def run(tickers: list | None = None, news: pd.DataFrame | None = None) -> dict:
    """news is load_news' frame; read here once for every ticker. If it
    cannot be read, every ticker fails, as it did when each read its own."""
    tickers = tickers or TICKERS
    all_results = {}
    if news is None:
        try:
            news = load_news()
        except Exception as e:
            logger.exception(f"Reading the news failed: {type(e).__name__}: {e}")
            return {ticker: {} for ticker in tickers}
    for ticker in tickers:
        try:
            all_results[ticker] = run_backtest(ticker, news)
        except Exception as e:
            logger.exception(f"{ticker}: backtest failed: {type(e).__name__}: {e}")
            all_results[ticker] = {}
    return all_results


def show_results():
    df = load_backtest_results()
    if df.empty:
        print("No backtest results found. Run: python -m quantflow.pipelines.backtest")
        return

    # Window length comes from the stored rows, which hold the true
    # out-of-fold evaluation length rather than the full holdout.
    windows = sorted({int(d) for d in df["holdout_days"].dropna()})
    window = f"{windows[0]}-day" if len(windows) == 1 else "mixed-length"

    print(f"\n{'=' * 90}")
    print(
        f"  QuantFlow Backtest Results "
        f"(Ensemble Long/Flat Strategy, {window} out-of-fold window)"
    )
    print(f"{'=' * 90}")

    display = df[
        [
            "ticker",
            "total_return",
            "annualized_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "num_trades",
            "benchmark_return",
            "alpha",
        ]
    ].copy()
    display.columns = [
        "Ticker",
        "Return%",
        "Ann.Return%",
        "Sharpe",
        "MaxDD%",
        "WinRate%",
        "Trades",
        "Benchmark%",
        "Alpha%",
    ]
    for col in ["Return%", "Ann.Return%", "MaxDD%", "Benchmark%", "Alpha%"]:
        display[col] = display[col].map(lambda x: f"{x:+.2f}%")
    display["Sharpe"] = display["Sharpe"].map(lambda x: f"{x:.3f}")
    display["WinRate%"] = display["WinRate%"].map(lambda x: f"{x:.1f}%")

    print(display.to_string(index=False))
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backtest the stacking ensemble strategy"
    )
    parser.add_argument("--ticker", type=str, help="Single ticker (e.g. AAPL)")
    parser.add_argument("--show", action="store_true", help="Print saved results")
    args = parser.parse_args(argv)

    if args.show:
        show_results()
        return 0

    tickers = [args.ticker.upper()] if args.ticker else None
    logger.info("Starting backtest pipeline...")
    results = run(tickers=tickers)

    print("\n--- Backtest Results ---")
    for ticker, m in results.items():
        if m:
            print(
                f"  {ticker}: return={m['total_return']:+.2f}% | "
                f"alpha={m['alpha']:+.2f}% | sharpe={m['sharpe_ratio']:.3f} | "
                f"win={m['win_rate']:.1f}%"
            )

    # Exit rule: fail only when EVERY ticker failed. A single ticker with
    # thin history should not red-flag the whole daily refresh, but a total
    # failure must not be reported as success.
    succeeded = [t for t, m in results.items() if m]
    empty = [t for t, m in results.items() if not m]

    if empty:
        print(f"\nWARNING: no results for {', '.join(empty)}", file=sys.stderr)

    if results and not succeeded:
        print("\n--- BACKTEST FAILED ---", file=sys.stderr)
        print(
            f"All {len(results)} ticker(s) failed; see the log above.",
            file=sys.stderr,
        )
        return 1

    print("\nTo see full table:  python -m quantflow.pipelines.backtest --show")
    return 0


if __name__ == "__main__":
    sys.exit(main())
