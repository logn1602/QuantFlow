"""
backtest.py
-----------
Phase 7 — Strategy Backtesting

Simulates a long/flat trading strategy driven by the stacking ensemble's
price direction signals.

The simulation runs only on OUT-OF-FOLD ensemble predictions: each day is
predicted by a meta-learner fitted solely on days before it. The first
META_WARMUP_DAYS of the 30-day holdout are consumed fitting that first
meta-learner, so the evaluation window is shorter than the holdout (~20 days)
and annualised figures — which scale by 252/n — are correspondingly noisier.
That is the honest trade: an in-sample simulation carried look-ahead bias.

Strategy:
  - Signal: if ensemble_pred[t] > close[t-1]  →  BUY at close[t-1]
  - Exit:   at close[t]  (1-day holding period)
  - Otherwise: hold cash  (no shorting, no leverage)
  - Transaction cost: 0.1% per round-trip trade

Metrics reported vs buy-and-hold benchmark:
  - Total return %
  - Annualized return %
  - Sharpe ratio  (risk-free rate = 4.5%, annualised)
  - Max drawdown %
  - Win rate %    (correct direction calls on signal days)
  - Alpha         (strategy annualised return - benchmark annualised return)
  - Number of trades

Results are saved to backtest_results table for dashboard rendering.

Usage:
    python backtest.py                  # run all tickers
    python backtest.py --ticker AAPL    # single ticker
    python backtest.py --show           # print saved results
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

# The imports below sit after the sys.path bootstrap above and therefore trip
# E402. Each carries a targeted suppression rather than a config-wide ignore.
# The src/ layout removes the bootstrap, and these all come out with it.
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402

from config import TICKERS  # noqa: E402
from db.connection import get_engine  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger("backtest")

HOLDOUT_DAYS = 30
INITIAL_CAPITAL = 10_000.0
TRANSACTION_COST = 0.001  # 0.1% per trade
RISK_FREE_RATE = 0.045  # 4.5% annualised (US T-bill, 2024–2026)  # noqa: RUF003
ZERO_VARIANCE_TOL = 1e-12  # below this, treat return variance as zero
MLFLOW_EXP = "stock_forecasting"


# ── Core simulation ───────────────────────────────────────────────────────────


def simulate_strategy(
    prev_prices: np.ndarray, actual_prices: np.ndarray, ensemble_preds: np.ndarray
) -> dict:
    """
    Long/flat strategy: enter long when ensemble predicts price rise,
    hold cash otherwise. 1-day holding period, 0.1% transaction cost.

    prev_prices[i]    = close on day i-1  (used for signal generation)
    actual_prices[i]  = close on day i    (trade settlement)
    ensemble_preds[i] = ensemble forecast for close on day i
    """
    # Binary signals: 1 = go long, 0 = stay flat
    signals = (ensemble_preds > prev_prices).astype(int)

    # Daily returns for the period
    daily_returns = (actual_prices - prev_prices) / prev_prices

    # Strategy daily returns: earn market return when long, 0 when flat
    # Transaction cost applied on signal changes (entry + exit)
    signal_diff = np.diff(np.concatenate([[0], signals]))
    trade_entries = (signal_diff > 0).astype(float)  # days we enter
    strategy_returns = signals * daily_returns - trade_entries * TRANSACTION_COST

    # Benchmark: buy-and-hold over the same period
    benchmark_returns = daily_returns

    # Portfolio value series (starting from INITIAL_CAPITAL)
    strategy_curve = INITIAL_CAPITAL * np.cumprod(1 + strategy_returns)
    benchmark_curve = INITIAL_CAPITAL * np.cumprod(1 + benchmark_returns)

    daily_values = np.concatenate([[INITIAL_CAPITAL], strategy_curve]).tolist()
    benchmark_values = np.concatenate([[INITIAL_CAPITAL], benchmark_curve]).tolist()

    # Win rate: of signal=1 days, how many had actual price rise
    long_days = signals == 1
    win_rate = (
        float((daily_returns[long_days] > 0).mean() * 100)
        if long_days.sum() > 0
        else 0.0
    )

    # Number of trades = number of entries (0→1 transitions)
    num_trades = int(trade_entries.sum())

    return {
        "strategy_returns": strategy_returns,
        "benchmark_returns": benchmark_returns,
        "daily_values": daily_values,
        "benchmark_values": benchmark_values,
        "signals": signals.tolist(),
        "win_rate": win_rate,
        "num_trades": num_trades,
    }


def compute_metrics(sim: dict) -> dict:
    """Derive performance metrics from simulation output."""
    sr = np.array(sim["strategy_returns"])
    dv = np.array(sim["daily_values"])
    bv = np.array(sim["benchmark_values"])
    n = len(sr)

    # Returns
    total_return = float((dv[-1] / INITIAL_CAPITAL - 1) * 100)
    annualised_return = float(((dv[-1] / INITIAL_CAPITAL) ** (252 / n) - 1) * 100)
    bench_total = float((bv[-1] / INITIAL_CAPITAL - 1) * 100)
    bench_annual = float(((bv[-1] / INITIAL_CAPITAL) ** (252 / n) - 1) * 100)

    # Sharpe ratio (annualised)
    # A constant return series has no variance and no meaningful Sharpe. Test
    # against a tolerance rather than > 0: subtracting rf_daily from identical
    # returns can leave ~1e-19 of floating-point residue, which a bare > 0
    # check lets through and which then divides into an absurd Sharpe
    # (0.001/day yielded ~1.2e17). Real daily-return variance is many orders of
    # magnitude above ZERO_VARIANCE_TOL, so nothing genuine is suppressed.
    rf_daily = RISK_FREE_RATE / 252
    excess = sr - rf_daily
    excess_sd = float(excess.std())
    sharpe = (
        float(excess.mean() / excess_sd * np.sqrt(252))
        if excess_sd > ZERO_VARIANCE_TOL
        else 0.0
    )

    # Max drawdown
    peak = np.maximum.accumulate(dv)
    drawdown = (dv - peak) / peak
    max_dd = float(drawdown.min() * 100)

    # Alpha
    alpha = round(annualised_return - bench_annual, 2)

    return {
        "total_return": round(total_return, 2),
        "annualised_return": round(annualised_return, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(max_dd, 2),
        "win_rate": round(sim["win_rate"], 1),
        "num_trades": sim["num_trades"],
        "benchmark_return": round(bench_total, 2),
        "alpha": alpha,
        "final_value": round(float(dv[-1]), 2),
    }


# ── DB operations ─────────────────────────────────────────────────────────────


def save_results(ticker: str, metrics: dict, sim: dict, eval_days: int):
    """Persist one backtest run. holdout_days stores the true evaluation
    length (the out-of-fold window), not the full 30-day holdout."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("""
            DELETE FROM backtest_results WHERE ticker = :t
        """),
            {"t": ticker},
        )

        conn.execute(
            text("""
            INSERT INTO backtest_results (
                ticker, run_at, holdout_days, initial_capital,
                final_value, total_return, annualized_return,
                sharpe_ratio, max_drawdown, win_rate, num_trades,
                benchmark_return, alpha,
                daily_values, benchmark_values
            ) VALUES (
                :ticker, :run_at, :holdout_days, :initial_capital,
                :final_value, :total_return, :annualized_return,
                :sharpe_ratio, :max_drawdown, :win_rate, :num_trades,
                :benchmark_return, :alpha,
                :daily_values, :benchmark_values
            )
        """),
            {
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
            },
        )


def load_results() -> pd.DataFrame:
    engine = get_engine()
    query = text("""
        SELECT ticker, total_return, annualized_return, sharpe_ratio,
               max_drawdown, win_rate, num_trades, benchmark_return, alpha,
               final_value, holdout_days, run_at::date AS run_date
        FROM backtest_results
        ORDER BY alpha DESC
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def load_daily_series(ticker: str) -> dict:
    engine = get_engine()
    query = text("""
        SELECT daily_values, benchmark_values
        FROM backtest_results
        WHERE ticker = :t
        ORDER BY run_at DESC LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(query, {"t": ticker}).fetchone()
    if not row:
        return {}
    return {
        "daily_values": json.loads(row[0]),
        "benchmark_values": json.loads(row[1]),
    }


# ── MLflow logging ─────────────────────────────────────────────────────────────


def log_to_mlflow(ticker: str, metrics: dict, eval_days: int):
    try:
        import mlflow

        mlflow.set_experiment(MLFLOW_EXP)
        with mlflow.start_run(run_name=f"backtest_{ticker}"):
            mlflow.log_param("ticker", ticker)
            mlflow.log_param("strategy", "long_flat_ensemble")
            mlflow.log_param("holdout_days", HOLDOUT_DAYS)
            mlflow.log_param("eval_days", eval_days)
            mlflow.log_param("transaction_cost", TRANSACTION_COST)
            mlflow.log_param("risk_free_rate", RISK_FREE_RATE)
            mlflow.log_metric("total_return", metrics["total_return"])
            mlflow.log_metric("annualised_return", metrics["annualised_return"])
            mlflow.log_metric("sharpe_ratio", metrics["sharpe_ratio"])
            mlflow.log_metric("max_drawdown", metrics["max_drawdown"])
            mlflow.log_metric("win_rate", metrics["win_rate"])
            mlflow.log_metric("benchmark_return", metrics["benchmark_return"])
            mlflow.log_metric("alpha", metrics["alpha"])
            mlflow.log_metric("num_trades", metrics["num_trades"])
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")


# ── Main backtest runner ──────────────────────────────────────────────────────


def run_backtest(ticker: str) -> dict:
    """
    End-to-end backtest for one ticker:
      1. Collect 30-day out-of-fold holdout predictions from all 4 base models
      2. Generate out-of-fold ensemble predictions (expanding-window NNLS)
      3. Simulate long/flat trading strategy on those predictions only
      4. Compute + save metrics
    """
    from ensemble import (
        HOLDOUT_DAYS as STACK_HOLDOUT_DAYS,
    )
    from ensemble import (
        collect_holdout_stacks,
        out_of_fold_meta_predictions,
        tune_and_train_meta,
    )
    from forecasting import load_prices

    logger.info(f"{'=' * 50}")
    logger.info(f"Backtesting — {ticker}")
    logger.info(f"{'=' * 50}")

    # Step 1: Holdout stacks
    logger.info("Collecting holdout predictions from 4 base models...")
    stacked_df = collect_holdout_stacks(ticker)
    if stacked_df.empty:
        logger.error(f"{ticker}: backtest skipped — insufficient data")
        return {}

    # Step 2: Out-of-fold ensemble predictions. Simulating on in-sample
    # predictions would hand the strategy look-ahead information.
    logger.info("Training NNLS meta-learner...")
    # Called for its logging side effect only: it reports the NNLS weights and
    # the out-of-fold comparison. The returned model is deliberately unused —
    # the simulation below runs on out_of_fold_meta_predictions instead.
    _meta_model, _meta_metrics = tune_and_train_meta(stacked_df)

    ensemble_preds, oof_actuals = out_of_fold_meta_predictions(stacked_df)
    eval_days = len(ensemble_preds)
    if eval_days == 0:
        logger.error(f"{ticker}: backtest skipped — no out-of-fold days available")
        return {}

    # Step 3: Price series aligned to the out-of-fold window.
    #
    # stacked_df row i  <->  price index (len_y - STACK_HOLDOUT_DAYS + i)
    # oof index j       <->  stacked row (warmup + j)
    # so day j's close is at price index (start + j), and the previous close
    # — needed to generate day j's signal — is at (start + j - 1).
    price_df = load_prices(ticker)
    y_all = price_df["y"].values
    warmup = len(stacked_df) - eval_days
    start = len(y_all) - STACK_HOLDOUT_DAYS + warmup

    if start < 1:
        logger.error(
            f"{ticker}: backtest skipped — not enough price history "
            f"for a previous-close on the first evaluation day"
        )
        return {}

    actual_prices = y_all[start : start + eval_days]
    prev_prices = y_all[start - 1 : start - 1 + eval_days]

    assert len(prev_prices) == len(actual_prices) == len(ensemble_preds), (
        f"length mismatch: prev={len(prev_prices)} "
        f"actual={len(actual_prices)} preds={len(ensemble_preds)}"
    )
    # oof_actuals comes from the stack, actual_prices from the raw price series.
    # If they disagree the windows are misaligned by at least one day.
    assert np.allclose(actual_prices, oof_actuals), (
        "evaluation window misaligned with out-of-fold actuals"
    )

    # Step 4: Strategy simulation
    logger.info(
        f"Simulating long/flat strategy on {eval_days} out-of-fold days "
        f"({warmup} warmup days consumed)..."
    )
    sim = simulate_strategy(prev_prices, actual_prices, ensemble_preds)
    metrics = compute_metrics(sim)

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

    # Step 5: Persist + track
    save_results(ticker, metrics, sim, eval_days)
    log_to_mlflow(ticker, metrics, eval_days)

    return metrics


def run(tickers: list | None = None) -> dict:
    tickers = tickers or TICKERS
    all_results = {}
    for ticker in tickers:
        try:
            all_results[ticker] = run_backtest(ticker)
        except Exception as e:
            logger.exception(f"{ticker}: backtest failed — {type(e).__name__}: {e}")
            all_results[ticker] = {}
    return all_results


# ── Display ───────────────────────────────────────────────────────────────────


def show_results():
    df = load_results()
    if df.empty:
        print("No backtest results found. Run: python backtest.py")
        return

    # Window length comes from the stored rows, which hold the true
    # out-of-fold evaluation length rather than the full holdout.
    windows = sorted({int(d) for d in df["holdout_days"].dropna()})
    window = f"{windows[0]}-day" if len(windows) == 1 else "mixed-length"

    print(f"\n{'=' * 90}")
    print(
        f"  QuantFlow — Backtest Results "
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


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest the stacking ensemble strategy"
    )
    parser.add_argument("--ticker", type=str, help="Single ticker (e.g. AAPL)")
    parser.add_argument("--show", action="store_true", help="Print saved results")
    args = parser.parse_args()

    if args.show:
        show_results()
    else:
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
                f"All {len(results)} ticker(s) failed — see the log above.",
                file=sys.stderr,
            )
            sys.exit(1)

        print("\nTo see full table:  python backtest.py --show")
