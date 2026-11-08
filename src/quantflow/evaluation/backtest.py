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

This module holds the simulation and its metrics only. The run that builds
the holdout stack, saves the results to the backtest_results table and
offers a command line is quantflow.pipelines.backtest: it needs the
ensemble, and evaluation never imports models (tests/unit/
test_package_boundaries.py). The meta-learner's out-of-fold loop comes in as
an argument, the way out_of_fold.py takes its fitter.

Usage:
    python -m quantflow.pipelines.backtest                # run all tickers
    python -m quantflow.pipelines.backtest --ticker AAPL  # single ticker
    python -m quantflow.pipelines.backtest --show         # print saved results
"""

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

from quantflow.evaluation.splits import HOLDOUT_DAYS, train_holdout_split
from quantflow.utils.logger import get_logger

warnings.filterwarnings("ignore")

logger = get_logger("backtest")

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


# ── The out-of-fold window ────────────────────────────────────────────────────


def backtest_out_of_fold(
    stacked_df: pd.DataFrame, closes, oof_fn: Callable, ticker: str
) -> dict | None:
    """Simulate the strategy on out-of-fold ensemble predictions only.

    stacked_df is the holdout stack built from `closes`, so its rows are the
    holdout's closes. oof_fn(stacked_df) returns (predictions, actuals), each
    day predicted by a meta-learner fitted only on the days before it;
    simulating on in-sample predictions would hand the strategy look-ahead
    information.

    Returns {"metrics", "sim", "eval_days", "warmup"}, or None when no
    out-of-fold day is left or the first one has no previous close.
    """
    ensemble_preds, oof_actuals = oof_fn(stacked_df)
    eval_days = len(ensemble_preds)
    if eval_days == 0:
        logger.error(f"{ticker}: backtest skipped: no out-of-fold days available")
        return None

    # stacked row i  <->  holdout day i, at price index len(train) + i
    # oof index j    <->  stacked row (warmup + j)
    # so day j's close is at price index (start + j), and the previous close
    # (needed to generate day j's signal) is at (start + j - 1).
    y_all = np.asarray(closes, dtype=float)
    train, _ = train_holdout_split(pd.Series(y_all))
    warmup = len(stacked_df) - eval_days
    start = len(train) + warmup

    if start < 1:
        logger.error(
            f"{ticker}: backtest skipped: not enough price history "
            f"for a previous-close on the first evaluation day"
        )
        return None

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

    logger.info(
        f"Simulating long/flat strategy on {eval_days} out-of-fold days "
        f"({warmup} warmup days consumed)..."
    )
    sim = simulate_strategy(prev_prices, actual_prices, ensemble_preds)
    return {
        "metrics": compute_metrics(sim),
        "sim": sim,
        "eval_days": eval_days,
        "warmup": warmup,
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
