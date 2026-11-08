"""Persistence for strategy backtest results (the backtest_results table)."""

import pandas as pd
from sqlalchemy import text

from quantflow.db.connection import get_engine

_DELETE_FOR_TICKER = text("DELETE FROM backtest_results WHERE ticker = :t")

_INSERT = text("""
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
""")

_RESULTS = text("""
    SELECT ticker, total_return, annualized_return, sharpe_ratio,
           max_drawdown, win_rate, num_trades, benchmark_return, alpha,
           final_value, holdout_days, run_at::date AS run_date
    FROM backtest_results
    ORDER BY alpha DESC
""")


def replace_backtest_result(row: dict) -> None:
    """Replace the ticker's stored result with row, in one transaction, so
    the table never holds a ticker's old and new results at once."""
    with get_engine().begin() as conn:
        conn.execute(_DELETE_FOR_TICKER, {"t": row["ticker"]})
        conn.execute(_INSERT, row)


def load_backtest_results() -> pd.DataFrame:
    """Every ticker's stored result, best alpha first."""
    with get_engine().connect() as conn:
        return pd.read_sql(_RESULTS, conn)
