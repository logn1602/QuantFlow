-- ============================================================
-- schema_backtest.sql
-- Backtest results for the stacking ensemble trading strategy.
-- Command: psql -U postgres -d stock_pipeline -f db/schema_backtest.sql
-- ============================================================

CREATE TABLE IF NOT EXISTS backtest_results (
    id                  BIGSERIAL PRIMARY KEY,
    ticker              VARCHAR(10)   NOT NULL,
    run_at              TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    holdout_days        INT           NOT NULL,
    initial_capital     FLOAT         NOT NULL DEFAULT 10000,

    -- Strategy performance
    final_value         FLOAT,
    total_return        FLOAT,          -- %
    annualized_return   FLOAT,          -- %
    sharpe_ratio        FLOAT,
    max_drawdown        FLOAT,          -- % (negative value)
    win_rate            FLOAT,          -- % correct direction calls on signal days
    num_trades          INT,

    -- Benchmark (buy-and-hold over same period)
    benchmark_return    FLOAT,          -- %
    alpha               FLOAT,          -- strategy annualized return - benchmark %

    -- Daily series stored as JSON for chart rendering
    daily_values        TEXT,           -- JSON array of portfolio values
    benchmark_values    TEXT            -- JSON array of buy-and-hold values
);

CREATE INDEX IF NOT EXISTS idx_backtest_ticker
    ON backtest_results (ticker, run_at DESC);
