-- ============================================================
-- schema_metrics.sql
-- Per-run model evaluation metrics: each daily run's 30-day holdout
-- figures, kept in Postgres because MLflow's local sqlite file never
-- leaves the machine that ran it. Nothing displays them now; the
-- dashboard shows the rolling-origin evaluation's published 7-day error.
-- Command: psql -U postgres -d stock_pipeline -f sql/schema_metrics.sql
-- ============================================================

-- Holdout metrics, one row per model run
CREATE TABLE IF NOT EXISTS model_metrics (
    id           BIGSERIAL PRIMARY KEY,
    ticker       VARCHAR(10)  NOT NULL,
    model        VARCHAR(30)  NOT NULL,   -- 'arima', 'prophet', 'xgboost', 'lightgbm', 'ensemble_stack'
    holdout_days INTEGER      NOT NULL,   -- length of the evaluation window
    rmse         NUMERIC(14, 6),
    mae          NUMERIC(14, 6),
    mape         NUMERIC(10, 4),
    run_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    UNIQUE (ticker, model, run_at)
);

-- Index for "latest metrics per model" lookups
CREATE INDEX IF NOT EXISTS idx_model_metrics_ticker_model_run
    ON model_metrics (ticker, model, run_at DESC);
