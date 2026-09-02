-- ============================================================
-- schema_metrics.sql
-- Per-run model evaluation metrics.
-- MLflow writes to a local sqlite file that Streamlit Cloud cannot
-- read, so anything the dashboard must display lives here instead.
-- Command: psql -U postgres -d stock_pipeline -f db/schema_metrics.sql
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

-- Index for "latest metrics per model" lookups from the dashboard
CREATE INDEX IF NOT EXISTS idx_model_metrics_ticker_model_run
    ON model_metrics (ticker, model, run_at DESC);
