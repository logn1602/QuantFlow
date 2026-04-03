-- ============================================================
-- schema.sql
-- Run once to set up your database tables.
-- Command: psql -U postgres -d stock_pipeline -f db/schema.sql
-- ============================================================

-- Raw OHLCV prices from any source
CREATE TABLE IF NOT EXISTS raw_prices (
    id          BIGSERIAL PRIMARY KEY,
    ticker      VARCHAR(10)     NOT NULL,
    source      VARCHAR(20)     NOT NULL,       -- 'yfinance' or 'alpha_vantage'
    ts          TIMESTAMPTZ     NOT NULL,        -- candle timestamp (market time)
    open        NUMERIC(12, 4),
    high        NUMERIC(12, 4),
    close       NUMERIC(12, 4),
    low         NUMERIC(12, 4),
    volume      BIGINT,
    inserted_at TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    -- Prevent duplicate rows for same ticker + source + candle time
    UNIQUE (ticker, source, ts)
);

-- Index for fast time-range queries per ticker
CREATE INDEX IF NOT EXISTS idx_raw_prices_ticker_ts
    ON raw_prices (ticker, ts DESC);


-- Technical indicators (computed in Phase 3)
CREATE TABLE IF NOT EXISTS technical_indicators (
    id          BIGSERIAL PRIMARY KEY,
    ticker      VARCHAR(10)     NOT NULL,
    ts          TIMESTAMPTZ     NOT NULL,
    rsi_14      NUMERIC(8, 4),
    macd        NUMERIC(12, 6),
    macd_signal NUMERIC(12, 6),
    macd_hist   NUMERIC(12, 6),
    bb_upper    NUMERIC(12, 4),
    bb_middle   NUMERIC(12, 4),
    bb_lower    NUMERIC(12, 4),
    inserted_at TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    UNIQUE (ticker, ts)
);

CREATE INDEX IF NOT EXISTS idx_indicators_ticker_ts
    ON technical_indicators (ticker, ts DESC);


-- Anomaly flags (computed in Phase 3)
CREATE TABLE IF NOT EXISTS anomalies (
    id          BIGSERIAL PRIMARY KEY,
    ticker      VARCHAR(10)     NOT NULL,
    ts          TIMESTAMPTZ     NOT NULL,
    close       NUMERIC(12, 4),
    zscore      NUMERIC(8, 4),
    flag        VARCHAR(10)     NOT NULL,        -- 'HIGH' or 'LOW'
    inserted_at TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_anomalies_ticker_ts
    ON anomalies (ticker, ts DESC);


-- Forecasts (computed in Phase 4)
CREATE TABLE IF NOT EXISTS forecasts (
    id             BIGSERIAL PRIMARY KEY,
    ticker         VARCHAR(10)    NOT NULL,
    model          VARCHAR(30)    NOT NULL,       -- 'arima' or 'prophet'
    forecast_date  DATE           NOT NULL,       -- the date being predicted
    predicted_close NUMERIC(12, 4),
    lower_bound    NUMERIC(12, 4),
    upper_bound    NUMERIC(12, 4),
    run_at         TIMESTAMPTZ    NOT NULL DEFAULT NOW(),

    UNIQUE (ticker, model, forecast_date, run_at)
);


-- Alert log (Phase 5)
CREATE TABLE IF NOT EXISTS alerts_log (
    id          BIGSERIAL PRIMARY KEY,
    ticker      VARCHAR(10)     NOT NULL,
    alert_type  VARCHAR(30)     NOT NULL,        -- 'PRICE_THRESHOLD', 'ANOMALY' etc.
    message     TEXT,
    sent_at     TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);
