-- ============================================================
-- Add this to your existing schema by running:
-- psql -U postgres -d stock_pipeline -f db/schema_sentiment.sql
-- ============================================================

CREATE TABLE IF NOT EXISTS news_sentiment (
    id            BIGSERIAL PRIMARY KEY,
    ticker        VARCHAR(10)     NOT NULL,
    source        VARCHAR(30)     NOT NULL,   -- 'newsapi' or 'rss'
    published_at  TIMESTAMPTZ     NOT NULL,
    headline      TEXT            NOT NULL,
    url           TEXT,
    sentiment     VARCHAR(10)     NOT NULL,   -- 'positive', 'negative', 'neutral'
    score_pos     NUMERIC(6, 4),              -- FinBERT positive probability
    score_neg     NUMERIC(6, 4),              -- FinBERT negative probability
    score_neu     NUMERIC(6, 4),              -- FinBERT neutral probability
    compound      NUMERIC(6, 4),              -- pos - neg (composite signal)
    inserted_at   TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    UNIQUE (ticker, headline, published_at)
);

CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_ts
    ON news_sentiment (ticker, published_at DESC);

CREATE INDEX IF NOT EXISTS idx_sentiment_compound
    ON news_sentiment (ticker, compound, published_at DESC);