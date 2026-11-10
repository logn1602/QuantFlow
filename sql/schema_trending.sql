-- ============================================================
-- schema_trending.sql
-- The trending tier's tables (docs/trending.md, section 6).
-- Command: psql -U postgres -d stock_pipeline -f sql/schema_trending.sql
--
-- The same statements as sql/migrations/002_trending.sql, which is what the
-- live database gets; tests/unit/test_trending_schema.py fails if the two
-- drift apart.
--
-- Kept apart from the core's tables by construction: nothing in the core or
-- the evaluation reads these, and the trending tier writes nothing else.
-- ============================================================

-- The trending tier's symbols, and the summary its tab shows. The core
-- watchlist is never copied here: it is read from config (TICKERS).
CREATE TABLE IF NOT EXISTS tickers (
    symbol            VARCHAR(10)     PRIMARY KEY,
    tier              VARCHAR(10)     NOT NULL DEFAULT 'trending'
                                      CHECK (tier IN ('core', 'trending')),
    short_name        VARCHAR(100),
    quote_type        VARCHAR(20),
    first_seen        TIMESTAMPTZ     NOT NULL,   -- first run it was a candidate or pinned
    last_seen         TIMESTAMPTZ     NOT NULL,   -- last run it was a candidate or pinned
    active            BOOLEAN         NOT NULL DEFAULT TRUE,
    pinned            BOOLEAN         NOT NULL DEFAULT FALSE,
    first_bar_date    DATE,                       -- its first stored bar's date
    first_trade_date  DATE,                       -- Yahoo's, or a later listing date
    promoted_at       TIMESTAMPTZ,                -- set by hand only (section 8)
    -- What the Trending tab's table shows, from the last run (section 7.4).
    summary_session   DATE,
    last_close        NUMERIC(14, 4),
    rsi_14            NUMERIC(8, 2),
    recent_flags      SMALLINT,
    big_move_date     DATE
);

-- raw_prices' columns for a daily bar, apart from it on purpose: no core or
-- evaluation query can pick these up by missing a filter. Each symbol's rows
-- are replaced in one transaction every run (section 7.2).
CREATE TABLE IF NOT EXISTS trending_prices (
    ticker      VARCHAR(10)     NOT NULL,
    ts          TIMESTAMPTZ     NOT NULL,   -- the trading date's 00:00 UTC, as raw_prices' daily bars
    open        NUMERIC(12, 4),
    high        NUMERIC(12, 4),
    low         NUMERIC(12, 4),
    close       NUMERIC(12, 4)  NOT NULL,
    volume      BIGINT,
    fetched_at  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    PRIMARY KEY (ticker, ts)
);

-- Every item each list returned, after symbol validation and before the
-- other rules, so any day's choice can be explained afterwards.
CREATE TABLE IF NOT EXISTS trending_snapshots (
    snapshot_ts   TIMESTAMPTZ     NOT NULL,
    session_date  DATE            NOT NULL,
    symbol        VARCHAR(10)     NOT NULL,
    source        VARCHAR(20)     NOT NULL,
    list_name     VARCHAR(40)     NOT NULL,
    rank          SMALLINT        NOT NULL,
    pct_change    NUMERIC(10, 4),
    volume        BIGINT,
    market_cap    BIGINT,
    UNIQUE (snapshot_ts, symbol, source, list_name)
);

CREATE INDEX IF NOT EXISTS trending_snapshots_symbol_session
    ON trending_snapshots (symbol, session_date);

-- Each list's outcome on each fetch (T8): what the absence rule (4.3) and the
-- tab's "last succeeded" line read. Explicit: read from trending_snapshots, a
-- list that answered with no items would look failed.
CREATE TABLE IF NOT EXISTS trending_fetches (
    fetched_at    TIMESTAMPTZ     NOT NULL,   -- the fetch's snapshot_ts
    session_date  DATE            NOT NULL,
    source        VARCHAR(20)     NOT NULL,
    list_name     VARCHAR(40)     NOT NULL,
    succeeded     BOOLEAN         NOT NULL,
    items         SMALLINT        NOT NULL,   -- rows it put in trending_snapshots
    error         VARCHAR(200),               -- a short reason, when it failed
    PRIMARY KEY (fetched_at, source, list_name)
);

-- news_sentiment's columns and key (sql/schema_sentiment.sql), and the index
-- the tab's reads use. Written out rather than copied with LIKE, whose
-- INCLUDING DEFAULTS would draw this table's ids from news_sentiment's own
-- sequence. Apart from it on purpose: the core's news features count every
-- stored news_sentiment row as a sign that a fetch happened.
CREATE TABLE IF NOT EXISTS trending_news (
    id            BIGSERIAL PRIMARY KEY,
    ticker        VARCHAR(10)     NOT NULL,
    source        VARCHAR(30)     NOT NULL,
    published_at  TIMESTAMPTZ     NOT NULL,
    headline      TEXT            NOT NULL,
    url           TEXT,
    sentiment     VARCHAR(10)     NOT NULL,
    score_pos     NUMERIC(6, 4),
    score_neg     NUMERIC(6, 4),
    score_neu     NUMERIC(6, 4),
    compound      NUMERIC(6, 4),
    inserted_at   TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    UNIQUE (ticker, headline, published_at)
);

CREATE INDEX IF NOT EXISTS idx_trending_news_ticker_ts
    ON trending_news (ticker, published_at DESC);
