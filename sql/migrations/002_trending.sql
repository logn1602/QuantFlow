-- ============================================================
-- 002_trending.sql
--
-- What it does
--   Creates the trending tier's five tables (docs/trending.md, section 6):
--   tickers, trending_prices, trending_snapshots, trending_fetches and
--   trending_news, with their keys and indexes. Nothing else: no core table
--   is read, changed or dropped.
--
-- Frozen once applied
--   Once the live database has run it, this file does not change: a later
--   change to these tables is a migration of its own (003_...). An edited
--   table's CREATE ... IF NOT EXISTS would skip the table that is there,
--   and change nothing.
--
-- Safe to run twice
--   Every statement is CREATE ... IF NOT EXISTS, inside one transaction, so a
--   second run changes nothing and a failed run leaves nothing behind.
--
-- When
--   Before the pull request that adds .github/workflows/trending.yml is
--   merged: the workflow starts after the next daily refresh on main, and
--   without the tables it stops at its first check, naming this file.
--
-- Who runs it
--   The owner, by hand: the pipeline never runs a migration. In Supabase's SQL
--   editor, which runs as postgres: the read-only role then reads the new
--   tables through its default privileges (sql/schema_roles.sql, step 4), so
--   the dashboard needs no grant. Run as any other role, follow it with
--   GRANT SELECT ON tickers, trending_prices, trending_snapshots,
--   trending_fetches, trending_news TO quantflow_readonly;
--
-- Check before (expected: one row of five empty values). A name instead
-- means a table by that name is there already, which this script would leave
-- as it is: stop and look at it first.
--   SELECT to_regclass('public.tickers'), to_regclass('public.trending_prices'),
--          to_regclass('public.trending_snapshots'),
--          to_regclass('public.trending_fetches'),
--          to_regclass('public.trending_news');
--
-- Check afterwards
--   The five tables, each with 0 rows:
--   SELECT relname, n_live_tup FROM pg_stat_user_tables
--   WHERE relname IN ('tickers', 'trending_prices', 'trending_snapshots',
--                     'trending_fetches', 'trending_news')
--   ORDER BY relname;
--   Their keys (expected: 7 rows; tickers p and c, trending_news p and u,
--   one each for the other three):
--   SELECT conrelid::regclass AS table_name, conname, contype
--   FROM pg_constraint
--   WHERE conrelid::regclass::text IN ('tickers', 'trending_prices',
--         'trending_snapshots', 'trending_fetches', 'trending_news')
--   ORDER BY 1, 2;
--   Who may do what with them (expected: postgres with every privilege, and
--   quantflow_readonly with SELECT once that role exists; any other role
--   listed, Supabase's anon and authenticated among them, is to be decided
--   before the tier writes there):
--   SELECT table_name, grantee,
--          string_agg(privilege_type, ', ' ORDER BY privilege_type)
--   FROM information_schema.role_table_grants
--   WHERE table_name IN ('tickers', 'trending_prices', 'trending_snapshots',
--                        'trending_fetches', 'trending_news')
--   GROUP BY 1, 2 ORDER BY 1, 2;
--
-- Undo (by hand, as the owner, and only while nothing in them matters)
--   DROP TABLE trending_news, trending_fetches, trending_snapshots,
--              trending_prices, tickers;
--   No core table refers to them, and the script creates nothing else.
--
-- Mirrored in sql/schema_trending.sql (make setup); tests/unit/
-- test_trending_schema.py fails if the two drift apart.
-- ============================================================

BEGIN;

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

COMMIT;
