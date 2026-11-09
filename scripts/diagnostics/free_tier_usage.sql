-- Free-tier usage, for docs/architecture.md ("Running on free tiers"): the
-- database's size against Supabase free's 500 MB, each table's share, what
-- the last 30 days added, and the raw_prices rows per ticker, which drive
-- the daily run's egress (load_daily_bars reads a ticker's whole history).
--
-- Read-only: every query is a single SELECT. Run it through the runner,
-- which opens one READ ONLY transaction and rolls it back:
--   uv run python scripts/diagnostics/run_1a_diagnostics.py --sql scripts/diagnostics/free_tier_usage.sql --live --out usage.md

-- @query database size
SELECT pg_database_size(current_database()) AS bytes,
       pg_size_pretty(pg_database_size(current_database())) AS size;

-- @query tables by size
-- Every schema's user tables, Supabase's own included, with indexes and TOAST.
SELECT schemaname AS schema_name,
       relname AS table_name,
       pg_total_relation_size(relid) AS bytes,
       pg_size_pretty(pg_total_relation_size(relid)) AS size,
       n_live_tup AS approx_rows
FROM pg_stat_user_tables
ORDER BY pg_total_relation_size(relid) DESC
LIMIT 30;

-- @query rows added in the last 30 days
SELECT 'raw_prices' AS table_name, count(*) AS added
FROM raw_prices WHERE inserted_at >= now() - interval '30 days'
UNION ALL
SELECT 'news_sentiment', count(*)
FROM news_sentiment WHERE inserted_at >= now() - interval '30 days'
UNION ALL
SELECT 'model_metrics', count(*)
FROM model_metrics WHERE run_at >= now() - interval '30 days'
UNION ALL
SELECT 'backtest_results', count(*)
FROM backtest_results WHERE run_at >= now() - interval '30 days';

-- @query raw_prices rows per ticker
-- Egress depends only on the total, since load_daily_bars reads every row.
-- The split is a rough one: rows at 00:00 UTC are where the daily bars sit,
-- the rest are mostly 15-minute bars (db/prices.classify_bars has the full
-- rule the loaders apply, which also sets aside rows off the session grid).
SELECT ticker,
       source,
       count(*) AS total_rows,
       count(*) FILTER (WHERE (ts AT TIME ZONE 'UTC')::time = time '00:00') AS rows_at_midnight_utc,
       count(*) FILTER (WHERE (ts AT TIME ZONE 'UTC')::time <> time '00:00') AS other_rows,
       -- UTC dates, as the bars are dated, whatever the session's time zone.
       min(ts AT TIME ZONE 'UTC')::date AS first_date,
       max(ts AT TIME ZONE 'UTC')::date AS last_date
FROM raw_prices
GROUP BY ticker, source
ORDER BY ticker, source;
