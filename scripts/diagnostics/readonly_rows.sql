-- ============================================================
-- readonly_rows.sql
--
-- What the read-only role sees: one row per core table, with the rows a
-- count finds (sql/schema_roles.sql, step 9). Run as the role itself, through
-- the diagnostics runner, whose --live logs in as DB_READONLY_USER:
--
--   uv run python scripts/diagnostics/run_1a_diagnostics.py --sql scripts/diagnostics/readonly_rows.sql --live --out readonly_rows.md
--
-- Before the role's read policies (step 7), row-level security hides every
-- row from it, and the runner stops before this query, naming each table
-- that shows none. After them, each count is the table's own: the same
-- query run as postgres in Supabase's SQL editor gives the same numbers,
-- give or take rows written in between. alerts_log is 0 until an alert
-- fires.
-- ============================================================

-- @query visible_rows
SELECT 'alerts_log' AS table_name, count(*) AS rows_visible FROM alerts_log
UNION ALL SELECT 'anomalies', count(*) FROM anomalies
UNION ALL SELECT 'backtest_results', count(*) FROM backtest_results
UNION ALL SELECT 'forecasts', count(*) FROM forecasts
UNION ALL SELECT 'model_metrics', count(*) FROM model_metrics
UNION ALL SELECT 'news_sentiment', count(*) FROM news_sentiment
UNION ALL SELECT 'raw_prices', count(*) FROM raw_prices
UNION ALL SELECT 'technical_indicators', count(*) FROM technical_indicators
ORDER BY table_name
