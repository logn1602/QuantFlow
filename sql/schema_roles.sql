-- ============================================================
-- schema_roles.sql
--
-- quantflow_readonly: a role that can only read, for every reader of the live
-- database outside the pipeline.
--
--   - The diagnostics (scripts/diagnostics/run_1a_diagnostics.py --live) and
--     the evaluation's snapshot export (pipelines/evaluate.py export --live)
--     log in as it, from DB_READONLY_USER and DB_READONLY_PASSWORD in .env,
--     and refuse to run without them (db/readonly.py). Their query checks
--     are scans of the SQL; this role is the guarantee that does not depend
--     on them.
--   - The public Streamlit dashboard, which only reads, logs in as it through
--     its DB_USER and DB_PASSWORD secrets, so the deployed app holds no
--     credential that can write.
--
-- Run by hand in Supabase's SQL editor, which runs as postgres. `make setup`
-- does not run this file. Roles are cluster-wide; Supabase's pooler takes the
-- user as quantflow_readonly.<project-ref>, the same suffix the postgres user
-- has in the connection string.
--
-- Run steps 1 to 4 as they are: the role gets no password here, so it cannot
-- log in until step 5. Then run step 5 on its own with a real secret.
-- Validated on a local Postgres 17 cluster (2026-11-09): the role reads every
-- table, including one created afterwards; INSERT fails; DELETE fails on
-- privileges even with the read-only default switched off; it cannot create
-- a table, a replication slot, or end another session.
--
-- Steps 7 to 9 (2026-11-11): row-level security. The live database has it on
-- for every core table, and a role that does not own a table sees none of
-- its rows unless a policy lets it, with no error: quantflow_readonly read
-- every table as empty. Step 7 gives it a SELECT policy on each core table
-- and takes every privilege from Supabase's anon and authenticated roles;
-- step 8 does the same for the trending tables on a fresh install (on the
-- live database, sql/migrations/002_trending.sql runs those statements);
-- step 9 checks. On the live database, steps 1 to 6 are done: run step 7 on
-- its own, then step 9.
-- ============================================================

-- 1. A role that can log in and do nothing else. Ten connections: the
--    dashboard's sessions and a --live run share it.
CREATE ROLE quantflow_readonly
    LOGIN
    NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS
    CONNECTION LIMIT 10;

-- 2. Its sessions start read-only and time-limited. These are defaults a
--    session could change; the privileges in step 3 are what stop writes.
ALTER ROLE quantflow_readonly SET default_transaction_read_only = on;
ALTER ROLE quantflow_readonly SET statement_timeout = '120s';
ALTER ROLE quantflow_readonly SET idle_in_transaction_session_timeout = '60s';

-- 3. Reach the database and the public schema, and read its tables: SELECT
--    only. Where postgres does not own the database or the schema, the first
--    two print "no privileges were granted"; every role has both through
--    PUBLIC by default, and step 6 checks it.
GRANT CONNECT ON DATABASE postgres TO quantflow_readonly;
GRANT USAGE ON SCHEMA public TO quantflow_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO quantflow_readonly;

-- 4. Tables postgres creates later (the trending tables, say) are readable
--    too.
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
    GRANT SELECT ON TABLES TO quantflow_readonly;

-- 5. Its password, run on its own. Use letters and digits only (for example
--    the output of `openssl rand -hex 32`): the dashboard builds its
--    connection URL from DB_PASSWORD unescaped, so a character such as @, /
--    or : would break it.
--
-- ALTER ROLE quantflow_readonly PASSWORD 'replace-with-the-secret';

-- 6. Check: it can connect and use the schema (both true), and holds only
--    SELECT, on every table.
--
-- SELECT has_database_privilege('quantflow_readonly', 'postgres', 'CONNECT'),
--        has_schema_privilege('quantflow_readonly', 'public', 'USAGE');
--
-- SELECT table_name, string_agg(privilege_type, ', ') AS privileges
-- FROM information_schema.role_table_grants
-- WHERE grantee = 'quantflow_readonly'
-- GROUP BY table_name
-- ORDER BY table_name;

-- 7. The core tables: row-level security on (already on in the live
--    database, where that line changes nothing), a policy that lets
--    quantflow_readonly read every row, and no privilege at all for anon and
--    authenticated, the roles Supabase's Data API uses and QuantFlow never
--    does (row-level security already hides every row from them; this takes
--    the privileges as well). postgres owns the tables, and row-level
--    security does not apply to a table's owner, so the pipeline, which
--    writes as postgres, is unaffected. The policies are FOR SELECT only,
--    and the role has no privilege to write either.
--    One transaction: a failure leaves nothing changed, and a second run
--    stops at its first CREATE POLICY ("already exists"). Each ALTER and
--    CREATE POLICY locks its table, so the lock timeout stops the script
--    rather than leave it queued behind a long read (a daily refresh in
--    progress), with the dashboard's reads queued behind it; run it again
--    once the refresh is done. On a Postgres without Supabase's roles (a
--    local cluster), leave out the two REVOKE statements.
BEGIN;
SET LOCAL lock_timeout = '10s';

ALTER TABLE raw_prices ENABLE ROW LEVEL SECURITY;
ALTER TABLE technical_indicators ENABLE ROW LEVEL SECURITY;
ALTER TABLE anomalies ENABLE ROW LEVEL SECURITY;
ALTER TABLE forecasts ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE news_sentiment ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE backtest_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY quantflow_readonly_read ON raw_prices
    FOR SELECT TO quantflow_readonly USING (true);
CREATE POLICY quantflow_readonly_read ON technical_indicators
    FOR SELECT TO quantflow_readonly USING (true);
CREATE POLICY quantflow_readonly_read ON anomalies
    FOR SELECT TO quantflow_readonly USING (true);
CREATE POLICY quantflow_readonly_read ON forecasts
    FOR SELECT TO quantflow_readonly USING (true);
CREATE POLICY quantflow_readonly_read ON alerts_log
    FOR SELECT TO quantflow_readonly USING (true);
CREATE POLICY quantflow_readonly_read ON news_sentiment
    FOR SELECT TO quantflow_readonly USING (true);
CREATE POLICY quantflow_readonly_read ON model_metrics
    FOR SELECT TO quantflow_readonly USING (true);
CREATE POLICY quantflow_readonly_read ON backtest_results
    FOR SELECT TO quantflow_readonly USING (true);

REVOKE ALL ON raw_prices, technical_indicators, anomalies, forecasts,
    alerts_log, news_sentiment, model_metrics, backtest_results
    FROM anon, authenticated;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM anon, authenticated;

COMMIT;

-- 8. The trending tables, on a fresh install only: on the live database,
--    sql/migrations/002_trending.sql runs these statements as it creates the
--    tables (tests/unit/test_trending_schema.py keeps the two the same). Run
--    after make setup, whose sql/schema_trending.sql creates the tables with
--    row-level security on. On a Postgres without Supabase's roles, leave
--    out the two REVOKE statements.
BEGIN;

CREATE POLICY quantflow_readonly_read ON tickers
    FOR SELECT TO quantflow_readonly USING (true);
CREATE POLICY quantflow_readonly_read ON trending_prices
    FOR SELECT TO quantflow_readonly USING (true);
CREATE POLICY quantflow_readonly_read ON trending_snapshots
    FOR SELECT TO quantflow_readonly USING (true);
CREATE POLICY quantflow_readonly_read ON trending_fetches
    FOR SELECT TO quantflow_readonly USING (true);
CREATE POLICY quantflow_readonly_read ON trending_news
    FOR SELECT TO quantflow_readonly USING (true);

REVOKE ALL ON tickers, trending_prices, trending_snapshots, trending_fetches,
    trending_news FROM anon, authenticated;
REVOKE ALL ON SEQUENCE trending_news_id_seq FROM anon, authenticated;

COMMIT;

-- 9. Checks, after step 7 (and step 8, or 002_trending.sql).
--
--    In the SQL editor, one row per table in public. Expected on every row:
--    rls_on, read_policy and readonly_select true; readonly_writes and
--    api_roles false. The eight core tables; the five trending ones too,
--    once they exist.
--
-- SELECT c.relname AS table_name,
--        c.relrowsecurity AS rls_on,
--        EXISTS (SELECT 1 FROM pg_policies p
--                WHERE p.schemaname = 'public' AND p.tablename = c.relname
--                  AND p.policyname = 'quantflow_readonly_read'
--                  AND p.cmd = 'SELECT' AND p.permissive = 'PERMISSIVE'
--                  AND p.roles = '{quantflow_readonly}' AND p.qual = 'true')
--            AS read_policy,
--        has_table_privilege('quantflow_readonly', c.oid, 'SELECT')
--            AS readonly_select,
--        has_table_privilege('quantflow_readonly', c.oid,
--                            'INSERT, UPDATE, DELETE, TRUNCATE')
--            AS readonly_writes,
--        has_table_privilege('anon', c.oid,
--            'SELECT, INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER')
--        OR has_table_privilege('authenticated', c.oid,
--            'SELECT, INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER')
--            AS api_roles
-- FROM pg_class c
-- JOIN pg_namespace n ON n.oid = c.relnamespace
-- WHERE n.nspname = 'public' AND c.relkind = 'r'
-- ORDER BY 1;
--
--    And as the role itself, the proof that it now reads rows: from the
--    repository, which logs in with DB_READONLY_USER and DB_READONLY_PASSWORD
--    from .env, in one read-only transaction. One count per core table,
--    which the same query run here as postgres matches; before step 7 it
--    stops with "no row visible to quantflow_readonly" instead.
--
--   uv run python scripts/diagnostics/run_1a_diagnostics.py --sql scripts/diagnostics/readonly_rows.sql --live --out readonly_rows.md
