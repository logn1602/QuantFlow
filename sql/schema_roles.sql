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
-- step 9 checks.
--
-- On the live database (2026-11-11): steps 1 to 10 are done, step 7 before
-- sql/migrations/002_trending.sql and step 10 after it; steps 11 to 14
-- wait for the owner.
--
-- Step 10 (2026-11-11): what postgres creates later grants Supabase's anon
-- and authenticated nothing. Run on the live database by the owner after
-- 002_trending.sql, and recorded here for a fresh install.
--
-- Steps 11 to 14: quantflow_trending, the trending tier's own login, for
-- .github/workflows/trending.yml (its TRENDING_DB_USER and
-- TRENDING_DB_PASSWORD secrets), so the tier no longer writes as postgres,
-- which owns every table, the core's included, and which row-level security
-- does not apply to. The new role owns nothing, so row-level security
-- applies to it, and it holds only the commands one run of the tier uses,
-- on the five trending tables: never a core one. On the live database, run
-- step 11, then step 12 on its own with a real secret, then step 13, then
-- step 14's checks; set the two secrets after that.
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

-- 10. What postgres creates later in public: no privilege for anon and
--     authenticated. Supabase's own default privileges granted them
--     everything on each new table, sequence and function, which is why
--     sql/migrations/002_trending.sql revokes it on the tables it creates;
--     now nothing is granted to begin with. service_role keeps its grants,
--     which are Supabase's own, and step 4's SELECT for quantflow_readonly
--     stays. Run on the live database by the owner on 2026-11-11; running
--     it again changes nothing. On a Postgres without Supabase's roles (a
--     local cluster), leave it out.
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
    REVOKE ALL ON TABLES FROM anon, authenticated;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
    REVOKE ALL ON SEQUENCES FROM anon, authenticated;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
    REVOKE ALL ON FUNCTIONS FROM anon, authenticated;

--    Check: postgres's default privileges in public name neither anon nor
--    authenticated. Expected: no row.
--
-- SELECT d.defaclobjtype, acl.grantee::regrole
-- FROM pg_default_acl d
-- JOIN pg_namespace n ON n.oid = d.defaclnamespace
-- CROSS JOIN LATERAL aclexplode(d.defaclacl) AS acl
-- WHERE d.defaclrole = 'postgres'::regrole AND n.nspname = 'public'
--   AND acl.grantee IN ('anon'::regrole, 'authenticated'::regrole);

-- 11. quantflow_trending: a login that can reach the database and the
--     public schema, and do nothing there until step 13. Five connections,
--     a margin over the one a run uses at a time (trending.yml runs one run
--     at a time; a local run logs in as DB_USER, not as this role).
--     A statement stuck behind a lock (a migration's, say) fails after two
--     minutes rather than hold the run to the job's 30-minute timeout; no
--     statement of a run takes more than seconds, and none leaves a
--     transaction idle. As in step 3, where postgres does not own the
--     database or the schema, the two GRANTs print "no privileges were
--     granted": every role has both through PUBLIC by default, and step 14
--     checks it.
CREATE ROLE quantflow_trending
    LOGIN
    NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS
    CONNECTION LIMIT 5;
ALTER ROLE quantflow_trending SET statement_timeout = '120s';
ALTER ROLE quantflow_trending SET idle_in_transaction_session_timeout = '60s';
GRANT CONNECT ON DATABASE postgres TO quantflow_trending;
GRANT USAGE ON SCHEMA public TO quantflow_trending;

-- 12. Its password, run on its own, with a real secret: letters and digits
--     only, as in step 5, since the pipeline puts DB_PASSWORD in its
--     connection URL unescaped (config.DATABASE_URL). The same value goes in
--     the repository's TRENDING_DB_PASSWORD secret, and nowhere else.
--
-- ALTER ROLE quantflow_trending PASSWORD 'replace-with-the-secret';

-- 13. What it may do: privileges, and a policy for each, since row-level
--     security, on for all five tables, refuses any row no policy allows.
--     Exactly the commands one run uses (src/quantflow/db/trending.py, and
--     db/sentiment.py for trending_news; tests/unit/test_schema_roles.py
--     takes the lists from that code):
--       - SELECT on all five: the run reads tickers and trending_fetches,
--         and an upsert, or an UPDATE or DELETE with a WHERE, reads the rows
--         it changes;
--       - INSERT on all five, and on tickers only in the columns the run
--         writes, and UPDATE of those but symbol and first_seen: it upserts
--         its rows there, and can never set tier or promoted_at, which
--         change by hand only, at a promotion (docs/trending.md, section 8);
--       - DELETE on trending_prices: each symbol's bars are replaced;
--       - USAGE on trending_news_id_seq, which trending_news's ids come from.
--     Nothing on any core table, no TRUNCATE, no DELETE elsewhere, and no
--     default privileges: a table added later gets nothing until its own
--     migration grants it. One transaction: a failure leaves nothing
--     changed, and a second run stops at its first CREATE POLICY ("already
--     exists"). As in step 7, the lock timeout stops it rather than leave it
--     queued behind a long read; run it again once the read is done. Run it
--     outside the daily refresh and the trending run after it (weekdays
--     from 22:30 UTC until trending.yml has finished): a run's
--     transactions lock tickers last and this step locks it first, so the
--     two can deadlock rather than wait, and if Postgres ends the run's
--     transaction, the run ends red.
BEGIN;
SET LOCAL lock_timeout = '10s';

GRANT SELECT ON tickers, trending_prices, trending_snapshots, trending_fetches,
    trending_news TO quantflow_trending;
GRANT INSERT ON trending_prices, trending_snapshots, trending_fetches,
    trending_news TO quantflow_trending;
GRANT INSERT (symbol, short_name, quote_type, first_seen, last_seen, active,
    pinned, first_bar_date, first_trade_date, summary_session, last_close,
    rsi_14, recent_flags, big_move_date) ON tickers TO quantflow_trending;
GRANT UPDATE (short_name, quote_type, last_seen, active, pinned,
    first_bar_date, first_trade_date, summary_session, last_close, rsi_14,
    recent_flags, big_move_date) ON tickers TO quantflow_trending;
GRANT DELETE ON trending_prices TO quantflow_trending;
GRANT USAGE ON SEQUENCE trending_news_id_seq TO quantflow_trending;

CREATE POLICY quantflow_trending_read ON tickers
    FOR SELECT TO quantflow_trending USING (true);
CREATE POLICY quantflow_trending_read ON trending_prices
    FOR SELECT TO quantflow_trending USING (true);
CREATE POLICY quantflow_trending_read ON trending_snapshots
    FOR SELECT TO quantflow_trending USING (true);
CREATE POLICY quantflow_trending_read ON trending_fetches
    FOR SELECT TO quantflow_trending USING (true);
CREATE POLICY quantflow_trending_read ON trending_news
    FOR SELECT TO quantflow_trending USING (true);

CREATE POLICY quantflow_trending_insert ON tickers
    FOR INSERT TO quantflow_trending WITH CHECK (true);
CREATE POLICY quantflow_trending_insert ON trending_prices
    FOR INSERT TO quantflow_trending WITH CHECK (true);
CREATE POLICY quantflow_trending_insert ON trending_snapshots
    FOR INSERT TO quantflow_trending WITH CHECK (true);
CREATE POLICY quantflow_trending_insert ON trending_fetches
    FOR INSERT TO quantflow_trending WITH CHECK (true);
CREATE POLICY quantflow_trending_insert ON trending_news
    FOR INSERT TO quantflow_trending WITH CHECK (true);

CREATE POLICY quantflow_trending_update ON tickers
    FOR UPDATE TO quantflow_trending USING (true) WITH CHECK (true);
CREATE POLICY quantflow_trending_delete ON trending_prices
    FOR DELETE TO quantflow_trending USING (true);

COMMIT;

-- 14. Checks, after step 13, in the SQL editor.
--
--    The role. Expected one row: rolcanlogin t; rolsuper, rolcreatedb,
--    rolcreaterole, rolreplication and rolbypassrls f; rolconnlimit 5;
--    rolconfig the two settings.
--
-- SELECT rolname, rolcanlogin, rolsuper, rolcreatedb, rolcreaterole,
--        rolreplication, rolbypassrls, rolconnlimit, rolconfig
-- FROM pg_roles WHERE rolname = 'quantflow_trending';
--
--    It can connect and use the schema, and cannot create in it. Expected:
--    t, t, f.
--
-- SELECT has_database_privilege('quantflow_trending', 'postgres', 'CONNECT'),
--        has_schema_privilege('quantflow_trending', 'public', 'USAGE'),
--        has_schema_privilege('quantflow_trending', 'public', 'CREATE');
--
--    What it holds on whole tables and sequences, in any schema. Expected
--    six rows, and no core table: tickers SELECT; trending_fetches,
--    trending_news and trending_snapshots INSERT, SELECT;
--    trending_news_id_seq USAGE; trending_prices DELETE, INSERT, SELECT.
--
-- SELECT c.relname,
--        string_agg(acl.privilege_type, ', ' ORDER BY acl.privilege_type)
-- FROM pg_class c
-- CROSS JOIN LATERAL aclexplode(c.relacl) AS acl
-- WHERE acl.grantee = 'quantflow_trending'::regrole
-- GROUP BY 1 ORDER BY 1;
--
--    And on tickers' columns. Expected 14 rows: symbol and first_seen
--    INSERT, the other twelve INSERT, UPDATE; no tier, no promoted_at.
--
-- SELECT a.attname,
--        string_agg(acl.privilege_type, ', ' ORDER BY acl.privilege_type)
-- FROM pg_attribute a
-- CROSS JOIN LATERAL aclexplode(a.attacl) AS acl
-- WHERE a.attrelid = 'public.tickers'::regclass
--   AND acl.grantee = 'quantflow_trending'::regrole
-- GROUP BY a.attname, a.attnum ORDER BY a.attnum;
--
--    Its policies. Expected 12 rows, each for {quantflow_trending}:
--    quantflow_trending_read (SELECT) and quantflow_trending_insert (INSERT)
--    on each of the five tables, quantflow_trending_update (UPDATE) on
--    tickers, quantflow_trending_delete (DELETE) on trending_prices.
--
-- SELECT tablename, policyname, cmd, roles::text, qual, with_check
-- FROM pg_policies
-- WHERE schemaname = 'public' AND 'quantflow_trending' = ANY (roles)
-- ORDER BY tablename, policyname;

-- 15. Undo, by hand, only if the role must go (as postgres). The role owns
--     nothing, so no row is at stake, but DROP ROLE fails while any grant or
--     policy still names it.
--       a. First put trending.yml's run step back on secrets.DB_USER and
--          secrets.DB_PASSWORD, merged, or every run after this fails at
--          login.
--       b. Then, in one transaction (the lock timeout as in step 13, and
--          outside the same window):
--
-- BEGIN;
-- SET LOCAL lock_timeout = '10s';
-- DROP POLICY quantflow_trending_read ON tickers;
-- DROP POLICY quantflow_trending_read ON trending_prices;
-- DROP POLICY quantflow_trending_read ON trending_snapshots;
-- DROP POLICY quantflow_trending_read ON trending_fetches;
-- DROP POLICY quantflow_trending_read ON trending_news;
-- DROP POLICY quantflow_trending_insert ON tickers;
-- DROP POLICY quantflow_trending_insert ON trending_prices;
-- DROP POLICY quantflow_trending_insert ON trending_snapshots;
-- DROP POLICY quantflow_trending_insert ON trending_fetches;
-- DROP POLICY quantflow_trending_insert ON trending_news;
-- DROP POLICY quantflow_trending_update ON tickers;
-- DROP POLICY quantflow_trending_delete ON trending_prices;
-- REVOKE ALL ON tickers, trending_prices, trending_snapshots, trending_fetches,
--     trending_news FROM quantflow_trending;
-- REVOKE ALL ON SEQUENCE trending_news_id_seq FROM quantflow_trending;
-- REVOKE CONNECT ON DATABASE postgres FROM quantflow_trending;
-- REVOKE USAGE ON SCHEMA public FROM quantflow_trending;
-- COMMIT;
--
--       c. Then DROP ROLE quantflow_trending; on its own (REVOKE ALL ON a
--          table takes its column grants too). Step 14's queries then find
--          no row for it. tests/integration/test_trending_on_postgres.py runs
--          steps 11 and 13 and this undo, and the DROP ROLE must succeed.
