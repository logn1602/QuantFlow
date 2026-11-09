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
