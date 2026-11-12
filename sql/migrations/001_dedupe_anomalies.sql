-- ============================================================
-- 001_dedupe_anomalies.sql
--
-- Problem
--   The live `anomalies` table carries only PRIMARY KEY (id). The
--   UNIQUE (ticker, ts) declared in sql/schema.sql was never applied,
--   because the table already existed by the time that clause was added
--   and CREATE TABLE IF NOT EXISTS cannot retrofit a constraint.
--
--   With no unique index, the bare `ON CONFLICT DO NOTHING` in
--   features.anomalies.save_anomalies had nothing to detect, so every run
--   appended another full copy of every flag. On 2026-08-04:
--
--     269,350 rows  /  4,993 distinct (ticker, ts)  =  ~54 copies each
--                                                      (max 82)
--
--   Dashboard anomaly counts were inflated by the same factor. The flagged
--   timestamps themselves were correct.
--
--   Since Phase 1, save_anomalies replaces each ticker's flags inside a
--   transaction, so the table holds one row per key again (769 rows in
--   the 2026-11-11 backup): this script now deletes few rows or none, and
--   what it still adds is the constraint.
--
-- Which duplicate to keep
--   Copies are NOT identical: 2,300 keys have differing close/zscore/flag,
--   because the rolling Z-score window and the IQR fences are recomputed
--   over a series that grows daily. The highest id is the most recent
--   estimate and therefore the one derived from the most price history,
--   so that is the row we keep.
--
-- Check first (expected: rows equal to keys, so nothing to delete):
--   SELECT count(*) AS rows, count(DISTINCT (ticker, ts)) AS keys
--   FROM anomalies;
--
-- Effect
--   Deletes every row but the newest of each (ticker, ts), then adds the
--   constraint. Irreversible: take a backup first (pg_dump; Supabase's
--   free plan keeps none of its own).
--
-- Run once, by hand: in Supabase's SQL editor, as postgres, outside the
-- daily refresh (weekdays from 22:30 UTC until it has finished). The ALTER
-- TABLE takes an exclusive lock on anomalies, which the refresh's anomaly
-- step and the dashboard use; the lock timeout below stops the script
-- rather than leave them queued behind it. A second run stops at its first
-- check. On a local database:
--   psql -U postgres -d stock_pipeline -f sql/migrations/001_dedupe_anomalies.sql
-- ============================================================

BEGIN;
SET LOCAL lock_timeout = '10s';

-- Fail loudly rather than silently doing nothing if this has already run.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_constraint con
        JOIN pg_class rel ON rel.oid = con.conrelid
        WHERE rel.relname = 'anomalies'
          AND con.contype = 'u'
          AND pg_get_constraintdef(con.oid) = 'UNIQUE (ticker, ts)'
    ) THEN
        RAISE EXCEPTION
            'anomalies already has UNIQUE (ticker, ts): migration already applied';
    END IF;
END $$;

-- 1. Drop duplicate flags, keeping the newest row per (ticker, ts).
DELETE FROM anomalies a
USING anomalies b
WHERE a.ticker = b.ticker
  AND a.ts     = b.ts
  AND a.id     < b.id;

-- 2. With the table unique on (ticker, ts), add the constraint the schema
--    always intended. This also gives ON CONFLICT something to detect.
ALTER TABLE anomalies
    ADD CONSTRAINT anomalies_ticker_ts_key UNIQUE (ticker, ts);

-- 3. Confirm the invariant holds before committing.
DO $$
DECLARE
    total    BIGINT;
    distinct_keys BIGINT;
BEGIN
    SELECT COUNT(*) INTO total FROM anomalies;
    SELECT COUNT(*) INTO distinct_keys
        FROM (SELECT DISTINCT ticker, ts FROM anomalies) x;
    IF total <> distinct_keys THEN
        RAISE EXCEPTION
            'dedupe failed: % rows vs % distinct keys', total, distinct_keys;
    END IF;
    RAISE NOTICE 'anomalies deduped: % rows, one per (ticker, ts)', total;
END $$;

COMMIT;
