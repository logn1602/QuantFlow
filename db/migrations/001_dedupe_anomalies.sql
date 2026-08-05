-- ============================================================
-- 001_dedupe_anomalies.sql
--
-- Problem
--   The live `anomalies` table carries only PRIMARY KEY (id). The
--   UNIQUE (ticker, ts) declared in db/schema.sql was never applied,
--   because the table already existed by the time that clause was added
--   and CREATE TABLE IF NOT EXISTS cannot retrofit a constraint.
--
--   With no unique index, the bare `ON CONFLICT DO NOTHING` in
--   anomaly_detection.save_anomalies had nothing to detect, so every run
--   appended another full copy of every flag:
--
--     269,350 rows  /  4,993 distinct (ticker, ts)  =  ~54 copies each
--                                                      (max 82)
--
--   Dashboard anomaly counts are inflated by the same factor. The flagged
--   timestamps themselves are correct.
--
-- Which duplicate to keep
--   Copies are NOT identical: 2,300 keys have differing close/zscore/flag,
--   because the rolling Z-score window and the IQR fences are recomputed
--   over a series that grows daily. The highest id is the most recent
--   estimate and therefore the one derived from the most price history,
--   so that is the row we keep.
--
-- Effect
--   Deletes 264,357 rows. Irreversible — take a Supabase backup /
--   point-in-time snapshot before running.
--
-- Run once:
--   psql "$DATABASE_URL" -f db/migrations/001_dedupe_anomalies.sql
-- ============================================================

BEGIN;

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
            'anomalies already has UNIQUE (ticker, ts) — migration already applied';
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
