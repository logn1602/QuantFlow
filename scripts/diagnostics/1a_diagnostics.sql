-- Read-only diagnostics for how stored bars became daily closes, run against
-- the live database on 2026-11-07, before the daily-bar backfill.
--   1. Which trading dates have only 15-min bars, and how much of the
--      published 30-day holdout (ending 2026-08-04) they cover.
--   2. How far the close the old loaders used (the 09:30 bar's) is from the
--      day's last 15-min close, and whether a seed-day daily bar was captured
--      mid-session.
--   3. Sanity checks: the session time zone, bar times, the seed runs.
--   4. Bars that are neither a daily bar nor on the 15-min session grid.
-- Run them with run_1a_diagnostics.py, or one at a time in the Supabase SQL
-- editor (pasted together, only the last result may show).

-- @query 1. trading dates with only 15-min bars
-- ============================================================================
-- 1a, query 1 of 3: which trading dates have only 15-min bars? (read-only)
--
-- A bar counts as a 15-min bar when its New York wall-clock time sits on the
-- regular-session grid, 09:30 to 15:45. Anything else is a daily bar: yfinance
-- stores those at midnight. A trading date is the UTC calendar date, which is
-- what the pipeline's ts::date returns in Supabase's UTC sessions.
-- ============================================================================
WITH bars AS (
    SELECT
        ticker,
        inserted_at,
        (ts AT TIME ZONE 'UTC')::date AS d,
        (ts AT TIME ZONE 'America/New_York')::time
            BETWEEN TIME '09:30' AND TIME '15:45' AS is_intraday
    FROM raw_prices
    WHERE source = 'yfinance'
),
per_date AS (
    SELECT
        ticker,
        d,
        bool_or(NOT is_intraday) AS has_daily,
        -- The table as the 2026-08-04 run saw it, assuming that run finished
        -- before 2026-08-05 12:00 UTC (08:00 in New York).
        bool_or(NOT is_intraday
                AND inserted_at < TIMESTAMPTZ '2026-08-05 12:00+00') AS had_daily_then
    FROM bars
    GROUP BY ticker, d
),
holdout AS (
    -- The published holdout window: the 30 latest trading dates up to 2026-08-04.
    SELECT ticker, d, has_daily, had_daily_then
    FROM (
        SELECT
            per_date.*,
            row_number() OVER (PARTITION BY ticker ORDER BY d DESC) AS rn
        FROM per_date
        WHERE d <= DATE '2026-08-04'
    ) ranked
    WHERE rn <= 30
),
holdout_summary AS (
    SELECT
        ticker,
        min(d)                                     AS holdout_start,
        max(d)                                     AS holdout_end,
        count(*)                                   AS holdout_dates,
        count(*) FILTER (WHERE NOT has_daily)      AS holdout_intraday_only_now,
        count(*) FILTER (WHERE NOT had_daily_then) AS holdout_intraday_only_then
    FROM holdout
    GROUP BY ticker
)
SELECT
    p.ticker,
    min(p.d)                                AS first_date,
    max(p.d)                                AS last_date,
    count(*)                                AS trading_dates,
    count(*) FILTER (WHERE p.has_daily)     AS dates_with_daily_bar,
    count(*) FILTER (WHERE NOT p.has_daily) AS intraday_only_dates,
    min(p.d) FILTER (WHERE NOT p.has_daily) AS first_intraday_only_date,
    max(p.d) FILTER (WHERE p.has_daily)     AS last_daily_bar_date,
    h.holdout_start,
    h.holdout_end,
    h.holdout_dates,
    h.holdout_intraday_only_now,
    h.holdout_intraday_only_then
FROM per_date p
LEFT JOIN holdout_summary h ON h.ticker = p.ticker
GROUP BY p.ticker, h.holdout_start, h.holdout_end, h.holdout_dates,
         h.holdout_intraday_only_now, h.holdout_intraday_only_then
ORDER BY p.ticker;

-- @query 2. how far off the close is on those dates
-- ============================================================================
-- 1a, query 2 of 3: how far off is the "close" on those dates? (read-only)
--
-- On an intraday-only date the loaders keep the 09:30 bar. Its close is
-- compared with the close of the day's last 15-min bar, which ends at 16:00.
-- ============================================================================
WITH bars AS (
    SELECT
        ticker,
        ts,
        close,
        inserted_at,
        (ts AT TIME ZONE 'UTC')::date AS d,
        (ts AT TIME ZONE 'America/New_York')::time
            BETWEEN TIME '09:30' AND TIME '15:45' AS is_intraday
    FROM raw_prices
    WHERE source = 'yfinance'
),
per_date AS (
    SELECT
        ticker,
        d,
        bool_or(NOT is_intraday)                                          AS has_daily,
        -- Same "as the 2026-08-04 run saw it" view as query 1.
        bool_or(NOT is_intraday
                AND inserted_at < TIMESTAMPTZ '2026-08-05 12:00+00')     AS had_daily_then,
        count(*) FILTER (WHERE is_intraday)                               AS n_bars,
        (array_agg(close ORDER BY ts)      FILTER (WHERE is_intraday))[1] AS first_bar_close,
        (array_agg(close ORDER BY ts DESC) FILTER (WHERE is_intraday))[1] AS last_bar_close,
        (array_agg(close ORDER BY ts) FILTER (WHERE NOT is_intraday))[1]  AS daily_close
    FROM bars
    GROUP BY ticker, d
),
flagged AS (
    SELECT
        per_date.*,
        -- Same 30-date window as query 1.
        d <= DATE '2026-08-04'
          AND row_number() OVER (
                  PARTITION BY ticker, d <= DATE '2026-08-04' ORDER BY d DESC
              ) <= 30 AS in_holdout
    FROM per_date
)
SELECT
    ticker,
    count(*) FILTER (WHERE NOT has_daily) AS intraday_only_dates,
    round(avg(abs(first_bar_close - last_bar_close) / last_bar_close * 100)
          FILTER (WHERE NOT has_daily), 3)                AS first_vs_last_bar_mean_pct,
    round(max(abs(first_bar_close - last_bar_close) / last_bar_close * 100)
          FILTER (WHERE NOT has_daily), 3)                AS first_vs_last_bar_max_pct,
    -- The published holdout's error: dates that were intraday-only when it ran.
    round(avg(abs(first_bar_close - last_bar_close) / last_bar_close * 100)
          FILTER (WHERE in_holdout AND NOT had_daily_then), 3) AS holdout_first_vs_last_bar_mean_pct,
    -- Dates that hold both kinds of bar: does the last 15-min bar match the
    -- daily close? That validates the proposed fallback, and exposes a daily
    -- bar captured mid-session (for example on the seed day).
    count(*) FILTER (WHERE has_daily AND n_bars > 0)      AS dates_with_both,
    round(avg(abs(daily_close - last_bar_close) / daily_close * 100)
          FILTER (WHERE has_daily AND n_bars > 0), 3)     AS daily_vs_last_bar_mean_pct,
    round(max(abs(daily_close - last_bar_close) / daily_close * 100)
          FILTER (WHERE has_daily AND n_bars > 0), 3)     AS daily_vs_last_bar_max_pct,
    -- 26 bars is a full session and 13 an early close; fewer means missing data.
    min(n_bars) FILTER (WHERE n_bars > 0)                 AS min_bars_per_date,
    percentile_disc(0.5) WITHIN GROUP (ORDER BY n_bars)
        FILTER (WHERE n_bars > 0)                         AS median_bars_per_date
FROM flagged
GROUP BY ticker
ORDER BY ticker;

-- @query 3. sanity checks behind queries 1 and 2
-- ============================================================================
-- 1a, query 3 of 3: sanity checks behind queries 1 and 2 (read-only)
-- ============================================================================
WITH bars AS (
    SELECT
        ticker,
        ts,
        inserted_at,
        (ts AT TIME ZONE 'UTC')::date AS d,
        (ts AT TIME ZONE 'America/New_York')::time
            BETWEEN TIME '09:30' AND TIME '15:45' AS is_intraday
    FROM raw_prices
    WHERE source = 'yfinance'
)
SELECT 'a. session TimeZone' AS check_name, NULL::varchar AS ticker,
       current_setting('TimeZone') AS value, NULL::bigint AS bars,
       NULL::date AS first_date, NULL::date AS last_date
UNION ALL
SELECT 'b. daily bar time of day, UTC', NULL,
       to_char(ts AT TIME ZONE 'UTC', 'HH24:MI'), count(*), min(d), max(d)
FROM bars
WHERE NOT is_intraday
GROUP BY to_char(ts AT TIME ZONE 'UTC', 'HH24:MI')
UNION ALL
SELECT 'c. 15-min bar times, New York', NULL,
       min((ts AT TIME ZONE 'America/New_York')::time)::text || ' to '
           || max((ts AT TIME ZONE 'America/New_York')::time)::text,
       count(*), min(d), max(d)
FROM bars
WHERE is_intraday
UNION ALL
SELECT 'd. daily bars by insert date (seed runs)', ticker,
       to_char(inserted_at AT TIME ZONE 'UTC', 'YYYY-MM-DD'), count(*), min(d), max(d)
FROM bars
WHERE NOT is_intraday
GROUP BY ticker, to_char(inserted_at AT TIME ZONE 'UTC', 'YYYY-MM-DD')
ORDER BY check_name, ticker, value;

-- @query 4. bars off the session grid
SELECT ticker, source, count(*) AS off_grid_rows,
       min(ts) AS first_ts, max(ts) AS last_ts
FROM raw_prices
WHERE NOT (
        (ts AT TIME ZONE 'UTC')::time = TIME '00:00'
     OR (ts AT TIME ZONE 'America/New_York')::time = TIME '00:00'
     OR (    (ts AT TIME ZONE 'America/New_York')::time
                 BETWEEN TIME '09:30' AND TIME '15:45'
         AND EXTRACT(SECOND FROM ts) = 0
         AND EXTRACT(MINUTE FROM ts AT TIME ZONE 'America/New_York')::int % 15 = 0)
)
GROUP BY ticker, source
ORDER BY ticker, source;
