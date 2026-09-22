-- Which segment of a provided session does a purchase fall in?
--
-- This is the mechanism behind sessionization_purchase_rate.sql. That query
-- prices the distortion as a business metric; this one says how often the
-- provided session definition stitches a purchase onto browsing it was already
-- disconnected from. A purchase at session_index = 0 sat in an unbroken run of
-- activity, so `user_session` described it correctly. A purchase at a higher
-- index sits after at least one gap longer than the timeout, which means the
-- provided definition merged two visits into one.
--
-- FILTER PLACEMENT IS THE WHOLE QUERY. The WHERE sits in the FINAL block, after
-- the window functions have already resolved. Moved up into the FROM events
-- block it would still parse, still run, and still return a plausible table --
-- but LAG would then mean "the previous PURCHASE" rather than "the previous
-- event", so the gaps, the marks and the segment numbers would all describe a
-- different population. Measured on the development sample, filtering first
-- yields 23,830 gaps averaging 359.6s against 151,783 averaging 111.0s the
-- correct way round: a 6.4x difference in rows, silently.
--
-- The two buckets are exhaustive by construction, and the reconciliation column
-- proves it rather than asserting it: session_index is a running SUM over 0/1
-- marks and so is never NULL, which means `= 0` and `> 0` cannot both miss a
-- row. If purchase_events ever stops matching the funnel count in
-- event_type_distribution.sql, the partitioning is wrong.
--
-- DIALECT: written in BigQuery form, TIMESTAMP_DIFF(end, start, SECOND) with
-- end first. DuckDB has no TIMESTAMP_DIFF; scripts/run_sql.py defines it as a
-- macro that only honours SECOND (docs/sql_dialects.md).
WITH events_with_previous_time AS (
    SELECT
        user_session,
        event_time,
        event_type,
        LAG(event_time, 1) OVER (
            PARTITION BY user_session ORDER BY event_time
        ) AS previous_event_time
    FROM events
),

events_with_gaps AS (
    SELECT
        *,
        TIMESTAMP_DIFF(event_time, previous_event_time, SECOND) AS gap_seconds
    FROM events_with_previous_time
),

marked_events AS (
    SELECT
        *,
        CASE WHEN gap_seconds > 1800 THEN 1 ELSE 0 END AS is_session_start
    FROM events_with_gaps
),

indexed_events AS (
    -- Default RANGE frame retained: see sessionization_purchase_rate.sql for
    -- why ROWS would make tied timestamps non-deterministic.
    SELECT
        *,
        SUM(is_session_start) OVER (
            PARTITION BY user_session ORDER BY event_time
        ) AS session_index
    FROM marked_events
)

-- Dataset grain, restricted to purchases. COUNT(*) is the purchase total
-- because the WHERE has already removed everything else, which makes it the
-- denominator these rates want without a second conditional count to keep in
-- sync.
SELECT
    COUNT(*) AS purchase_events,
    COUNT(CASE WHEN session_index = 0 THEN 1 END) AS purchases_in_first_segment,
    COUNT(CASE WHEN session_index > 0 THEN 1 END) AS purchases_after_a_break,
    ROUND(
        COUNT(CASE WHEN session_index > 0 THEN 1 END) * 100.0 / COUNT(*), 4
    ) AS purchases_after_a_break_pct
FROM indexed_events
WHERE event_type = 'purchase';
