-- Events per session under two session definitions.
--
-- Session counts alone cannot say whether the cut changed the shape of a
-- session or merely its label. This distribution answers that: if the timeout
-- is doing real work, the cut definition should hold visibly more short
-- sessions, because every split turns one long session into two shorter ones.
--
-- The comparison lives in the cumulative column rather than the per-bucket
-- share. Session length is heavily right-skewed -- a third of sessions hold a
-- single event while the maximum runs into the hundreds -- so quantiles would
-- all collapse into the 1-5 range and hide the difference. The cumulative share
-- is the inverse function of the quantiles and keeps the resolution where the
-- mass is: "what fraction of sessions are at most this long".
--
-- Both window functions PARTITION BY session_definition. Without it the total
-- would span both definitions at once and every share would be halved, which is
-- a plausible-looking number rather than an error.
--
-- The cumulative window orders by event_count, NOT by session_count. Ordering
-- by frequency happens to give the right answer at the head, because frequency
-- falls monotonically there, and a wrong one in the tail -- correct exactly
-- where it gets checked. event_count is unique inside each partition here, so
-- the RANGE-versus-ROWS distinction that matters upstream is moot at this grain.
--
-- SUM(...) OVER () is cheap at this grain and only at this grain: the input is
-- a few hundred rows, one per bucket. The same construct over the event table
-- is a measured 4.4x slower than a separate aggregate and should not be copied
-- upward from here.
--
-- DIALECT: date_diff(part, start, end) is the DuckDB signature and returns
-- end - start. BigQuery spells this TIMESTAMP_DIFF(end, start, part) -- different
-- name, different argument order, and the two timestamps swap places. Porting
-- the arguments verbatim flips the sign silently. Not yet verified on BigQuery.
WITH events_with_previous_time AS (
    SELECT
        user_session,
        event_time,
        LAG(event_time, 1) OVER (
            PARTITION BY user_session ORDER BY event_time
        ) AS previous_event_time
    FROM events
),

events_with_gaps AS (
    SELECT
        *,
        DATE_DIFF('second', previous_event_time, event_time) AS gap_seconds
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
),

native_sessions AS (
    -- Session grain, definition one. One row per user_session as provided.
    SELECT COUNT(*) AS event_count
    FROM indexed_events
    GROUP BY user_session
),

cut_sessions AS (
    -- Session grain, definition two. The ONLY difference is the extra key.
    SELECT COUNT(*) AS event_count
    FROM indexed_events
    GROUP BY user_session, session_index
),

labelled_sessions AS (
    -- Stacking the definitions rather than widening the table keeps one row per
    -- session throughout, so the bucketing below is written once, not twice.
    SELECT
        'user_session (as provided)' AS session_definition,
        event_count
    FROM native_sessions
    UNION ALL
    SELECT
        '30-minute inactivity timeout' AS session_definition,
        event_count
    FROM cut_sessions
),

sessions_by_event_count AS (
    -- Bucket grain: one row per (definition, session length).
    SELECT
        session_definition,
        event_count,
        COUNT(*) AS session_count
    FROM labelled_sessions
    GROUP BY session_definition, event_count
)

SELECT
    session_definition,
    event_count,
    session_count,
    ROUND(
        session_count * 100.0
        / SUM(session_count) OVER (PARTITION BY session_definition), 4
    ) AS session_share_pct,
    ROUND(
        SUM(session_count) OVER (
            PARTITION BY session_definition ORDER BY event_count
        ) * 100.0
        / SUM(session_count) OVER (PARTITION BY session_definition), 4
    ) AS cumulative_share_pct
FROM sessions_by_event_count
ORDER BY session_definition, event_count;
