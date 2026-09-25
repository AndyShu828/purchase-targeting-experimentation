-- Trend feature group: the DuckDB side of src/pte/features/trend.py.
--
-- Andy's query (2026-09-25); Claude fixed the variable references, renamed per the
-- naming rules and formatted it for sqlfluff. Run through scripts/bench_features.py,
-- which sets the variable cutoff (the first label day). One row per population user.
--
-- Plan v2: events in the last 7 days divided by events in the 7 days before, all event
-- types. Decided 2026-09-25 (option a): NULL when the previous 7 days have no event,
-- 0 when the last 7 days have none. Both counts come from one GROUP BY, so an empty
-- half counts 0 rather than turning NULL through a join, and NULLIF keeps DuckDB from
-- returning inf for x / 0 without an error.
WITH population AS (
    SELECT DISTINCT user_id
    FROM events
    WHERE
        event_date >= GETVARIABLE('cutoff') - INTERVAL 14 DAY AND event_date < GETVARIABLE('cutoff')
),

windowed_events AS (
    SELECT
        user_id,
        CASE
            WHEN event_date >= GETVARIABLE('cutoff') - INTERVAL 7 DAY THEN 1
        END AS is_recent,
        CASE
            WHEN event_date < GETVARIABLE('cutoff') - INTERVAL 7 DAY THEN 1
        END AS is_previous
    FROM events
    WHERE
        event_date >= GETVARIABLE('cutoff') - INTERVAL 14 DAY AND event_date < GETVARIABLE('cutoff')
),

user_event_counts AS (
    SELECT
        user_id,
        COUNT(is_recent) AS n_recent_events,
        COUNT(is_previous) AS n_previous_events
    FROM windowed_events
    GROUP BY user_id
)

SELECT
    p.user_id,
    c.n_recent_events * 1.0 / NULLIF(c.n_previous_events, 0) AS recent_to_previous_event_ratio
FROM population AS p LEFT JOIN user_event_counts AS c ON p.user_id = c.user_id
