-- Activity feature group: the DuckDB side of src/pte/features/activity.py.
--
-- Andy's query (2026-09-25); Claude parameterised it and formatted it for sqlfluff.
-- Run through scripts/bench_features.py, which sets the variables: cutoff (the first
-- label day), window_start (cutoff - window_days). One row per population user.
--
-- Population users with no event in the window get 0 counts and a NULL
-- days_since_last_event, so the counts are COALESCEd after the LEFT JOIN and the
-- recency is not (decided 2026-09-24).
WITH population AS (
    SELECT DISTINCT user_id
    FROM events
    WHERE
        event_date >= GETVARIABLE('cutoff') - INTERVAL 14 DAY AND event_date < GETVARIABLE('cutoff')
),

active AS (
    SELECT
        user_id,
        COUNT(*) AS n_events,
        COUNT(CASE WHEN event_type = 'view' THEN 1 END) AS n_views,
        COUNT(CASE WHEN event_type = 'cart' THEN 1 END) AS n_carts,
        COUNT(DISTINCT event_date) AS active_days,
        DATE_DIFF('day', MAX(event_date), GETVARIABLE('cutoff')) AS days_since_last_event
    FROM events
    WHERE event_date >= GETVARIABLE('window_start') AND event_date < GETVARIABLE('cutoff')
    GROUP BY user_id
)

SELECT
    p.user_id,
    b.days_since_last_event,
    COALESCE(b.n_events, 0) AS n_events,
    COALESCE(b.n_views, 0) AS n_views,
    COALESCE(b.n_carts, 0) AS n_carts,
    COALESCE(b.active_days, 0) AS active_days
FROM population AS p LEFT JOIN active AS b ON p.user_id = b.user_id
