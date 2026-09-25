-- Intent feature group: the DuckDB side of src/pte/features/intent.py.
--
-- Andy's query (2026-09-25, view_to_cart_rate added 2026-09-26); Claude parameterised it
-- and formatted it for sqlfluff. Variables: cutoff, window_start. One row per population
-- user.
--
-- cart_abandon_rate = abandoned / cart (session, category) pairs, P1's unit
-- (cart_abandonment_by_category.sql). Pairs in sessions overlapping the purchase outage
-- are dropped; a user with no remaining cart pair gets NULL (decided 2026-09-25).
--
-- view_to_cart_rate = pairs with a view and a cart / pairs with a view, same unit, order
-- ignored. Outage pairs are KEPT (decided 2026-09-26): only features that use purchases
-- drop them, and the outage lost purchase records only. So the outage condition sits in
-- each count, not in a WHERE -- copying it into the view counts turns 128,750 users who
-- viewed only during the outage into NULLs at the test cutoff. No view pair -> NULL.
--
-- EVENTS ARE FILTERED TO THE WINDOW BEFORE SESSIONS ARE CUT, so a visit crossing the
-- cutoff ends there and its later purchase cannot leak into the feature.
--
-- SESSIONS ARE CUT WITHIN (user_id, user_session). P1 used user_session alone; 939
-- cookies carry more than one user, and a feature belongs to one.
--
-- THE BOUNDS JOIN IS NULL-SAFE. 12 events in the data have a NULL user_session, and
-- they are kept (decided 2026-09-25). With `=`, NULL = NULL is not true and the inner
-- join drops them silently -- 2 users here. The Spark version attaches the bounds with
-- window functions and has no such join.
WITH population AS (
    SELECT DISTINCT user_id
    FROM events
    WHERE
        event_date >= GETVARIABLE('cutoff') - INTERVAL 14 DAY AND event_date < GETVARIABLE('cutoff')
),

previous_event_times AS (
    SELECT
        user_id,
        event_time,
        user_session,
        category_id,
        LAG(event_time, 1)
            OVER (PARTITION BY user_session, user_id ORDER BY event_time)
            AS previous_event_time,
        CASE WHEN event_type = 'view' THEN 1 ELSE 0 END AS is_view,
        CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END AS is_cart,
        CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END AS is_purchase
    FROM events
    WHERE event_date >= GETVARIABLE('window_start') AND event_date < GETVARIABLE('cutoff')
),

is_new_sessions AS (
    SELECT
        user_id,
        event_time,
        user_session,
        category_id,
        is_view,
        is_cart,
        is_purchase,
        CASE
            WHEN DATE_DIFF('second', previous_event_time, event_time) > 1800 THEN 1
            ELSE 0
        END AS is_new_session
    FROM previous_event_times
),

session_events AS (
    SELECT
        user_id,
        event_time,
        user_session,
        category_id,
        is_view,
        is_cart,
        is_purchase,
        SUM(is_new_session)
            OVER (PARTITION BY user_session, user_id ORDER BY event_time)
            AS new_index
    FROM is_new_sessions
),

session_bounds AS (
    SELECT
        user_id,
        user_session,
        new_index,
        MIN(event_time) AS session_start,
        MAX(event_time) AS session_end
    FROM session_events
    GROUP BY user_session, new_index, user_id
),

session_category_pairs AS (
    SELECT
        session_events.user_id,
        session_events.user_session,
        session_events.new_index,
        session_events.category_id,
        MIN(session_bounds.session_start) AS session_start,
        MAX(session_events.is_view) AS has_view,
        MAX(session_events.is_cart) AS has_cart,
        MAX(session_events.is_purchase) AS has_purchase,
        MAX(
            CASE
                WHEN
                    session_bounds.session_start < TIMESTAMP '2019-11-16 08:32:34'
                    AND session_bounds.session_end > TIMESTAMP '2019-11-14 18:09:37'
                    THEN 1
                ELSE 0
            END
        ) AS is_outage_session
    FROM session_events
    INNER JOIN session_bounds
        ON
            session_events.user_session IS NOT DISTINCT FROM session_bounds.user_session
            AND session_events.new_index = session_bounds.new_index
            AND session_events.user_id = session_bounds.user_id
    GROUP BY
        session_events.user_session,
        session_events.new_index,
        session_events.category_id,
        session_events.user_id
),

user_pair_counts AS (
    SELECT
        user_id,
        COUNT(CASE WHEN is_outage_session = 0 AND has_cart = 1 THEN 1 END) AS cart_pairs,
        COUNT(
            CASE WHEN is_outage_session = 0 AND has_cart = 1 AND has_purchase = 0 THEN 1 END
        ) AS abandoned_pairs,
        COUNT(CASE WHEN has_view = 1 THEN 1 END) AS view_pairs,
        COUNT(CASE WHEN has_view = 1 AND has_cart = 1 THEN 1 END) AS view_cart_pairs
    FROM session_category_pairs
    GROUP BY user_id
)

SELECT
    p.user_id,
    c.abandoned_pairs * 1.0 / NULLIF(c.cart_pairs, 0) AS cart_abandon_rate,
    c.view_cart_pairs * 1.0 / NULLIF(c.view_pairs, 0) AS view_to_cart_rate
FROM population AS p LEFT JOIN user_pair_counts AS c ON p.user_id = c.user_id
