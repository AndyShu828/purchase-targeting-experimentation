-- Session-level funnel: view -> cart -> purchase, under two definitions.
--
-- GRAIN IS ONE ROW PER STEP. Four steps (session, view, cart, purchase), each
-- counted in cut sessions, side by side under two definitions:
--
--   loose   a session reaches a step if it holds that event type at all
--   strict  a session reaches a step only if it also holds every earlier step
--
-- "REACHES" MEANS "HOLDS", NOT "IN THAT ORDER". MAX() over the session says an
-- event type occurred, not when; a session that carts and then views still
-- reaches cart under both definitions.
--
-- THE TWO DEFINITIONS PART AT PURCHASE, NOT BEFORE. On the full data:
--
--   step       loose          strict
--   session    23,517,313     23,517,313
--   view       23,501,913     23,501,913
--   cart        2,331,376      2,320,209
--   purchase    1,405,954        938,451
--
-- The 11,167 cart sessions with no view are 0.48% of cart sessions. The
-- purchase step is where it matters: strict drops 467,503 purchase sessions,
-- a third of them, because they hold no cart event. That is not a quirk of the
-- cut -- session_funnel_purchase_decomposition.sql shows 415,635 purchase
-- sessions never carted the purchased product anywhere in the data. Loose is
-- the session purchase rate (5.98%); strict is the rate through the funnel
-- (3.99%). Report both, and never the strict rate as "the purchase rate".
--
-- THE STEP CONVERSION IS STRICT ONLY. strict_step_conversion_pct divides each
-- step by the one before, which is a conversion rate only because every strict
-- step is a subset of the previous one. The loose steps are not nested -- the
-- 467,503 purchase sessions above are not in the cart step -- so loose
-- purchase / loose cart (60.3%) would be a ratio of two overlapping sets, not
-- anyone's conversion rate. The column is deliberately absent.
--
-- CHECKS: step 0 is the cut session count in docs/sessionization.md
-- (23,517,313), loose is never below strict at any step, and strict never rises
-- from one step to the next. Loose purchase equals the purchase-session count
-- in the same document (1,405,954).
--
-- SESSIONS ARE CUT ON THE 30-MINUTE TIMEOUT rather than taken from user_session
-- (conclusion 5, docs/sessionization.md). The first four CTEs are the pipeline
-- from category_conversion_rate.sql, with category_id dropped: the grain here
-- is the session, not the (session, category) pair, so the denominator is
-- sessions and the "denominators sum to more than the session count" caveat in
-- that file does not apply.
--
-- DIALECT: written in BigQuery form, TIMESTAMP_DIFF(end, start, SECOND) with
-- end first. DuckDB has no TIMESTAMP_DIFF; scripts/run_sql.py defines it as a
-- macro that only honours SECOND (docs/sql_dialects.md).
WITH previous_event_times AS (
    SELECT
        event_type,
        event_time,
        user_session,
        LAG(event_time, 1)
            OVER (PARTITION BY user_session ORDER BY event_time)
            AS previous_event_time,

        CASE WHEN
            event_type = 'view'
            THEN 1
        ELSE 0 END
            AS is_view,

        CASE WHEN
            event_type = 'cart'
            THEN 1
        ELSE 0 END
            AS is_cart,

        CASE WHEN
            event_type = 'purchase'
            THEN 1
        ELSE 0 END
            AS is_purchase

    FROM
        events
),

gap_seconds AS (
    SELECT
        event_time,
        user_session,
        is_view,
        is_cart,
        is_purchase,
        TIMESTAMP_DIFF(event_time, previous_event_time, SECOND) AS gap_second
    FROM
        previous_event_times
),

is_new_sessions AS (
    SELECT
        event_time,
        user_session,
        is_view,
        is_cart,
        is_purchase,

        CASE WHEN
            gap_second > 1800
            THEN 1
        ELSE 0 END
            AS is_new_session
    FROM
        gap_seconds
),

new_indexs AS (
    SELECT
        event_time,
        user_session,
        is_view,
        is_cart,
        is_purchase,

        SUM(is_new_session) OVER (PARTITION BY user_session ORDER BY event_time) AS new_index

    FROM
        is_new_sessions
),

cart_purchase_indexs AS (
    SELECT
        user_session,
        new_index,

        MAX(is_view) AS has_view,
        MAX(is_cart) AS has_cart,
        MAX(is_purchase) AS has_purchase,

        CASE WHEN
            MAX(is_view) = 1 AND MAX(is_cart) = 1
            THEN 1
        ELSE 0 END
            AS has_view_and_cart,

        CASE WHEN
            MAX(is_view) = 1 AND MAX(is_cart) = 1 AND MAX(is_purchase) = 1
            THEN 1
        ELSE 0 END
            AS has_view_and_cart_and_purchase

    FROM
        new_indexs
    GROUP BY
        user_session, new_index
),

funnel_steps AS (
    SELECT
        0 AS step_order,
        'session' AS step,
        COUNT(*) AS loose_session_count,
        COUNT(*) AS strict_session_count
    FROM
        cart_purchase_indexs

    UNION ALL

    SELECT
        1 AS step_order,
        'view' AS step,
        SUM(has_view) AS loose_session_count,
        SUM(has_view) AS strict_session_count
    FROM
        cart_purchase_indexs

    UNION ALL

    SELECT
        2 AS step_order,
        'cart' AS step,
        SUM(has_cart) AS loose_session_count,
        SUM(has_view_and_cart) AS strict_session_count
    FROM
        cart_purchase_indexs

    UNION ALL

    SELECT
        3 AS step_order,
        'purchase' AS step,
        SUM(has_purchase) AS loose_session_count,
        SUM(has_view_and_cart_and_purchase) AS strict_session_count
    FROM
        cart_purchase_indexs
)

SELECT
    step_order,
    step,

    loose_session_count,
    strict_session_count,

    ROUND(loose_session_count * 100.0 / MAX(loose_session_count) OVER (), 4) AS loose_pct,
    ROUND(strict_session_count * 100.0 / MAX(strict_session_count) OVER (), 4) AS strict_pct,
    ROUND(
        strict_session_count * 100.0 / LAG(strict_session_count) OVER (ORDER BY step_order),
        4)
        AS strict_step_conversion_pct

FROM
    funnel_steps
ORDER BY
    step_order;
