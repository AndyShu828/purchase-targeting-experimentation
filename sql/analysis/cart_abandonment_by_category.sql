-- Cart abandonment by category.
--
-- GRAIN IS ONE ROW PER category_id. The denominator is the number of (cut
-- session, category) pairs that contain a cart event; the numerator is those
-- with no purchase event in the same category in the same session.
--
-- PAIRS WITHOUT A CART ARE FILTERED AT THE PAIR LAYER, not divided away. 44 of
-- the 691 categories have no cart pair at all, and dividing by their zero
-- denominator returns NULL in DuckDB but RAISES in BigQuery. Filtering early is
-- therefore not an optimisation, it is the only version that runs on BigQuery.
-- This direction of the dialect gap cannot be caught locally: here DuckDB is
-- the lenient one, the opposite of the SAFE_DIVIDE case in scripts/run_sql.py.
--
-- ABANDONMENT IS MEASURED WITHIN A SESSION and does not follow the user. A user
-- who carts category A on Monday and buys A on Wednesday leaves an abandoned
-- pair on Monday. A session that carts A and buys B counts A as abandoned,
-- which the pair grain makes deliberate -- the two categories are independent
-- rows -- but it means this rate is not "carted and then bought nothing".
--
-- THERE IS NO remove_from_cart EVENT (event_type_distribution.sql: only view,
-- cart and purchase), so "carted but did not buy" is literally the absence of a
-- purchase and needs no handling of items added and then removed.
--
-- CHECK: cart pairs summed over all categories came to 2,448,962 against
-- 3,955,446 cart events on the full data -- 1.62 cart events per pair, and it
-- can only ever be at least one, because collapsing removes rows and never adds
-- them. The same ratio on the development sample is 1.63.
--
-- THE RATE IS NOT CONSTANT OVER THE 61 DAYS. Cart pairs starting in
-- 2019-10-01..07 abandon at 49.02%, 2019-10-08..31 at 49.62%, 2019-11-01..14 at
-- 58.89% and 2019-11-15..30 at 65.00%, the last window holding 57% of all cart
-- pairs. The development sample is representative of October to within 0.6pp
-- and understates the full-data figure only because November is different.
-- Any baseline drawn for the 11/15..11/28 experiment window has to come from
-- November, not October.
--
-- is_reportable flags a numerator of at least 100, the same threshold and the
-- same reasoning as category_conversion_rate.sql: the tail is categories whose
-- rate is 0% or 100% by arithmetic rather than behaviour. It passes 379 of the
-- 647 emitted categories. Thresholding the denominator instead would pass 423;
-- the 44 categories in between all sit between 54% and 85% on denominators
-- under 200.
--
-- SESSIONS ARE CUT ON THE 30-MINUTE TIMEOUT rather than taken from user_session
-- (conclusion 5, docs/sessionization.md). The first four CTEs are the pipeline
-- from category_conversion_rate.sql.
--
-- LABELS ARE NOT JOINED HERE. is_code_null3 reports only whether the category
-- has a category_code at all; 256 of the 647 emitted categories do. Reading a
-- rate next to a 19-digit id needs category_conversion_rate.sql alongside.
--
-- DIALECT: written in BigQuery form, TIMESTAMP_DIFF(end, start, SECOND) with
-- end first. DuckDB has no TIMESTAMP_DIFF; scripts/run_sql.py defines it as a
-- macro that only honours SECOND (docs/sql_dialects.md).
WITH previous_event_times AS (
    SELECT
        event_type,
        event_time,
        user_session,
        category_id,
        category_code,
        LAG(event_time, 1)
            OVER (PARTITION BY user_session ORDER BY event_time)
            AS previous_event_time,

        CASE WHEN
            event_type = 'cart'
            THEN 1
        ELSE 0 END
            AS is_cart,

        CASE WHEN
            event_type = 'purchase'
            THEN 1
        ELSE 0 END
            AS is_purchase,

        CASE WHEN
            category_code IS NULL
            THEN 1
        ELSE 0 END
            AS is_code_null

    FROM
        events
),

gap_seconds AS (
    SELECT
        event_time,
        user_session,
        category_id,
        previous_event_time,
        is_cart,
        is_purchase,
        is_code_null,
        TIMESTAMP_DIFF(event_time, previous_event_time, SECOND) AS gap_second
    FROM
        previous_event_times
),

is_new_sessions AS (
    SELECT
        event_time,
        user_session,
        category_id,
        is_cart,
        is_purchase,
        is_code_null,
        gap_second,

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
        category_id,
        is_cart,
        is_purchase,
        is_code_null,
        is_new_session,

        SUM(is_new_session) OVER (PARTITION BY user_session ORDER BY event_time) AS new_index

    FROM
        is_new_sessions
),

cart_purchase_indexs AS (
    SELECT
        user_session,
        new_index,
        category_id,
        MAX(is_cart) AS has_cart,
        MAX(is_purchase) AS has_purchase,
        MAX(is_code_null) AS is_code_null2,
        CASE WHEN MAX(is_cart) = 1 AND MAX(is_purchase) = 0 THEN 1 END AS has_cart_no_purchase
    FROM
        new_indexs
    GROUP BY
        user_session, new_index, category_id
),

cart_no_purchase_rate_pcts AS (
    SELECT
        category_id,
        COUNT(CASE WHEN has_cart = 1 THEN 1 END) AS pairs_with_cart,
        COUNT(CASE WHEN has_cart_no_purchase = 1 THEN 1 END) AS pairs_cart_no_purchase,
        ROUND(
            COUNT(CASE WHEN has_cart_no_purchase = 1 THEN 1 END)
            * 100.0
            / COUNT(CASE WHEN has_cart = 1 THEN 1 END),
            4)
            AS cart_no_purchase_rate_pct,

        MAX(is_code_null2) AS is_code_null3,
        CASE WHEN COUNT(CASE WHEN has_cart_no_purchase = 1 THEN 1 END) >= 100 THEN 1 ELSE 0 END
            AS is_reportable

    FROM
        cart_purchase_indexs
    WHERE
        has_cart = 1
    GROUP BY
        category_id
)

SELECT

    *,
    CASE WHEN is_reportable = 1 THEN cart_no_purchase_rate_pct END
        AS reportable_cart_no_purchase_rate_pct

FROM
    cart_no_purchase_rate_pcts
ORDER BY
    cart_no_purchase_rate_pct;
