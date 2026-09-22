-- Purchase sessions split by whether they hold a view and a cart.
--
-- GRAIN IS ONE ROW PER (has_view, has_cart) over cut sessions holding a
-- purchase: four rows that sum to every purchase session. It explains why the
-- strict funnel in session_funnel.sql keeps only two thirds of them.
--
-- On the full data:
--
--   has_view  has_cart  sessions   share    later segment  earlier cart
--   1         1         938,451    66.75%   1.57%          99.08%
--   1         0         459,984    32.72%   1.64%           9.64%
--   0         1           3,286     0.23%  11.84%          99.76%
--   0         0           4,233     0.30%   7.04%          40.42%
--
-- THE (1, 0) ROW IS THE WHOLE GAP: purchase sessions with a view and no cart
-- are 98% of what the strict funnel drops. Two explanations were tested:
--
--   1. The cut split one visit, leaving the cart in the previous segment of the
--      same user_session. new_cut_session_cnt counts sessions that are not the
--      first segment (new_index > 0). It is 1.64% of the (1, 0) row against
--      1.57% for (1, 1), which carts in-session and so needs no explanation --
--      no excess, so the cut is not the cause. The count is an upper bound
--      anyway: being a later segment does not mean the earlier one carted.
--   2. The cart happened in an earlier session, possibly under another
--      user_session. has_earlier_cart_cnt counts sessions whose purchase has a
--      cart of the same (user_id, product_id) at or before it, at any distance.
--      A (1, 0) session holds no cart itself, so any such cart is necessarily
--      in an earlier session. It covers 9.64% of the row.
--
-- The rest -- 415,635 sessions, 29.56% of all purchase sessions -- bought a
-- product the user never carted anywhere in the 61 days. These purchases do not
-- pass through a cart event, which is why the strict funnel is not the path
-- most purchases take in this data.
--
-- has_earlier_cart OVERSTATES AND UNDERSTATES, both slightly.
--   - Left truncation: a purchase on 2019-10-01 cannot see a September cart, so
--     early-October purchases look cartless. This pushes the 29.56% up.
--   - Consumed carts: a cart already followed by an earlier purchase of the same
--     product still counts. This pushes has_earlier_cart up.
--   - Same product only: a cart of a different product does not count, by
--     design -- the question is whether this purchase was carted first.
-- For scale, among the 1,149,955 purchase events with an earlier cart of the
-- same product, the latest such cart precedes the purchase by a median of 1.2
-- minutes and 95.87% of them by 30 minutes or less; 99% fall within 6.2 days.
-- Carts that convert mostly convert in the same visit, so a longer window would
-- recover little.
--
-- THE DEVELOPMENT SAMPLE DOES NOT REPRESENT THIS. On 2019-10-01..07 the (1, 0)
-- row is 55.8% of purchase sessions (32.72% on the full data) and its earlier-
-- cart share is 5.23% (9.64%). The first gap is drift over time; the second is
-- also the sample's seven-day look-back. Read only the full-data figures.
--
-- THE WINDOW ON (user_id, product_id) sorts all 110 million rows a second time.
-- It sits in the second CTE because it reads is_cart and is_purchase, which are
-- defined in the first; referencing them in the same SELECT runs on DuckDB and
-- fails on BigQuery. With no frame clause the default frame includes rows that
-- tie on event_time, so a cart in the same second as the purchase counts as
-- earlier, deterministically; a ROWS frame would decide ties at random.
--
-- CHECKS: the four rows sum to 1,405,954, the purchase-session count in
-- docs/sessionization.md; the (1, 1) row equals the strict purchase step in
-- session_funnel.sql (938,451); purchase_session_pct sums to 100.
--
-- SESSIONS ARE CUT ON THE 30-MINUTE TIMEOUT, as in session_funnel.sql.
--
-- DIALECT: written in BigQuery form, TIMESTAMP_DIFF(end, start, SECOND) with
-- end first. DuckDB has no TIMESTAMP_DIFF; scripts/run_sql.py defines it as a
-- macro that only honours SECOND (docs/sql_dialects.md).
WITH previous_event_times AS (
    SELECT
        event_type,
        event_time,
        user_id,
        product_id,
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
        TIMESTAMP_DIFF(event_time, previous_event_time, SECOND) AS gap_second,

        CASE
            WHEN is_purchase = 1
                THEN MAX(is_cart) OVER (PARTITION BY user_id, product_id ORDER BY event_time)
        END
            AS before_purchase_has_cart

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
        before_purchase_has_cart,

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
        before_purchase_has_cart,

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
        MAX(before_purchase_has_cart) AS has_earlier_cart

    FROM
        new_indexs
    GROUP BY
        user_session, new_index
)

SELECT
    has_view,
    has_cart,

    COUNT(*) AS new_session_cnt,
    COUNT(CASE WHEN new_index > 0 THEN 1 END) AS new_cut_session_cnt,
    COUNT(CASE WHEN has_earlier_cart = 1 THEN 1 END) AS has_earlier_cart_cnt,

    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 4) AS purchase_session_pct

FROM
    cart_purchase_indexs
WHERE
    has_purchase = 1
GROUP BY
    has_view, has_cart
ORDER BY
    has_view DESC, has_cart DESC;
