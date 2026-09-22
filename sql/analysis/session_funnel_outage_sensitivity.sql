-- Does the 2019-11-15 outage move the session funnel?
--
-- GRAIN IS ONE ROW PER STEP, the four steps of session_funnel.sql, each
-- reported twice: over every cut session, and over the sessions that do not
-- touch the outage.
--
-- THE OUTAGE IS A PURCHASE-SIDE BLACKOUT, 2019-11-14 18:09:37 to 2019-11-16
-- 08:32:34 UTC, 38.4 hours with no purchase event recorded while views and
-- carts ran at a record rate (data_quality_daily_event_anomalies.sql). A session
-- inside it cannot reach the purchase step whatever it did, so the purchase
-- step is understated under both definitions. View and cart were tracked
-- normally, so those steps are not biased by the outage itself.
--
-- THE UNIT REMOVED IS A SESSION, as in cart_abandonment_outage_sensitivity.sql
-- and for the same reasons: the filter is written on session_start and
-- session_end after the 30-minute cut, never on event_time, and overlap rather
-- than containment decides it. Here the funnel grain is already the session, so
-- the bounds and the flag come out of the same GROUP BY with no join back.
--
-- THIS IS NOT AN OVERCORRECTION, unlike repurchase_interval_outage_sensitivity.
-- There, dropping intervals that span the hole selects on the outcome (long
-- intervals go). Here a session is dropped for when it happened, not for what
-- it did, so the kept columns are an unbiased funnel for the kept population.
-- What changes is the population: the dropped hours are the start of the
-- 11/15-17 traffic spike, so the kept columns describe the 61 days minus that
-- stretch. The view and cart rows measure that composition shift and nothing
-- else, since the outage did not touch those events.
--
-- REGRESSION VALUES, --full, 4 rows. The raw columns reproduce the header of
-- session_funnel.sql digit for digit, which is what says the pipeline is the
-- same one. On the development sample (10/01-07) raw and kept are identical,
-- since no sample session is near the outage.
--   step      loose_raw   loose_kept  strict_raw  strict_kept  loose %          strict step %
--   session   23,517,313  22,258,942  23,517,313  22,258,942   100 / 100       -
--   view      23,501,913  22,244,816  23,501,913  22,244,816   99.93 / 99.94   99.93 / 99.94
--   cart       2,331,376   1,942,212   2,320,209   1,932,319    9.91 / 8.73     9.87 / 8.69
--   purchase   1,405,954   1,404,633     938,451     937,236    5.98 / 6.31    40.45 / 48.50
--
-- WHAT IT SAYS: the headline rates barely move, the cart -> purchase step does.
-- 1,258,371 sessions are dropped (5.35%). Loose purchase goes from 5.98% to
-- 6.31%, and strict through the funnel from 3.99% to 4.21%. But strict cart ->
-- purchase goes from 40.45% to 48.50%, 8.06pp: the dropped sessions carted at
-- 30.93% (387,890 strict carts) and converted at 0.31% (1,215), so they add a
-- block of carts with no purchases to the raw denominator. Quote 48.50% for the
-- step, and 40.45% only as what the outage does to it.
--
-- NOT EXPLAINED: why the outage sessions cart at 30.93% against 8.73% for the
-- rest. The spike explains some of it (11-15 carted at 7.8% of events against
-- 3.6% over the window), but so would shoppers re-adding items to a cart they
-- could not check out. The data cannot tell those apart.
--
-- Only 1,321 purchase sessions are dropped, 0.094% of them, so the four-row
-- split in session_funnel_purchase_decomposition.sql (conclusion 17) can move
-- by at most that much. The purchases the outage never recorded are missing
-- from both columns here and cannot be recovered by any filter.
--
-- DIALECT: written in BigQuery form, TIMESTAMP_DIFF(end, start, SECOND) with
-- end first. DuckDB has no TIMESTAMP_DIFF; scripts/run_sql.py defines it as a
-- macro that only honours SECOND (docs/sql_dialects.md).
WITH previous_event_times AS (
    SELECT
        event_time,
        user_session,
        LAG(event_time, 1)
            OVER (PARTITION BY user_session ORDER BY event_time)
            AS previous_event_time,
        CASE WHEN event_type = 'view' THEN 1 ELSE 0 END AS is_view,
        CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END AS is_cart,
        CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END AS is_purchase
    FROM events
),

is_new_sessions AS (
    SELECT
        event_time,
        user_session,
        is_view,
        is_cart,
        is_purchase,
        CASE
            WHEN TIMESTAMP_DIFF(event_time, previous_event_time, SECOND) > 1800 THEN 1
            ELSE 0
        END AS is_new_session
    FROM previous_event_times
),

session_events AS (
    SELECT
        event_time,
        user_session,
        is_view,
        is_cart,
        is_purchase,
        SUM(is_new_session) OVER (PARTITION BY user_session ORDER BY event_time) AS new_index
    FROM is_new_sessions
),

sessions AS (
    SELECT
        MAX(is_view) AS has_view,
        MAX(is_cart) AS has_cart,
        MAX(is_purchase) AS has_purchase,
        CASE
            WHEN MAX(is_view) = 1 AND MAX(is_cart) = 1 THEN 1
            ELSE 0
        END AS has_view_and_cart,
        CASE
            WHEN MAX(is_view) = 1 AND MAX(is_cart) = 1 AND MAX(is_purchase) = 1 THEN 1
            ELSE 0
        END AS has_view_and_cart_and_purchase,
        CASE
            WHEN
                MIN(event_time) < TIMESTAMP '2019-11-16 08:32:34'
                AND MAX(event_time) > TIMESTAMP '2019-11-14 18:09:37'
                THEN 0
            ELSE 1
        END AS is_kept
    FROM session_events
    GROUP BY user_session, new_index
),

funnel_steps AS (
    SELECT
        0 AS step_order,
        'session' AS step,
        COUNT(*) AS loose_raw,
        COUNT(*) AS strict_raw,
        SUM(is_kept) AS loose_kept,
        SUM(is_kept) AS strict_kept
    FROM sessions

    UNION ALL

    SELECT
        1 AS step_order,
        'view' AS step,
        SUM(has_view) AS loose_raw,
        SUM(has_view) AS strict_raw,
        SUM(has_view * is_kept) AS loose_kept,
        SUM(has_view * is_kept) AS strict_kept
    FROM sessions

    UNION ALL

    SELECT
        2 AS step_order,
        'cart' AS step,
        SUM(has_cart) AS loose_raw,
        SUM(has_view_and_cart) AS strict_raw,
        SUM(has_cart * is_kept) AS loose_kept,
        SUM(has_view_and_cart * is_kept) AS strict_kept
    FROM sessions

    UNION ALL

    SELECT
        3 AS step_order,
        'purchase' AS step,
        SUM(has_purchase) AS loose_raw,
        SUM(has_view_and_cart_and_purchase) AS strict_raw,
        SUM(has_purchase * is_kept) AS loose_kept,
        SUM(has_view_and_cart_and_purchase * is_kept) AS strict_kept
    FROM sessions
)

SELECT
    step_order,
    step,
    loose_raw,
    loose_kept,
    strict_raw,
    strict_kept,
    ROUND(loose_raw * 100.0 / MAX(loose_raw) OVER (), 4) AS loose_raw_pct,
    ROUND(loose_kept * 100.0 / MAX(loose_kept) OVER (), 4) AS loose_corrected_pct,
    ROUND(strict_raw * 100.0 / LAG(strict_raw) OVER (ORDER BY step_order), 4)
        AS strict_step_raw_pct,
    ROUND(strict_kept * 100.0 / LAG(strict_kept) OVER (ORDER BY step_order), 4)
        AS strict_step_corrected_pct
FROM funnel_steps
ORDER BY step_order;
