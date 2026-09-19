-- Daily active identities and event volume, one row per calendar day.
--
-- active_users and active_sessions are NOT the same quantity and the gap
-- between them is the point of reporting both. user_id is a stable person;
-- user_session is a cookie. Measured over the whole stream there are 5,316,649
-- of the former and 23,016,650 of the latter, so a person carries 4.33 cookies
-- over 61 days against 1.96 over the 7-day development sample. The two scale
-- differently: widening the window 8.71x multiplied events by 12.45, cookies by
-- 12.26 and people by only 5.55. Cookies are minted per unit of activity;
-- people saturate because the same ones are seen again. Name the column you
-- quote -- "daily active users" and "daily active cookies" differ by 4x here.
--
-- DO NOT SUM THE TWO DISTINCT COLUMNS ACROSS DAYS. COUNT(DISTINCT ...) is not
-- additive: the dedup happens inside each GROUP BY group and no group knows
-- what another saw, so a session live on two dates is counted on both. Summing
-- active_sessions over the full stream yields 23,109,633 against 23,016,650
-- actual, an excess of 92,983. That excess is sum(days_present - 1), not a
-- session count: 64,841 sessions (0.2817%) actually span two or more dates.
-- A period-level distinct count can only be recomputed from the base table.
--
-- unclassified_events is a reconciliation column and must be 0 on every row.
-- It can fail: it is the count of events whose type falls outside the
-- view/cart/purchase vocabulary established in event_type_distribution.sql, so
-- a fourth type appearing in a later load shows up here per day rather than
-- silently widening a denominator. event_count summed over the rows is the
-- second reconciliation: 8,829,315 on the sample, 109,950,743 on the full set.
--
-- PERFORMANCE: the GROUP BY key is the Parquet partition key, so each partition
-- is exactly one group -- no hashing and no cross-partition shuffle. This is
-- where the P0 decision to partition by event_date pays out.
SELECT
    event_date,
    COUNT(DISTINCT user_id) AS active_users,
    COUNT(DISTINCT user_session) AS active_sessions,
    COUNT(CASE WHEN event_type = 'view' THEN 1 END) AS view_events,
    COUNT(CASE WHEN event_type = 'cart' THEN 1 END) AS cart_events,
    COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) AS purchase_events,
    COUNT(*) AS event_count,
    COUNT(*) - COUNT(
        CASE WHEN event_type IN ('view', 'cart', 'purchase') THEN 1 END
    ) AS unclassified_events
FROM events
GROUP BY event_date
ORDER BY event_date;
