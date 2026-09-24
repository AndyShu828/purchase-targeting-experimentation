-- Purchase feature group: the DuckDB side of src/pte/features/purchase.py.
--
-- Andy's query (2026-09-25); Claude parameterised it and formatted it for sqlfluff.
-- Variables: cutoff, window_start, effective_days (window_days minus the overlap with
-- the 2019-11-14..16 purchase outage, from pte.features.outage.effective_days; 12.40 for
-- the 11-18 test window). One row per population user.
--
-- NO ELSE 0 INSIDE THE SUMS. A user with no purchase keeps a NULL spend in buyer and
-- gets 0 after the join. With ELSE 0 the first draft computed avg_price as 0 * 1.0 / 0,
-- which DuckDB returns as NaN -- not NULL, so COALESCE let 2,021,300 of them through.
-- NULLIF on the divisor stays as the second guard.
WITH population AS (
    SELECT DISTINCT user_id
    FROM events
    WHERE
        event_date >= GETVARIABLE('cutoff') - INTERVAL 14 DAY AND event_date < GETVARIABLE('cutoff')
),

buyer AS (
    SELECT
        user_id,
        COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) AS n_purchases,
        COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN event_date END) AS purchase_days,
        SUM(CASE WHEN event_type = 'purchase' THEN price END) AS total_spend,
        COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) * 1.0
        / GETVARIABLE('effective_days') AS n_purchases_per_day,
        COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN event_date END) * 1.0
        / GETVARIABLE('effective_days') AS purchase_days_per_day,
        SUM(CASE WHEN event_type = 'purchase' THEN price END) * 1.0
        / GETVARIABLE('effective_days') AS total_spend_per_day,
        SUM(CASE WHEN event_type = 'purchase' THEN price END) * 1.0
        / NULLIF(COUNT(CASE WHEN event_type = 'purchase' THEN 1 END), 0) AS avg_price,
        DATE_DIFF(
            'day', MAX(CASE WHEN event_type = 'purchase' THEN event_date END), GETVARIABLE('cutoff')
        ) AS days_since_last_purchase
    FROM events
    WHERE event_date >= GETVARIABLE('window_start') AND event_date < GETVARIABLE('cutoff')
    GROUP BY user_id
)

SELECT
    p.user_id,
    b.avg_price,
    b.days_since_last_purchase,
    COALESCE(b.n_purchases, 0) AS n_purchases,
    COALESCE(b.purchase_days, 0) AS purchase_days,
    COALESCE(b.total_spend, 0) AS total_spend,
    COALESCE(b.n_purchases_per_day, 0) AS n_purchases_per_day,
    COALESCE(b.purchase_days_per_day, 0) AS purchase_days_per_day,
    COALESCE(b.total_spend_per_day, 0) AS total_spend_per_day
FROM population AS p LEFT JOIN buyer AS b ON p.user_id = b.user_id
