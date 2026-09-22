-- Weekly cohorts split by the DAY of the first purchase, with k = 1 retention.
--
-- GRAIN IS ONE ROW PER first_purchase_date, 60 rows: every day of the window
-- except 2019-11-15, which has no first purchase because it has no purchase at
-- all (data_quality_daily_event_anomalies.sql). Its absence from this output
-- is the defect showing through, not a filter.
--
-- WHY SPLIT A COHORT BY DAY. weekly_cohort_retention.sql reads the 11-11
-- cohort as the worst retainer in the table, k = 1 at 7.17, and attributes it
-- to promotion-week newcomers. That reading compares cohorts to each other,
-- and cohorts differ in two ways at once: who they are, and which calendar
-- week their k = 1 falls in. Splitting one cohort by first-purchase day fixes
-- the second: every member of the 11-11 cohort has k = 1 in the same week,
-- 11-18, so a difference between days is a difference between people.
--
-- WHAT IT SHOWS FOR 11-11: members who first bought on 11-11..11-14 retain at
-- 8.16 to 9.59, members who first bought on the 11-16 and 11-17 promotion peak
-- retain at 7.12 and 6.43, and the peak days are 76.5% of the cohort. The
-- cohort-level 7.17 is mostly the weight of those two days. The original
-- reading survives a sharper test than the one that produced it.
--
-- WHAT THE OUTAGE DOES TO THIS COHORT: its denominator is too small. Anyone
-- whose first purchase fell in the 38.4-hour blackout has no first purchase on
-- record, so they are missing from the cohort unless they bought again later
-- the same week. 226,123 users carted during the blackout and 15.72% of them
-- have no recorded purchase by the time it ends; over the same-length window
-- immediately after it, equally busy at 231,507 carting users, that figure is
-- 2.29%. The 13.4pp excess is roughly 30,000 users, which is the order of the
-- undercount against a cohort of 146,154. A missing member is a missing
-- denominator, so the true retention of this cohort is at or below 7.17 --
-- correcting it would deepen the finding, not remove it. The estimate is only
-- an order of magnitude: the control window sits right after the blackout and
-- absorbs demand deferred from it, which flatters the control.
--
-- THE 11-18 COHORT INHERITS THE OTHER HALF. Users whose blackout purchase went
-- unrecorded and who bought again in a later week enter the table as new
-- customers of that later cohort, when they are really returning ones. That
-- inflates 11-18 (k = 1 of 11.36 against the 11-11 cohort's 7.17), so the
-- contrast between those two cohorts is overstated by an unknown amount.
--
-- K IS DEFINED ON WEEKS, NOT ON DAYS. The split is only on how members are
-- grouped for reporting; retention is still "bought again in the calendar week
-- after the cohort week", identical to weekly_cohort_retention.sql. Two users
-- who first bought on 11-11 and 11-14 share a k = 1 window, which is what
-- makes the columns comparable.
--
-- THE LAST COHORT READS 0.0, WHICH IS CENSORING, NOT BEHAVIOUR. Members whose
-- first purchase fell in the 11-25 week have their k = 1 window in December,
-- outside the data, so every one of those days reports 0 retained. The same
-- caveat as the empty lower-right triangle in weekly_cohort_retention.sql.
--
-- REGRESSION VALUES, --full, 60 rows; the 11-11 cohort in full:
--   first_purchase_date  members  k1_retention_pct
--   2019-11-11             9,527            8.1558
--   2019-11-12             8,426            8.7705
--   2019-11-13             8,269            9.5900
--   2019-11-14             8,194            9.4093
--   2019-11-15                 -                 -   <- outage, no row
--   2019-11-16            32,399            7.1237
--   2019-11-17            79,339            6.4256
--   sum of members = 146,154, the cohort size in weekly_cohort_retention.sql
WITH purchase_weeks AS (
    SELECT
        user_id,
        TIMESTAMP_TRUNC(event_time, ISOWEEK) AS purchase_week,
        MIN(event_time) OVER (PARTITION BY user_id) AS first_purchase_time,
        TIMESTAMP_TRUNC(MIN(event_time) OVER (PARTITION BY user_id), ISOWEEK)
            AS first_purchase_week
    FROM events
    WHERE event_type = 'purchase'
),

-- One row per user: which day they arrived, and whether they came back in the
-- following calendar week. 604800 seconds is one week; both sides are already
-- truncated to Monday 00:00, so the difference is an exact multiple of it.
user_returns AS (
    SELECT
        user_id,
        first_purchase_week,
        CAST(first_purchase_time AS DATE) AS first_purchase_date,
        MAX(
            CASE
                WHEN
                    TIMESTAMP_DIFF(purchase_week, first_purchase_week, SECOND) = 604800
                    THEN 1
                ELSE 0
            END
        ) AS returned_k1
    FROM purchase_weeks
    GROUP BY user_id, first_purchase_week, CAST(first_purchase_time AS DATE)
)

SELECT
    first_purchase_date,
    CAST(first_purchase_week AS DATE) AS cohort_week,
    COUNT(*) AS members,
    SUM(returned_k1) AS returned_k1,
    ROUND(SUM(returned_k1) * 100.0 / COUNT(*), 4) AS k1_retention_pct
FROM user_returns
GROUP BY first_purchase_date, first_purchase_week
ORDER BY first_purchase_date;
