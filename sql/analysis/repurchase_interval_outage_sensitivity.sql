-- Does the 2019-11-15 outage move "80% of repurchases happen within N days"?
--
-- GRAIN IS ONE ROW PER (definition, threshold): two interval definitions from
-- repurchase_interval_distribution.sql against five day thresholds, each
-- reported over all intervals and over the intervals the outage cannot have
-- touched.
--
-- WHY THIS METRIC AND NOT ANOTHER. Conclusion 16 -- 80% of repurchases inside 6
-- days at event level, 12 at occasion level -- is one of the two headline
-- findings of P1, and it is an interval metric, which is exactly what a hole in
-- the timeline distorts. Purchase tracking was down for 38.4 hours from
-- 2019-11-14 18:09:37 UTC (data_quality_daily_event_anomalies.sql). A purchase
-- inside that hole is not recorded, so the interval it would have split is
-- reported as one long gap instead of two short ones. The bias has a direction:
-- intervals are overstated, so the true share within N days is at least what
-- the raw column says.
--
-- THE UNIT REMOVED IS AN INTERVAL, and an interval is dropped when its own span
-- overlaps the hole: previous purchase before the hole ends, current purchase
-- after it starts. Not the purchases themselves -- a purchase before the hole
-- is real data and anchors an interval that ends before the hole begins. This
-- is the same overlap test as cart_abandonment_outage_sensitivity.sql, one
-- grain up: there the corrupted unit was a session, here it is a gap.
--
-- SHARES, NOT PERCENTILES. The question is "what share falls within N days",
-- which is a counted threshold and spells the same on both engines. A
-- percentile would need APPROX_QUANTILES on BigQuery against quantile_cont on
-- DuckDB, with no common spelling and no shim (docs/sql_dialects.md).
--
-- 6 AND 12 DAYS ARE IN THE LIST because they are the two figures conclusion 16
-- quotes; 1, 3 and 30 are there so the shape of the curve is visible and a
-- change at 6 can be read against its neighbours.
--
-- THE CORRECTION OVERSHOOTS, SO READ A RANGE AND NOT A NUMBER. Dropping every
-- interval that spans the hole also drops intervals that were genuinely long
-- and merely happened to cross 2019-11-15, and those are long by definition. So
-- the raw column is biased low on "share within N days" and the corrected
-- column is biased high, and the truth sits between them. Measured on the full
-- data, the 80th percentile of the gap is 5.96 days raw against 3.63 corrected
-- at event level, and 10.88 against 6.95 at occasion level.
--
-- WHAT SURVIVES: the ratio between the two definitions. 10.88 / 5.96 = 1.83,
-- 6.95 / 3.63 = 1.91 -- conclusion 16's point that the two definitions answer
-- the same question a factor of two apart holds at either end of the range,
-- while its absolute figures (6 and 12 days) are the upper bound rather than
-- the estimate.
--
-- REGRESSION VALUES, --full, 10 rows. 81,419 intervals are dropped under BOTH
-- definitions, which is not a copied number: exactly one interval per user
-- bridges the hole, and it is the same set of users either way, since every
-- gap spanning 38.4 hours also exceeds the 1,800-second occasion threshold.
--   definition         days  raw %     corrected %
--   purchase_event        1  59.2165   64.6897
--   purchase_event        3  71.4848   77.8007
--   purchase_event        6  80.1451   86.3831
--   purchase_event       12  88.6006   93.8902
--   purchase_event       30  97.1390   99.3689
--   shopping_occasion     1  35.3730   40.8449
--   shopping_occasion     3  54.8357   62.8315
--   shopping_occasion     6  68.5569   77.2052
--   shopping_occasion    12  81.9489   89.7732
--   shopping_occasion    30  95.4697   98.9438
-- Interval totals: 962,318 event level, 607,754 occasion level.
WITH previous_purchase_times AS (
    SELECT
        user_id,
        event_time,
        LAG(event_time, 1) OVER (
            PARTITION BY user_id ORDER BY event_time
        ) AS previous_purchase_time
    FROM events
    WHERE event_type = 'purchase'
),

purchase_gaps AS (
    SELECT
        user_id,
        event_time,
        previous_purchase_time,
        TIMESTAMP_DIFF(event_time, previous_purchase_time, SECOND) AS gap_seconds
    FROM previous_purchase_times
),

occasion_indexes AS (
    SELECT
        user_id,
        event_time,
        SUM(CASE WHEN gap_seconds > 1800 THEN 1 ELSE 0 END) OVER (
            PARTITION BY user_id ORDER BY event_time
        ) AS occasion_index
    FROM purchase_gaps
),

occasion_starts AS (
    SELECT
        user_id,
        occasion_index,
        MIN(event_time) AS occasion_start
    FROM occasion_indexes
    GROUP BY user_id, occasion_index
),

occasion_gaps AS (
    SELECT
        user_id,
        occasion_start,
        LAG(occasion_start, 1) OVER (
            PARTITION BY user_id ORDER BY occasion_start
        ) AS previous_occasion_start,
        TIMESTAMP_DIFF(
            occasion_start,
            LAG(occasion_start, 1) OVER (
                PARTITION BY user_id ORDER BY occasion_start
            ),
            SECOND
        ) AS gap_seconds
    FROM occasion_starts
),

-- One row per interval, carrying both ends so the overlap test can be written
-- once below rather than twice above.
labelled_gaps AS (
    SELECT
        'purchase_event' AS gap_definition,
        previous_purchase_time AS interval_start,
        event_time AS interval_end,
        gap_seconds
    FROM purchase_gaps
    WHERE gap_seconds IS NOT NULL
    UNION ALL
    SELECT
        'shopping_occasion' AS gap_definition,
        previous_occasion_start AS interval_start,
        occasion_start AS interval_end,
        gap_seconds
    FROM occasion_gaps
    WHERE gap_seconds IS NOT NULL
),

flagged_gaps AS (
    SELECT
        gap_definition,
        gap_seconds,
        CASE
            WHEN
                interval_start < TIMESTAMP '2019-11-16 08:32:34'
                AND interval_end > TIMESTAMP '2019-11-14 18:09:37'
                THEN 1
            ELSE 0
        END AS spans_outage
    FROM labelled_gaps
),

-- CROSS JOIN rather than five near-identical CASE columns: a threshold is a
-- row, so adding one is adding a value, not another branch.
--
-- DIALECT: written as UNION ALL rather than FROM UNNEST([...]) AS threshold_days,
-- which binds differently on the two engines: on BigQuery the alias names the
-- value, on DuckDB it names a struct, so the arithmetic below fails there with
-- "No function matches ... STRUCT(unnest INTEGER)". UNION ALL of literals is
-- plain and identical on both.
thresholds AS (
    SELECT 1 AS threshold_days
    UNION ALL
    SELECT 3 AS threshold_days
    UNION ALL
    SELECT 6 AS threshold_days
    UNION ALL
    SELECT 12 AS threshold_days
    UNION ALL
    SELECT 30 AS threshold_days
)

SELECT
    flagged_gaps.gap_definition,
    thresholds.threshold_days,
    COUNT(*) AS intervals_total,
    COUNTIF(flagged_gaps.spans_outage = 1) AS intervals_dropped,
    ROUND(
        COUNTIF(flagged_gaps.gap_seconds <= thresholds.threshold_days * 86400) * 100.0
        / COUNT(*),
        4
    ) AS within_raw_pct,
    ROUND(
        COUNTIF(
            flagged_gaps.spans_outage = 0
            AND flagged_gaps.gap_seconds <= thresholds.threshold_days * 86400
        ) * 100.0
        / NULLIF(COUNTIF(flagged_gaps.spans_outage = 0), 0),
        4
    ) AS within_corrected_pct
FROM flagged_gaps
CROSS JOIN thresholds
GROUP BY flagged_gaps.gap_definition, thresholds.threshold_days
ORDER BY flagged_gaps.gap_definition, thresholds.threshold_days;
