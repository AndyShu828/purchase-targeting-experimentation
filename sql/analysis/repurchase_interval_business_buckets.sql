-- Distribution of the interval between consecutive purchases on business
-- boundaries (1 hour / 1 day / 3 / 7 / 14 / 30 / 61 days), under the same two
-- definitions as repurchase_interval_distribution.sql.
--
-- GRAIN IS ONE ROW PER (definition, bucket). This is the companion of the
-- power-of-two file: that one shows the SHAPE of the distribution, this one
-- answers "how long dormant before a coupon" in units a marketer acts on.
-- Everything up to labelled_gaps is copied verbatim from that file, so the two
-- stay comparable only while those CTEs are kept identical. They are two files
-- rather than one because they are two output tables of different grain.
--
-- BUCKETS ARE LEFT-OPEN, RIGHT-CLOSED: (low, high]. An interval of exactly one
-- hour is "within an hour", which is how the boundary reads in a campaign
-- rule, and it matches the occasion cut below, where a gap of exactly 1800
-- seconds stays inside the occasion. The power-of-two file is the opposite,
-- [low, high), so rows of the two tables do not line up at the boundaries.
--
-- ZERO GETS ITS OWN BUCKET AS A CONSEQUENCE, NOT A SPECIAL CASE. The first
-- interval is (0, 3600], which excludes 0, so the 85 zero-second intervals
-- need a bucket of their own. They are the known duplicate purchase rows
-- (see the companion file, artefact 1), and they stay in the denominator for
-- the same reason given there. Once duplicates are dropped upstream this row
-- should vanish exactly.
--
-- A CASE LADDER DOES NOT GUARANTEE "NO GAP, NO OVERLAP"; the arithmetic
-- buckets of the companion file did. The two halves are checked differently:
--   gap     -- ELSE 'UNBUCKETED' turns any row that falls between branches
--              into a visible row of its own. It has already paid for itself:
--              the 61-day bound was first typed as 5,070,400 (58.7 days), and
--              69 intervals per definition surfaced as UNBUCKETED. ELSE NULL
--              would have shown the same rows as a blank label.
--   overlap -- CASE takes the first matching branch, so an overlap never
--              double-counts; it silently moves rows into the wrong bucket.
--              Neither ELSE nor the totals can see that. Only boundary values
--              (3600, 86400, ...) checked against their intended bucket can.
-- The totals, 962,318 and 607,754, must equal the companion file's. They
-- guard the pipeline upstream of the ladder, not the ladder itself.
--
-- THE LAST BUCKET HAS AN UPPER BOUND ON PURPOSE. The longest observed
-- interval is 5,235,330 seconds (60.6 days), so on this window "> 30 days"
-- would return the same counts. The bound is kept so that extending the data
-- (2019-12) cannot silently widen the last bucket: intervals past 61 days
-- will land in UNBUCKETED and force the ladder to be revisited.
--
-- LABELS CARRY A NUMERIC PREFIX because the bucket column is VARCHAR and
-- sorts lexically: without it "14–30天" sorts before "1–3天", and the
-- cumulative share, whose window is ordered by bucket, accumulates in that
-- wrong order too. A single digit holds for up to nine buckets; a tenth needs
-- zero-padding ("01.") or it sorts before "2.".
--
-- THE OCCASION DEFINITION HAS NO ZERO BUCKET, and its "within an hour" bucket
-- holds only 1,801-3,600 seconds: the 30-minute cut removes everything below.
-- That bucket is split by the threshold, not by behaviour -- the same artefact
-- as bucket 10 in the companion file.
--
-- READING IT. Above one day the two definitions agree almost row for row
-- (118,060 against 118,285 at 1-3 days); the difference is concentrated in
-- the first hour, 383,649 against 25,984. The 80% line falls in the 3-7 day
-- bucket for purchase events and the 7-14 day bucket for occasions,
-- consistent with the 6 and 12 days read off the companion file.
--
-- RIGHT-CENSORED, like every interval metric here: run --full. Conclusion 14
-- found the 7-day sample shifting interval shares by up to 1.29x.
--
-- DIALECT: written in BigQuery form, TIMESTAMP_DIFF(end, start, SECOND) with
-- end first. DuckDB has no TIMESTAMP_DIFF; scripts/run_sql.py defines it as a
-- macro that only honours SECOND (docs/sql_dialects.md).
--
-- REGRESSION VALUES, full set, --full:
--   bucket         purchase_event  shopping_occasion
--   1. 0秒                     85                  -
--   2. 1小时内            383,649             25,984
--   3. 1小时–1天          186,117            188,997
--   4. 1–3天              118,060            118,285
--   5. 3–7天              102,049            102,099
--   6. 7–14天              79,891             79,914
--   7. 14–30天             64,935             64,942
--   8. 30–61天             27,532             27,533
--   total                 962,318            607,754
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
        TIMESTAMP_DIFF(event_time, previous_purchase_time, SECOND) AS gap_seconds
    FROM previous_purchase_times
),

-- A new occasion starts where and only where the gap exceeds the threshold.
-- A user's first purchase has a NULL gap, which is not > 1800, so it lands in
-- occasion 0 rather than opening a spurious one.
occasion_indexes AS (
    SELECT
        user_id,
        event_time,
        SUM(CASE WHEN gap_seconds > 1800 THEN 1 ELSE 0 END) OVER (
            PARTITION BY user_id ORDER BY event_time
        ) AS occasion_index
    FROM purchase_gaps
),

-- One row per occasion. The identity of an occasion is the pair, never the
-- index alone.
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
        TIMESTAMP_DIFF(
            occasion_start,
            LAG(occasion_start, 1) OVER (
                PARTITION BY user_id ORDER BY occasion_start
            ),
            SECOND
        ) AS gap_seconds
    FROM occasion_starts
),

-- One row per interval, labelled with the definition it came from. A NULL gap
-- is a user's first purchase or first occasion and is dropped here: it is not
-- an interval, and it is 697,470 rows at event level.
labelled_gaps AS (
    SELECT
        'purchase_event' AS gap_definition,
        gap_seconds
    FROM purchase_gaps
    WHERE gap_seconds IS NOT NULL
    UNION ALL
    SELECT
        'shopping_occasion' AS gap_definition,
        gap_seconds
    FROM occasion_gaps
    WHERE gap_seconds IS NOT NULL
),

-- One row per interval, row count unchanged. The ladder is written once here
-- so the grouping below does not have to repeat it.
labelled_buckets AS (
    SELECT
        gap_definition,
        gap_seconds,
        CASE
            WHEN gap_seconds = 0 THEN '1. 0秒'
            WHEN gap_seconds > 0 AND gap_seconds <= 3600 THEN '2. 1小时内'
            WHEN gap_seconds > 3600 AND gap_seconds <= 86400 THEN '3. 1小时–1天'
            WHEN gap_seconds > 86400 AND gap_seconds <= 259200 THEN '4. 1–3天'
            WHEN gap_seconds > 259200 AND gap_seconds <= 604800 THEN '5. 3–7天'
            WHEN gap_seconds > 604800 AND gap_seconds <= 1209600 THEN '6. 7–14天'
            WHEN gap_seconds > 1209600 AND gap_seconds <= 2592000 THEN '7. 14–30天'
            WHEN gap_seconds > 2592000 AND gap_seconds <= 5270400 THEN '8. 30–61天'
            ELSE 'UNBUCKETED'
        END AS bucket
    FROM labelled_gaps
),

bucket_counts AS (
    SELECT
        gap_definition,
        bucket,
        COUNT(*) AS gap_count
    FROM labelled_buckets
    GROUP BY gap_definition, bucket
)

SELECT
    gap_definition,
    bucket,
    gap_count,
    ROUND(
        gap_count * 100.0 / SUM(gap_count) OVER (PARTITION BY gap_definition), 4
    ) AS gap_share_pct,
    ROUND(
        SUM(gap_count * 100.0) OVER (
            PARTITION BY gap_definition ORDER BY bucket
        ) / SUM(gap_count) OVER (PARTITION BY gap_definition), 4
    ) AS cumulative_share_pct
FROM bucket_counts
ORDER BY gap_definition, bucket;
