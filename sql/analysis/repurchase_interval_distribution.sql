-- Distribution of the interval between consecutive purchases, under two
-- definitions of what counts as one purchase.
--
-- GRAIN IS ONE ROW PER (definition, bucket). Two pipelines feed one bucketing
-- layer through a UNION ALL, so the shares are computed per definition and the
-- two are read side by side. The comparison is the deliverable: the definition
-- changes the answer to the question the project actually asks.
--
-- INTERVALS ARE KEYED ON user_id, NEVER user_session. Conclusion 5 established
-- that user_session is a cookie, and the full set carries 4.33 of them per
-- person, so partitioning on it would cut one person's repurchases apart and
-- overstate every interval.
--
-- THE TWO DEFINITIONS:
--   purchase_event    -- every purchase event is its own purchase
--   shopping_occasion -- consecutive purchases by one user with no 30-minute
--                        pause collapse into one, anchored at the first
-- The occasion cut is chained, not windowed: three purchases 25 minutes apart
-- are one occasion spanning 75 minutes, with no upper bound on its length.
-- That is the same property measured in step 2, where cookie sessions were
-- found spanning the entire 61-day window.
--
-- WHY BOTH. 36.84% of event-level intervals fall inside 30 minutes, so the
-- choice moves a third of the data. It is not spread evenly: per-bucket counts
-- above roughly 4.5 hours agree between the two to within 0.3%, and the whole
-- difference sits below that. The shares still diverge there because the
-- denominators do -- 962,318 against 607,754 -- which is why "80% of
-- repurchases happen within N days" resolves to 6 days under one definition
-- and 12 under the other. Quoting either number without naming its definition
-- is the error this file exists to prevent.
--
-- The anchor is MIN, so an occasion interval runs first-purchase to
-- first-purchase: a shopping cycle. Anchoring on MAX instead would measure the
-- dormancy between trips, a different and equally defensible quantity that is
-- not computed here.
--
-- WHERE event_type = 'purchase' SITS BEFORE THE WINDOW, unlike
-- sessionization_purchase_segment.sql where it must sit after. Both placements
-- parse and both return a plausible table; only this one answers the question,
-- because LAG must reach the previous PURCHASE, not the previous event of any
-- kind. Placing it after the window computes time-from-last-action-to-purchase,
-- which peaks at 13-19 seconds and is a view-to-buy latency. The placement is
-- also worth about two orders of magnitude: filtering first cuts 109,950,743
-- rows to 1,659,788 before the sort, and this file runs in about a second
-- against 77-86s for category_conversion_rate.sql over the same data.
--
-- BUCKETS ARE POWERS OF TWO, computed rather than listed. FLOOR(LOG(2, x))
-- makes the boundaries arithmetic, so they cannot overlap, cannot leave a gap
-- and cannot go stale when the window is extended -- a class of bug a
-- hand-written CASE ladder has to be re-checked for every time. Base 10 was
-- rejected on resolution: it yields 7 buckets for the event definition and 4
-- for the occasion definition, and 4 points do not show a shape. Base 2 yields
-- 23 and 13, and at that resolution the event-level distribution is visibly
-- BIMODAL -- a peak at 1-4 minutes and a second at 18 hours to 3 days, with a
-- trough between. Base 10 hides this.
--
-- THREE ARTEFACTS THAT ARE NOT BEHAVIOUR, all visible in the output:
--   1. bucket_exponent = -1 holds intervals of exactly 0 seconds, 85 of them
--      at event level, and they are not behaviour at all. All 85 pairs share a
--      user, a second, a product and a price, which makes them the same 85
--      redundant rows that the nine-column duplicate check in
--      data_quality_duplicates.sql found among purchases -- 0.0051% of
--      purchase events, every group exactly a pair, never a triple. They are
--      machine repeats, so the bucket is an artefact of a known data defect
--      rather than a customer buying twice in one second, and it is kept
--      visible for that reason. Dropping the duplicates upstream, once that
--      cleaning rule is applied, should empty this bucket exactly; that it
--      does is a free check on the cleaning.
--      Mechanically, LOG of zero raises rather than returning NULL, so the
--      argument is passed through NULLIF; CASE would read better but BigQuery
--      does not promise to leave the unmatched branch unevaluated.
--   2. bucket_exponent = 10 spans 1024-2048 seconds and the 30-minute cut
--      falls inside it, so the occasion definition keeps only the 1801-2048
--      part: 4,174 against 34,700. The bucket is split by the threshold, not
--      by behaviour. A 2048-second threshold would have aligned with the
--      boundary, but 30 minutes is the one with meaning, so this is annotated
--      rather than tuned away.
--   3. bucket_exponent = 22 spans 48.5-97 days while the data spans 61, so it
--      is roughly a quarter full. The whole distribution is right-censored:
--      the longest interval observed is 5,235,330 seconds = 60.6 days, hard
--      against the window. Conclusion 14 ran this query on the 7-day sample
--      and on the full set and found shares shifting by up to 1.29x --
--      censoring removes intervals but not events, so ANY interval metric read
--      off the sample is wrong in a predictable direction. Run --full.
--
-- THE OCCASION DEFINITION EMITS NO BUCKET BELOW 10, because its shortest
-- possible interval is 1,801 seconds and FLOOR(LOG(2, 1801)) is 10. Those rows
-- are absent rather than zero: the grouping only produces buckets that have
-- data. That is a true consequence of the cut and not a filter, but a reader
-- comparing the two blocks sees eleven missing rows and has to know why.
-- A spine of bucket numbers LEFT JOINed onto the counts would show them as
-- zeroes. It is left out by decision (2026-09-23), for three reasons:
--   1. There is no common spelling for it: DuckDB has no GENERATE_ARRAY, and
--      FROM UNNEST([...]) AS x binds the alias differently on the two engines,
--      so it would be 24 literal rows or a recursive CTE.
--   2. A literal spine hard-codes the top bucket, 22, and so goes stale when
--      the window is extended -- the one failure computed buckets are immune to.
--   3. These eleven zeroes are impossible by definition, not measured. Printed
--      as 0 they invite "nobody repurchases within an hour"; absent, with this
--      note, they say the definition cannot produce such an interval.
-- The event-level block has no empty bucket between -1 and 22 on the full
-- data, so the occasion floor is the only place rows are missing.
--
-- cumulative_share_pct ALIGNS WITH bucket_high_seconds_excl, not with the low
-- bound on the same row: it is the share at or below the top of this bucket.
-- On a log scale, reading it against the low bound is wrong by a full factor
-- of two. Both bounds are emitted so the row cannot be misread.
--
-- DIALECT: written in BigQuery form, TIMESTAMP_DIFF(end, start, SECOND) with
-- end first. DuckDB has no TIMESTAMP_DIFF; scripts/run_sql.py defines it as a
-- macro that only honours SECOND (docs/sql_dialects.md).
-- The base-2 bucket is LN(x) / LN(2), not LOG(2, x), because the two-argument
-- LOG takes its arguments in the opposite order on the two engines: DuckDB
-- reads LOG(base, x), BigQuery reads LOG(x, base). The original spelling was
-- expected to fail quietly on BigQuery -- every row collapsing into one bucket
-- as log base gap_seconds of 2 -- and it did fail, but loudly, and only by
-- luck: one interval is exactly 1 second, log base 1 is undefined, and the job
-- raised "Floating point error in function: LOG(2, 1)". Without that single row
-- the answer would have been wrong and the query green. LN over LN has no
-- argument order to get backwards, and it was checked to agree with the old
-- spelling on exact powers of two, where a float division could have rounded
-- into the neighbouring bucket.
--
-- ⚠️ THESE SHARES ARE AN UPPER BOUND ON THE INTERVAL (2026-09-23). Purchase
-- tracking was down for 38.4 hours from 2019-11-14 18:09:37 UTC, so an
-- interval spanning it is reported as one long gap instead of two short ones.
-- The 80th percentile is 5.96 days here against 3.63 with those intervals
-- dropped at event level, and 10.88 against 6.95 at occasion level; the
-- dropping overshoots, so the truth is inside each pair.
-- repurchase_interval_outage_sensitivity.sql holds both columns. The factor of
-- two between the definitions survives at either end.
--
-- REGRESSION VALUES, full set, --full:
--   purchase_event     962,318 intervals, 85 at zero, buckets -1 through 22
--   shopping_occasion  607,754 intervals, 1,305,224 occasions, min gap 1,801
--   The occasion interval count equals the event-level count above 1800
--   seconds exactly, because an occasion boundary occurs where and only where
--   an event-level interval exceeds the threshold.
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

gap_buckets AS (
    SELECT
        gap_definition,
        COALESCE(FLOOR(LN(NULLIF(gap_seconds, 0)) / LN(2)), -1) AS bucket_exponent,
        COUNT(*) AS gap_count
    FROM labelled_gaps
    GROUP BY gap_definition, COALESCE(FLOOR(LN(NULLIF(gap_seconds, 0)) / LN(2)), -1)
)

SELECT
    gap_definition,
    bucket_exponent,
    gap_count,
    CASE WHEN bucket_exponent < 0 THEN 0 ELSE POWER(2, bucket_exponent) END
        AS bucket_low_seconds,
    CASE WHEN bucket_exponent < 0 THEN 1 ELSE POWER(2, bucket_exponent + 1) END
        AS bucket_high_seconds_excl,
    ROUND(
        gap_count * 100.0 / SUM(gap_count) OVER (PARTITION BY gap_definition), 4
    ) AS gap_share_pct,
    ROUND(
        SUM(gap_count * 100.0) OVER (
            PARTITION BY gap_definition ORDER BY bucket_exponent
        ) / SUM(gap_count) OVER (PARTITION BY gap_definition), 4
    ) AS cumulative_share_pct
FROM gap_buckets
ORDER BY gap_definition, bucket_exponent;
