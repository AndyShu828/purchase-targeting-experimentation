-- Daily coverage and price validity, in one pass over the stream.
--
-- Two of the seven checks share a scan because both need every row and neither
-- needs anything the other does not: is the calendar complete, and is `price`
-- usable. Splitting them would cost a second full scan for nothing.
--
-- COMPLETENESS, NOT UNIFORMITY. The date test asks whether every day arrived,
-- not whether the days resemble each other. Row counts per day are wildly
-- uneven by design -- Black Friday falls inside the window -- so a mean +/- k*sigma
-- rule would flag real business volume as corruption, and dirty days would
-- inflate the very mean and sigma meant to catch them. The equality below has
-- no such failure mode: dates are contiguous if and only if the span implied by
-- the endpoints equals the number of days actually present.
--
-- The +1 is a fencepost: 2019-10-01 to 2019-10-07 is a difference of 6 across
-- 7 days. COUNT(*) is the number of distinct dates because daily_counts already
-- holds exactly one row per date.
--
-- The four price buckets are mutually exclusive and exhaustive, so summing them
-- must reproduce the row count -- the same self-checking table convention used
-- in data_quality_nulls.sql. KNOWN BLIND SPOT: `price` is DOUBLE, and DuckDB
-- orders NaN above every number rather than following IEEE, so a NaN would be
-- counted as positive and the bucket check would still pass. Detecting NaN
-- needs an explicit isnan(); this query does not claim to exclude it.
--
-- DIALECT: DATE - DATE returns a day count in DuckDB. The subtraction is chosen
-- over DATE_DIFF deliberately: BigQuery spells that DATE_DIFF(end, start, part)
-- while DuckDB takes date_diff(part, start, end), so porting the arguments
-- verbatim reverses the sign in silence. Written as `MAX - MIN` the order is
-- visible in the expression and cannot be quietly transposed. Whether BigQuery
-- accepts DATE - DATE is not yet verified; if it does not, it fails loudly.
WITH daily_counts AS (
    SELECT
        event_date,
        COUNT(*) AS event_count,
        -- No ELSE anywhere below: non-matching rows become NULL and COUNT skips
        -- them. NULL prices fall through the three comparisons silently, which
        -- is why price_null is counted separately rather than assumed to be 0 --
        -- the missing-value check never covered this column.
        COUNT(CASE WHEN price > 0 THEN 1 END) AS price_positive,
        COUNT(CASE WHEN price = 0 THEN 1 END) AS price_zero,
        COUNT(CASE WHEN price < 0 THEN 1 END) AS price_negative,
        COUNT(CASE WHEN price IS NULL THEN 1 END) AS price_null,
        -- How negative and how large, not just how many. A sentinel -1 and a
        -- corrupt -99999 need different cleaning rules, and a count alone cannot
        -- tell them apart.
        MIN(price) AS min_price,
        MAX(price) AS max_price
    FROM events
    GROUP BY event_date
)

SELECT
    -- Reconciliation, not a finding: grouping by date must preserve the row
    -- count, and it must agree with the same total reached by grouping by
    -- session in data_quality_session_gaps.sql.
    SUM(event_count) AS total_events,
    MIN(event_date) AS first_date,
    MAX(event_date) AS last_date,
    COUNT(*) AS actual_dates,
    MAX(event_date) - MIN(event_date) + 1 AS expected_dates,
    -- Both numbers stay above so that a failure says how many days are missing,
    -- not merely that some are.
    (MAX(event_date) - MIN(event_date) + 1 = COUNT(*)) AS date_equal_check,
    SUM(price_positive) AS price_positive,
    SUM(price_zero) AS price_zero,
    SUM(price_negative) AS price_negative,
    SUM(price_null) AS price_null,
    MIN(min_price) AS min_price,
    MAX(max_price) AS max_price,
    (
        SUM(price_positive) + SUM(price_zero) + SUM(price_negative)
        + SUM(price_null) = SUM(event_count)
    ) AS price_bucket_check
FROM daily_counts;
