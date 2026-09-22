-- Weekly purchase retention by first-purchase cohort: of the users whose first
-- purchase fell in week W, what share bought again k weeks later.
--
-- GRAIN IS ONE ROW PER (cohort, k), a long table of 45 rows. The usual
-- cohort triangle, one row per cohort and one column per k, is the same data
-- pivoted, and the pivot is left to the presentation layer (the Looker page in
-- P6). In SQL every column of that pivot is its own CASE, nine near-identical
-- branches to keep in step; as rows, the retention share is one expression.
--
-- THE ANCHOR IS THE FIRST PURCHASE, NOT THE FIRST VISIT. The question is the
-- one a payment provider asks -- after a customer's first checkout, do they
-- come back -- and views and carts are the merchant's data, not the payer's.
-- The filter to purchases therefore sits in the innermost layer: no window
-- here needs to see other event types.
--
-- WEEKS ARE CALENDAR WEEKS STARTING MONDAY (DuckDB DATE_TRUNC('week')), so the
-- window 2019-10-01 .. 11-30 spans nine weeks, 09-30 .. 11-25. The first and
-- last are six days long. k is the calendar distance in weeks between the week
-- of a purchase and the week of the user's first purchase, a subtraction, not
-- a count of active weeks: a user who buys in weeks 0 and 2 has no k = 1 row,
-- and that absence is what makes k = 1 a retention figure.
--
-- USER-WEEKS BEFORE COHORT-WEEKS. purchase_ks collapses purchases to one row
-- per (user, week), so the count at (cohort, k) is people, not purchases. A
-- user who buys five times in a week is counted once. It also absorbs the 85
-- known duplicate purchase rows without special handling.
--
-- THE DENOMINATOR IS THE k = 0 ROW OF THE SAME COHORT, reached by a window.
-- Every cohort member has a k = 0 row by construction, so its count is the
-- cohort size. Two wrong denominators return plausible tables:
--   SUM(COUNT(*)) OVER (PARTITION BY cohort) -- user-weeks of the cohort, not
--       users. It deflates early cohorts more than late ones, because they
--       have more weeks to sum, so cohorts stop being comparable.
--   COUNT(CASE WHEN k = 0 ...) without OVER -- sees only its own (cohort, k)
--       cell, so it is 0 for every k > 0: inf in DuckDB, an error on BigQuery.
-- Check: every k = 0 row reads 100, and the nine cohort sizes sum to 697,470,
-- the number of distinct purchasers.
--
-- The window wraps an aggregate, SUM(COUNT(...)) OVER (...). This is legal
-- because GROUP BY is evaluated before windows in the same SELECT; the
-- reverse, an aggregate over a window, is not. The expression is written twice
-- rather than referenced as cohort_size, since BigQuery rejects a reference to
-- an alias defined in the same SELECT (lateral alias, conclusion 52).
--
-- THE LOWER-RIGHT TRIANGLE IS EMPTY, NOT ZERO. Cohort 11-25 can only be
-- observed at k = 0; cohort 09-30 up to k = 8. Missing cells are right
-- censoring by the end of the data. Extending the window (2019-12) fills them.
--
-- THE FIRST COHORT IS LIKELY INFLATED BY LEFT CENSORING. The data starts on
-- 10-01, so "first purchase" means first purchase seen. Customers who last
-- bought in September are counted as new in the first weeks, and existing
-- customers repurchase more. k = 1 falls across the first four cohorts,
-- 20.33 / 16.09 / 13.45 / 10.98, consistent with that; it is not proven, as
-- there is no September data to separate old customers out. The six-day first
-- week shrinks the cohort, not the rate: numerator and denominator are drawn
-- from the same users.
--
-- ONE CALENDAR WEEK SHOWS UP AS A DIAGONAL. The spike of 11-15 .. 11-17
-- (conclusion 6; not Black Friday, which is 11-29) lies in week 11-11, which
-- is k = 6 for cohort 09-30, k = 5 for 10-07, and so on. Retention jumps in
-- that cell of every cohort (09-30: 11.33 at k = 5, 13.85 at k = 6) and drops
-- in the following week 11-18 for every cohort. Read along a row, a cohort's
-- decay has a false bump at that diagonal: it is a calendar effect, not a
-- property of the cohort.
--
-- THE 11-11 COHORT RETAINS WORST, beyond the calendar effect. Its k = 1 is
-- 7.17, the lowest in the table. Part of that is week 11-18, which is low for
-- everyone. Compared within that same week, cohort 11-04 reads 7.57 at k = 2,
-- so cohort 11-11 at k = 1 sits below a neighbour one step further into its
-- decay, where k = 1 normally exceeds k = 2 by a wide margin (11-04 itself:
-- 14.67 against 7.57). This is consistent with the promotion bringing in
-- low-intent first-time buyers -- a reading of behaviour, not of motive.
-- For P4: the observation window 11-15 .. 11-28 starts on this spike, so the
-- share of promotion-week newcomers must be balanced between arms or used as
-- a stratum.
--
-- NEEDS --full. The dev sample, 10-01 .. 10-07, holds two cohorts (09-30 and
-- 10-07) and cannot observe any k above 1 for the first one.
--
-- DIALECT: DATE_TRUNC('week', ts) and DATE_DIFF('week', start, end) are the
-- DuckDB forms. On BigQuery the arguments are (ts, WEEK) and (end, start,
-- WEEK), and WEEK starts on Sunday -- ISOWEEK is the Monday week. Unverified;
-- see docs/sql_dialects.md for the pending decision on DATE_DIFF.
--
-- REGRESSION VALUES, --full, retention_pct (k = 0 is 100 throughout):
--   cohort   size     k=1    k=2    k=3    k=4    k=5    k=6    k=7    k=8
--   09-30    78,674  20.33  17.48  13.56  11.28  11.33  13.85   8.51   7.64
--   10-07    85,245  16.09  11.09   8.91   8.90  12.12   6.74   6.33
--   10-14    81,004  13.45   9.14   9.01  11.50   6.70   5.77
--   10-21    69,837  10.98   8.90  11.25   6.28   5.55
--   10-28    58,851  12.25  11.92   6.64   5.74
--   11-04    68,252  14.67   7.57   6.09
--   11-11   146,154   7.17   5.26
--   11-18    57,783  11.36
--   11-25    51,670
WITH purchase_weeks AS (
    SELECT
        user_id,
        event_time,
        DATE_TRUNC('week', event_time) AS purchase_week,
        DATE_TRUNC('week', MIN(event_time) OVER (PARTITION BY user_id))
            AS first_purchase_week
    FROM events
    WHERE event_type = 'purchase'
),

-- One row per (user, week). k is computed here, in its own layer, because it
-- reads first_purchase_week; placed in the SELECT that defines it, it would be
-- a lateral alias reference.
purchase_ks AS (
    SELECT
        user_id,
        purchase_week,
        first_purchase_week,
        DATE_DIFF('week', first_purchase_week, purchase_week) AS purchase_gap_k
    FROM purchase_weeks
    GROUP BY user_id, purchase_week, first_purchase_week
)

SELECT
    first_purchase_week,
    purchase_gap_k,
    COUNT(user_id) AS retained_users,
    SUM(COUNT(CASE WHEN purchase_gap_k = 0 THEN 1 END)) OVER (
        PARTITION BY first_purchase_week
    ) AS cohort_size,
    ROUND(
        COUNT(user_id) * 100.0
        / SUM(COUNT(CASE WHEN purchase_gap_k = 0 THEN 1 END)) OVER (
            PARTITION BY first_purchase_week
        ),
        4
    ) AS retention_pct
FROM purchase_ks
GROUP BY first_purchase_week, purchase_gap_k
ORDER BY first_purchase_week, purchase_gap_k;
