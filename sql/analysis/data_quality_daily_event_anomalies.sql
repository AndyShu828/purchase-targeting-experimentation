-- Days where one event type is missing or far off its own daily level.
--
-- WHY THIS CHECK EXISTS: 2019-11-15 has 6.2M events, 483,305 of them carts,
-- and not one purchase. The gap runs from 2019-11-14 18:09:37 to 2019-11-16
-- 08:32:34 UTC, 38.4 hours; views and carts continue at a record rate
-- throughout. The dataset author, asked about it on the Kaggle discussion
-- page, points to a tracking failure on the retailer's front end.
--
-- The number was in our own output all along: daily_activity.sql prints
-- purchase_events per day, and that row reads 0. Nothing flagged it, because
-- every self-check in this project reconciles TOTALS -- the three event types
-- sum to 109,950,743, the session grain and the date grain agree -- and a
-- missing day breaks no total. A zero is invisible to a sum. This query is the
-- complement: it asserts something about each cell, not about the total.
--
-- THE GRID IS THE POINT. GROUP BY event_date, event_type cannot emit a row for
-- a combination that has no rows, so the very case worth catching is the one
-- that silently disappears. Every date is crossed with every event type first,
-- and the counts are LEFT JOINed onto that grid, which turns "no rows" into an
-- explicit 0. A check that can only describe data that exists is not a check.
--
-- ITS BLIND SPOT, for the same reason: the grid is built from the event types
-- that appear SOMEWHERE in the data, so a type absent from the whole window
-- stays invisible. remove_from_cart is exactly that case -- the source
-- describes it, these two months contain none -- and no query over this table
-- can surface it. That one needs the source documentation, not SQL.
--
-- THRESHOLDS ARE A SCREEN, NOT A VERDICT. Below a quarter of the type's daily
-- mean, or above four times it, gets listed for a human to read. The mean is
-- deliberately crude: it is portable to any engine, unlike the median, whose
-- spelling differs between DuckDB and BigQuery. It is also inflated by the
-- 11/15-17 spike, which makes the LOW side harder to trip -- acceptable, since
-- MISSING is exact and does not depend on a threshold at all.
--
-- ZERO ROWS IS THE HEALTHY RESULT. This query answers a yes/no question, so it
-- returns only what it objects to.
--
-- PARTIAL DAYS AT THE EDGES WOULD TRIP IT. The window starts and ends on whole
-- UTC days, so nothing is clipped here. Extending to another month means the
-- first and last day of that load can read LOW for honest reasons.
--
-- REGRESSION VALUES, --full, 5 rows:
--   event_date   event_type  event_count  share_of_mean  verdict
--   2019-11-15   cart            483,305         7.4534  HIGH
--   2019-11-15   purchase              0         0.0     MISSING
--   2019-11-16   cart            406,778         6.2732  HIGH
--   2019-11-17   cart            426,941         6.5842  HIGH
--   2019-11-17   purchase        185,195         6.8062  HIGH
-- view never trips: its 11-15 peak is 3.35x its mean, under the 4x line.
WITH event_dates AS (
    SELECT DISTINCT event_date FROM events
),

event_types AS (
    SELECT DISTINCT event_type FROM events
),

day_type_grid AS (
    SELECT
        event_dates.event_date,
        event_types.event_type
    FROM event_dates
    CROSS JOIN event_types
),

observed_counts AS (
    SELECT
        event_date,
        event_type,
        COUNT(*) AS event_count
    FROM events
    GROUP BY event_date, event_type
),

-- The LEFT JOIN is what manufactures the missing cell: grid on the left, so a
-- combination with no events survives with a NULL count, and COALESCE makes
-- that a 0. An INNER JOIN here would delete the finding.
daily_counts AS (
    SELECT
        day_type_grid.event_date,
        day_type_grid.event_type,
        COALESCE(observed_counts.event_count, 0) AS event_count
    FROM day_type_grid
    LEFT JOIN observed_counts
        ON
            day_type_grid.event_date = observed_counts.event_date
            AND day_type_grid.event_type = observed_counts.event_type
),

-- Each day is judged against its own event type, never against the others:
-- purchases are two orders of magnitude rarer than views, so one shared level
-- would flag every purchase day. The window expression is written twice rather
-- than referenced as an alias, which BigQuery rejects in the same SELECT.
scored_days AS (
    SELECT
        event_date,
        event_type,
        event_count,
        AVG(event_count) OVER (PARTITION BY event_type) AS type_daily_mean,
        ROUND(
            event_count / NULLIF(AVG(event_count) OVER (PARTITION BY event_type), 0),
            4
        ) AS share_of_mean
    FROM daily_counts
)

SELECT
    event_date,
    event_type,
    event_count,
    CAST(ROUND(type_daily_mean) AS INT64) AS type_daily_mean,
    share_of_mean,
    CASE
        WHEN event_count = 0 THEN 'MISSING'
        WHEN share_of_mean < 0.25 THEN 'LOW'
        WHEN share_of_mean > 4 THEN 'HIGH'
        ELSE 'OK'
    END AS verdict
FROM scored_days
WHERE
    event_count = 0
    OR share_of_mean < 0.25
    OR share_of_mean > 4
ORDER BY event_date, event_type;
