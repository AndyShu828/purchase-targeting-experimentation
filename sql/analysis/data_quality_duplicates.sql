-- Fully duplicated events, split by event_type.
--
-- A group is one distinct event: all nine columns identical. event_date is left
-- out of the key because it is derived from event_time and would only widen the
-- hash without changing which rows group together.
--
-- GROUP BY treats NULL as equal to NULL, so rows with a missing brand or
-- category_code pair up correctly. A self join on the same columns would NOT:
-- `a.brand = b.brand` is NULL when both are NULL, silently dropping 13.9% of
-- brand rows and 32.2% of category_code rows from the comparison.
--
-- Splitting by event_type is what makes the result interpretable: view and
-- purchase act as a built-in control group for what ordinary user-driven
-- repetition looks like, and cart is read against them rather than against an
-- arbitrary threshold.
WITH duplicate_groups AS (
    SELECT
        event_type,
        product_id,
        category_id,
        category_code,
        brand,
        price,
        user_id,
        user_session,
        event_time,
        COUNT(*) AS occurrences
    FROM events
    GROUP BY
        event_type,
        product_id,
        category_id,
        category_code,
        brand,
        price,
        user_id,
        user_session,
        event_time
)

SELECT
    event_type,
    -- Restores the original row count, so this column reconciles against
    -- event_type_distribution.sql. COUNT(*) would instead give the number of
    -- distinct events, which is 130,739 lower on the full stream.
    SUM(occurrences) AS total_events,
    -- No ELSE: non-matching rows become NULL and COUNT skips them.
    COUNT(CASE WHEN occurrences > 1 THEN 1 END) AS duplicated_groups,
    -- ELSE 0 is load-bearing here: without it, an event_type with no duplicate
    -- groups at all would SUM to NULL rather than 0, and NULL compared against
    -- a threshold is NULL, not TRUE -- the failure mode that makes a broken
    -- feed pass a monitoring check silently.
    SUM(CASE WHEN occurrences > 1 THEN occurrences - 1 ELSE 0 END) AS redundant_rows,
    -- Cheap anomaly probe. On the full stream this is 4 for view, 2 for
    -- purchase and 78 for cart; the first two do not move as the data grows,
    -- the third does.
    MAX(occurrences) AS max_occurrences
FROM duplicate_groups
GROUP BY event_type
ORDER BY total_events DESC;
