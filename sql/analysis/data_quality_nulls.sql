-- Missing-value shape for the four columns that can plausibly be absent.
--
-- Each column is split into three MUTUALLY EXCLUSIVE buckets, so that
-- _null + _empty + _whitespace is the column's total unusable count and the
-- table carries its own check:
--   _null        IS NULL
--   _empty       = ''                      -- non-NULL, zero length
--   _whitespace  TRIM(col) = '' AND <> ''  -- non-NULL, spaces only
--
-- TRIM(NULL) is NULL, so `TRIM(col) = ''` evaluates to NULL on NULL rows and is
-- skipped by COUNT. The blank counters therefore cover empty + whitespace only,
-- which is why _whitespace is derived as (blank - empty).
--
-- The ratios are computed in the outer SELECT because BigQuery and PostgreSQL
-- do not allow an alias defined in a SELECT list to be referenced in that same
-- SELECT list; DuckDB does, which would hide the error until the query moved.
WITH empty_blank AS (
    SELECT
        COUNT(*) - COUNT(brand) AS brand_null,
        COUNT(*) - COUNT(category_code) AS category_code_null,
        COUNT(*) - COUNT(event_type) AS event_type_null,
        COUNT(*) - COUNT(user_session) AS user_session_null,
        COUNT(CASE WHEN brand = '' THEN 1 END) AS brand_empty,
        COUNT(CASE WHEN category_code = '' THEN 1 END) AS category_code_empty,
        COUNT(CASE WHEN event_type = '' THEN 1 END) AS event_type_empty,
        COUNT(CASE WHEN user_session = '' THEN 1 END) AS user_session_empty,
        COUNT(CASE WHEN TRIM(brand) = '' THEN 1 END) AS brand_blank,
        COUNT(CASE WHEN TRIM(category_code) = '' THEN 1 END) AS category_code_blank,
        COUNT(CASE WHEN TRIM(event_type) = '' THEN 1 END) AS event_type_blank,
        COUNT(CASE WHEN TRIM(user_session) = '' THEN 1 END) AS user_session_blank
    FROM events
)

-- ST06 wants plain columns before calculated ones, which would split the
-- three buckets of each column apart. Grouping by column is what makes the
-- mutual-exclusivity check readable, so the rule is waived here only.
SELECT  -- noqa: ST06
    brand_null,
    brand_empty,
    brand_blank - brand_empty AS brand_whitespace,
    category_code_null,
    category_code_empty,
    category_code_blank - category_code_empty AS category_code_whitespace,
    event_type_null,
    event_type_empty,
    event_type_blank - event_type_empty AS event_type_whitespace,
    user_session_null,
    user_session_empty,
    user_session_blank - user_session_empty AS user_session_whitespace
FROM empty_blank;
