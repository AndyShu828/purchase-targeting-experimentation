-- Can the 32.21% of events with no category_code be recovered?
--
-- Answer: no, not one row. category_code is missing all-or-nothing per
-- category, never intermittently, so there is nowhere to copy a label from.
--
-- THE SHAPE OF THE MISSINGNESS IDENTIFIES WHERE THE FIELD COMES FROM. Compare
-- data_quality_zero_price_by_product.sql: price fails intermittently, 93.91% of
-- affected products carry a real price somewhere else, and 97.60% of the bad
-- rows are recoverable. price is captured per event, so individual captures
-- fail. category_code is the opposite -- a category either has a label on every
-- one of its rows or on none of them, which is what a join against a category
-- dictionary produces. A lookup does not succeed intermittently. Neither file
-- inspects the pipeline; both infer its shape from the pattern of the nulls.
--
-- The two denominators do NOT agree here and both are therefore reported.
-- 59.9% of categories are unlabelled but they hold only 32.21% of events: an
-- unlabelled category averages 85,541 events against 269,087 for a labelled
-- one, 3.1x smaller. Quoting either figure alone misleads. This is the case
-- zero-price was not: there the product and row views agreed (93.91% vs
-- 97.60%) and the conclusion held whichever was chosen.
--
-- Prerequisites established separately and both required for the above:
--   * category_id and product_id have zero nulls over all 109,950,743 rows, so
--     every event does have a category -- what is missing is its NAME, not its
--     classification. Grouping by category_id loses nothing.
--   * No category_id carries two different codes (max distinct = 1), so a
--     backfill would have been unambiguous had anything been backfillable.
--
-- categories_sometimes_missing is expected to be 0 and is the column to watch:
-- if a later load makes it non-zero, the join-shaped story above is wrong and
-- backfill becomes possible. missing_rows_total reconciles across queries
-- against the 35,413,780 counted by grouping on nothing in
-- data_quality_nulls.sql.
WITH category_code_coverage AS (
    SELECT
        category_id,
        COUNT(*) AS event_count,
        COUNT(DISTINCT category_code) AS distinct_codes,
        COUNT(CASE WHEN category_code IS NULL THEN 1 END) AS missing_rows
    FROM events
    GROUP BY category_id
)

SELECT
    COUNT(*) AS total_categories,
    COUNT(
        CASE WHEN distinct_codes = 1 AND missing_rows = 0 THEN 1 END
    ) AS categories_never_missing,
    COUNT(
        CASE WHEN distinct_codes = 1 AND missing_rows > 0 THEN 1 END
    ) AS categories_sometimes_missing,
    COUNT(CASE WHEN distinct_codes = 0 THEN 1 END) AS categories_always_missing,
    SUM(CASE WHEN distinct_codes = 1 THEN event_count ELSE 0 END) AS events_labelled,
    SUM(CASE WHEN distinct_codes = 0 THEN event_count ELSE 0 END) AS events_unlabelled,
    SUM(missing_rows) AS missing_rows_total,
    SUM(
        CASE WHEN distinct_codes = 1 THEN missing_rows ELSE 0 END
    ) AS missing_rows_recoverable
FROM category_code_coverage;
