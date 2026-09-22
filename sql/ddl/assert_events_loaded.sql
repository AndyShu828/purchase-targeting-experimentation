-- Post-load assertions for pte.events. ZERO ROWS IS THE PASS.
--
-- This file carries the guarantee that create_events.sql cannot: NOT NULL is
-- unloadable here, because DuckDB writes every Parquet field as OPTIONAL and
-- BigQuery refuses OPTIONAL data into a REQUIRED column. A constraint the
-- database enforces on write is stronger than a query someone has to run --
-- this is the weaker instrument, and it has to actually be run, every load.
--
-- Each row this returns is a failed assertion, named and with its count.
-- Anything other than an empty result means do not query the table yet.
--
-- EXPECTED VALUES for the October load, all from the local Parquet:
--   rows               42,448,764
--   days                       31 (2019-10-01 .. 2019-10-31)
--   SUM(price)     370,635,242.26 on 2019-10-01 alone
-- The row count and the day count are the two that catch a partial load, which
-- is the realistic failure: a per-day loop that stops early leaves a table that
-- looks healthy and is short by a few days.
--
-- Cost: this reads event_date and the seven asserted columns across every
-- partition on purpose. It is the one query in the project that is supposed to
-- scan everything, because "nothing is missing" cannot be established from a
-- sample.
--
-- Run: bq query --use_legacy_sql=false < sql/ddl/assert_events_loaded.sql
WITH null_counts AS (
    SELECT
        COUNT(*) AS rows_total,
        COUNT(DISTINCT event_date) AS days_total,
        COUNTIF(event_time IS NULL) AS null_event_time,
        COUNTIF(event_date IS NULL) AS null_event_date,
        COUNTIF(event_type IS NULL) AS null_event_type,
        COUNTIF(product_id IS NULL) AS null_product_id,
        COUNTIF(category_id IS NULL) AS null_category_id,
        COUNTIF(price IS NULL) AS null_price,
        COUNTIF(user_id IS NULL) AS null_user_id,
        COUNTIF(day_index IS NULL) AS null_day_index,
        -- The partition key is derived, so it can disagree with the column it
        -- was derived from. A row in the wrong partition still answers a
        -- day_index filter wrongly while looking perfectly normal.
        COUNTIF(day_index <> DATE_DIFF(event_date, DATE '2019-10-01', DAY)) AS day_index_mismatch
    FROM pte.events
),

-- UNPIVOT turns the one-row report into one row per assertion, so a new check
-- is a new column above rather than another branch below.
failures AS (
    SELECT
        assertion,
        observed
    FROM null_counts
    UNPIVOT (
        observed FOR assertion IN (
            null_event_time,
            null_event_date,
            null_event_type,
            null_product_id,
            null_category_id,
            null_price,
            null_user_id,
            null_day_index,
            day_index_mismatch
        )
    )
    WHERE observed > 0
)

SELECT
    assertion,
    observed,
    0 AS expected
FROM failures

UNION ALL

SELECT
    'row_count' AS assertion,
    rows_total AS observed,
    42448764 AS expected
FROM null_counts
WHERE rows_total <> 42448764

UNION ALL

SELECT
    'day_count' AS assertion,
    days_total AS observed,
    31 AS expected
FROM null_counts
WHERE days_total <> 31

ORDER BY assertion;
