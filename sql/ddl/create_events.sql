-- The events table on BigQuery: October 2019, partitioned and clustered.
--
-- THIS FILE IS THE TABLE. The sandbox drops every table 60 days after it is
-- created, so the table is disposable and this DDL is not: rebuilding means
-- running this file and then scripts/load_bq_month.sh. Nothing about the table
-- may exist only in a console click.
--
-- PARTITION BY RANGE_BUCKET(day_index), ONE PARTITION PER DAY BUT KEYED ON AN
-- INTEGER, because a sandbox project cannot hold a DATE-partitioned table of
-- 2019 data. Partition expiry there is capped at 60 days and is measured from
-- the partition's own date, so every 2019 partition expires the moment it
-- lands. Measured, not assumed: a first attempt with PARTITION BY event_date
-- took a load job that reported SUCCESS and 1,244,245 output rows into a table
-- that then held 0. Removing the expiry is refused ("must be less than 60 days
-- while in sandbox mode") both on the table and on the dataset default, and a
-- partitioned table without an expiry cannot be written to at all. An integer
-- range partition carries no such clock.
--
-- day_index is days since 2019-10-01, 0..60 across both months, written by
-- scripts/export_bq_parquet.py. The buckets are GENERATE_ARRAY(0, 62, 1): 62
-- daily partitions with room for November, and anything outside the range
-- lands in __UNPARTITIONED__ rather than failing, so a bad day_index shows up
-- as a partition rather than as a silent miss.
--
-- WHAT THIS COSTS. Pruning now needs a filter on day_index; a filter on
-- event_date alone scans everything. None of the 11 analysis queries filters
-- on a date at all, so none of them changes -- the cost falls on step 5's
-- pruning benchmark and on P5's daily DAG, which must translate their date
-- into day_index. event_date stays as an ordinary column, so GROUP BY
-- event_date still reads naturally.
--
-- The honest framing for this decision: the sandbox forced it. With billing
-- enabled the plan's PARTITION BY event_date is the right design, and this
-- file would change to it.
--
-- CLUSTER BY event_type, user_id -- deliberately NOT the plan's order of
-- user_id, event_type. Clustering sorts the data by the listed columns in
-- order, and only the leading columns prune: a filter on the second column
-- cannot skip a block unless the first one also narrows it. Of the 11 analysis
-- queries, 4 filter on event_type ('purchase', 1.75% of rows) and none filters
-- on user_id. user_id keeps the second slot for P2, where per-user feature
-- windows may want it; in the second slot it costs nothing.
--
-- EVERY COLUMN IS NULLABLE, AND THAT IS A LOADER CONSTRAINT, NOT A CLAIM ABOUT
-- THE DATA. Seven columns are null-free across all 109,950,743 rows
-- (docs/data_quality.md check 1, plus a direct count for the numeric ones), so
-- NOT NULL was the first version of this file. It cannot be loaded: DuckDB
-- writes every Parquet field as OPTIONAL even for a NOT NULL column (checked
-- on 1.5.5), and BigQuery refuses OPTIONAL data into a REQUIRED column --
-- "Field event_type has changed mode from REQUIRED to NULLABLE". Making the
-- file REQUIRED needs a different writer, which is a dependency bought for one
-- constraint. The guarantee is kept as an assertion instead:
-- sql/ddl/assert_events_loaded.sql must return zero rows after every load.
--
-- Had NOT NULL been loadable, user_session would still be the trap: null on 12
-- rows out of 109,950,743, which rounds to 0% in every report, and those 12
-- would fail the whole batch. "Almost never missing" and "guaranteed present"
-- are different claims. brand (13.94% null) and category_code (32.21% null)
-- are obvious by comparison.
--
-- price IS NUMERIC, NOT FLOAT64. Money is summed, and float summation is
-- inexact in a way that differs between engines: one million additions of 0.01
-- give 10000.000000171856 on BigQuery and 9999.999999999878 on DuckDB, both
-- wrong and in opposite directions. NUMERIC is exact decimal. The type is left
-- unparameterised: NUMERIC(10, 2) would also be true of this data (max price
-- 2,574.07, no row past two decimals) but would reject a whole load batch over
-- one out-of-range value, and the export already casts to DECIMAL(10, 2).
--
-- NO partition_expiration_days, ON PURPOSE. Partition expiry is measured from
-- the partition's own date, not from when it was loaded, and these partitions
-- are dated 2019 -- any finite value would expire all of them on arrival. The
-- 60-day clock that does apply is the sandbox's table expiration, which runs
-- from table creation and cannot be turned off without billing.
--
-- require_partition_filter = FALSE, ON PURPOSE. Requiring a partition filter is
-- the usual guard against an accidental full scan, but step 5 of P1 measures
-- exactly that: no partition filter, then a partition filter, then a cluster
-- column filter as well. The guard would make the first of the three
-- impossible to run. Revisit after that benchmark.
--
-- COLUMN ORDER DOES NOT BIND THE LOAD. A Parquet load matches columns by name,
-- so the order here is for reading, not for matching; the export happens to
-- emit price last.
CREATE TABLE pte.events
(
    event_time TIMESTAMP
    OPTIONS (description = "Event timestamp in UTC, second resolution."),
    event_date DATE
    OPTIONS (description = "UTC calendar day of event_time; day_index is the partition key."),
    event_type STRING
    OPTIONS (description = "One of view, cart, purchase. No remove_from_cart in these months."),
    product_id INT64
    OPTIONS (description = "Product identifier. Not unique per physical item."),
    category_id INT64
    OPTIONS (description = "Category identifier, 19 digits; fits INT64 with room to spare."),
    category_code STRING
    OPTIONS (description = "Category taxonomy; null on 32.21% of rows, all-or-nothing per id."),
    brand STRING
    OPTIONS (description = "Lower-cased brand; null on 13.94% of rows."),
    price NUMERIC
    OPTIONS (description = "Product price at event time. Currency is not stated by the source."),
    user_id INT64
    OPTIONS (description = "Permanent user identifier."),
    user_session STRING
    OPTIONS (description = "Vendor session id; null on 12 rows, not a session (conclusion 5)."),
    day_index INT64
    OPTIONS (description = "Days since 2019-10-01; the partition key. 0..30 for October.")
)
PARTITION BY RANGE_BUCKET(day_index, GENERATE_ARRAY(0, 62, 1))
CLUSTER BY event_type, user_id
OPTIONS (
    description
    = "REES46 e-commerce events, 2019-10. Rebuild: this file, then scripts/load_bq_month.sh.",
    require_partition_filter = FALSE
);
