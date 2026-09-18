-- Where zero prices land in the funnel.
--
-- The split matters because the two ends mean different things. A zero on a
-- view or a cart event is a page that failed to show a price; a zero on a
-- purchase would be an order settled for nothing, which is business data
-- corruption rather than a capture fault. Establishing that the second kind
-- does not exist is what keeps the revenue side of the project out of scope
-- of this defect.
--
-- The rate carries four decimals to match the convention in docs/data_quality.md
-- and because the defect rates in this dataset routinely sit in the 1e-4 range:
-- two decimals would round cart's rate to 0.00 and make it indistinguishable
-- from purchase's true zero. The raw counts are kept alongside, so the rounding
-- can never hide anything -- they are the source of truth, the rate is a
-- convenience.
--
-- The `* 100.0` sits BEFORE the division on purpose. Written the other way
-- round, PostgreSQL would do integer division first and return 0 without
-- complaint; DuckDB and BigQuery would not, so the bug would only appear after
-- the query moved.
SELECT
    event_type,
    COUNT(*) AS event_count,
    -- No ELSE: non-matching rows become NULL and COUNT skips them.
    COUNT(CASE WHEN price = 0 THEN 1 END) AS price_zero,
    ROUND(COUNT(CASE WHEN price = 0 THEN 1 END) * 100.0 / COUNT(*), 4) AS zero_rate_pct
FROM events
GROUP BY event_type
ORDER BY event_count DESC;
