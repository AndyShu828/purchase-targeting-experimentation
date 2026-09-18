-- Are zero prices recoverable? Classify every product by whether it is ever
-- seen with a real price.
--
-- This is the test that separates the two explanations. If a product carries a
-- normal price on other rows, a zero on this row cannot be the item's price --
-- the value is known and simply was not captured here, so it is repairable. If
-- a product is never seen with a price at all, there is nothing to repair and
-- nothing to impute from; it is plausibly an item that was never priced.
--
-- TWO DENOMINATORS, BOTH REPORTED. "What share of affected PRODUCTS are
-- repairable" and "what share of zero-price ROWS are repairable" are different
-- questions, and they can disagree sharply: a handful of high-traffic products
-- can own most of the rows while being a rounding error among products. Which
-- one to quote depends on the question -- products for the mechanism, rows for
-- sizing the cleanup. Reporting only one invites the reader to assume the other.
--
-- The three classes are mutually exclusive and exhaustive. price_positive = 0
-- implies price_zero > 0, because every product has at least one row and every
-- row on the full stream is either zero or positive -- the missing-value and
-- negative-price counts are both zero there. total_products therefore equals
-- the three class counts added together, and the table checks itself, the same
-- convention as data_quality_nulls.sql.
--
-- ELSE 0 is load-bearing in the conditional SUMs and deliberately absent from
-- the conditional COUNTs. A SUM over no matching rows returns NULL rather than
-- 0, and NULL compared against a threshold is NULL rather than FALSE -- the
-- shape that lets a broken feed pass a monitoring check in silence.
WITH product_prices AS (
    SELECT
        product_id,
        COUNT(*) AS event_count,
        COUNT(CASE WHEN price > 0 THEN 1 END) AS price_positive,
        COUNT(CASE WHEN price = 0 THEN 1 END) AS price_zero
    FROM events
    GROUP BY product_id
)

SELECT
    COUNT(*) AS total_products,
    COUNT(CASE WHEN price_zero = 0 THEN 1 END) AS products_no_zero,
    COUNT(
        CASE WHEN price_positive > 0 AND price_zero > 0 THEN 1 END
    ) AS products_sometimes_zero,
    COUNT(CASE WHEN price_positive = 0 THEN 1 END) AS products_always_zero,
    -- Reconciles against data_quality_dates_and_price.sql, which reaches the
    -- same total by grouping on date instead of product.
    SUM(price_zero) AS zero_rows_total,
    SUM(
        CASE WHEN price_positive > 0 AND price_zero > 0 THEN price_zero ELSE 0 END
    ) AS zero_rows_sometimes,
    SUM(
        CASE WHEN price_positive = 0 THEN price_zero ELSE 0 END
    ) AS zero_rows_always
FROM product_prices;
