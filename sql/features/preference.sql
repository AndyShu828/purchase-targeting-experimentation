-- Preference feature group: the DuckDB side of src/pte/features/preference.py.
--
-- Andy's four queries (2026-09-25); Claude merged them into one, parameterised it and
-- formatted it for sqlfluff. Variables: cutoff, window_start. One row per population user.
-- main_category_conversion_rate (step 2) is still to come.
--
-- n_brands counts distinct non-NULL brands; a user whose every event has no brand gets 0.
--
-- price_percentile: each event with price > 0 is ranked by price within its category_id
-- (PERCENT_RANK, 0..1), then averaged per user. ZERO PRICES ARE DROPPED IN THE WHERE, not
-- turned into NULL: a NULL row still takes part in the ranking and pushes every real
-- price down. The ranking only sees the feature window, so no price from after the
-- cutoff can move it. A user with no positive-price event gets NULL (1,025 at 11-18).
--
-- main_category_id: the category with the most events; ties go to the category the user
-- touched most recently, then to the smaller id. The last key is there so the choice is
-- deterministic: 40 users at 11-18 tie on both count and last event time, and without it
-- ROW_NUMBER picks between them arbitrarily, so DuckDB and Spark could disagree.
--
-- Andy's price_percentile joined the population to the ranked events and averaged after
-- the join; here the average is taken first so all three parts join the same way.
WITH population AS (
    SELECT DISTINCT user_id
    FROM events
    WHERE
        event_date >= GETVARIABLE('cutoff') - INTERVAL 14 DAY AND event_date < GETVARIABLE('cutoff')
),

n_categories_brands AS (
    SELECT
        user_id,
        COUNT(DISTINCT category_id) AS n_categories,
        COUNT(DISTINCT brand) AS n_brands
    FROM events
    WHERE event_date >= GETVARIABLE('window_start') AND event_date < GETVARIABLE('cutoff')
    GROUP BY user_id
),

positive_prices AS (
    SELECT
        category_id,
        user_id,
        price
    FROM events
    WHERE
        event_date >= GETVARIABLE('window_start') AND event_date < GETVARIABLE('cutoff')
        AND price > 0
),

percent_ranks AS (
    SELECT
        user_id,
        PERCENT_RANK() OVER (PARTITION BY category_id ORDER BY price) AS percent_rank
    FROM positive_prices
),

price_percentiles AS (
    SELECT
        user_id,
        AVG(percent_rank) AS price_percentile
    FROM percent_ranks
    GROUP BY user_id
),

pairs_user_category_id AS (
    SELECT
        user_id,
        category_id,
        COUNT(*) AS cnt_pairs_user_category_id,
        MAX(event_time) AS last_event_time
    FROM events
    WHERE event_date >= GETVARIABLE('window_start') AND event_date < GETVARIABLE('cutoff')
    GROUP BY user_id, category_id
),

rns AS (
    SELECT
        user_id,
        category_id,
        ROW_NUMBER() OVER (
            PARTITION BY user_id
            ORDER BY cnt_pairs_user_category_id DESC, last_event_time DESC, category_id ASC
        ) AS rn
    FROM pairs_user_category_id
),

rn_1 AS (
    SELECT
        user_id,
        category_id AS main_category_id
    FROM rns
    WHERE rn = 1
)

SELECT
    p.user_id,
    pp.price_percentile,
    m.main_category_id,
    COALESCE(c.n_categories, 0) AS n_categories,
    COALESCE(c.n_brands, 0) AS n_brands
FROM population AS p
LEFT JOIN n_categories_brands AS c ON p.user_id = c.user_id
LEFT JOIN price_percentiles AS pp ON p.user_id = pp.user_id
LEFT JOIN rn_1 AS m ON p.user_id = m.user_id
