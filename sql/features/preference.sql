-- Preference feature group: the DuckDB side of src/pte/features/preference.py.
--
-- Andy's five queries (2026-09-25); Claude merged them into one, parameterised it and
-- formatted it for sqlfluff. Variables: cutoff, window_start. One row per population user.
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
-- main_category_conversion_rate: the purchase rate of the user's main category over
-- (session, category) pairs in the feature window -- P1's category_conversion_rate.sql
-- unit, as a 0..1 share. A pair counts once however many purchases it holds, and no cart
-- is required (the loose funnel: conclusion 17 found 30% of purchase sessions never
-- carted the item). Sessions are cut as in intent.sql: window first, then 30 minutes
-- within (user_id, user_session). Sessions overlapping the purchase outage are dropped
-- whole; the outage bounds are themselves recorded purchases, so the comparison is
-- strict. Small categories are left as they are (decided 2026-09-25): 33 of 652 have
-- under 100 pairs at 11-18, and they are the main category of 73 users.
--
-- Andy's price_percentile joined the population to the ranked events and averaged after
-- the join; here the average is taken first so all the parts join the same way.
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
),

previous_event_times AS (
    SELECT
        user_id,
        user_session,
        category_id,
        event_time,
        LAG(event_time, 1)
            OVER (PARTITION BY user_session, user_id ORDER BY event_time)
            AS previous_event_time,
        CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END AS is_purchase
    FROM events
    WHERE event_date >= GETVARIABLE('window_start') AND event_date < GETVARIABLE('cutoff')
),

is_new_sessions AS (
    SELECT
        user_id,
        user_session,
        category_id,
        event_time,
        is_purchase,
        CASE
            WHEN DATE_DIFF('second', previous_event_time, event_time) > 1800 THEN 1
            ELSE 0
        END AS is_new_session
    FROM previous_event_times
),

indexes AS (
    SELECT
        user_id,
        user_session,
        category_id,
        event_time,
        is_purchase,
        SUM(is_new_session)
            OVER (PARTITION BY user_session, user_id ORDER BY event_time)
            AS new_index
    FROM is_new_sessions
),

session_start_ends AS (
    SELECT
        user_id,
        user_session,
        category_id,
        is_purchase,
        new_index,
        CASE
            WHEN
                MIN(event_time) OVER (PARTITION BY user_session, new_index, user_id)
                < TIMESTAMP '2019-11-16 08:32:34'
                AND MAX(event_time) OVER (PARTITION BY user_session, new_index, user_id)
                > TIMESTAMP '2019-11-14 18:09:37'
                THEN 1
            ELSE 0
        END AS is_outage_session
    FROM indexes
),

has_purchases AS (
    SELECT
        user_id,
        user_session,
        new_index,
        category_id,
        MAX(is_purchase) AS has_purchase
    FROM session_start_ends
    WHERE is_outage_session = 0
    GROUP BY user_session, user_id, new_index, category_id
),

category_conversion_rates AS (
    SELECT
        category_id,
        COUNT(CASE WHEN has_purchase = 1 THEN 1 END) * 1.0 / COUNT(*) AS category_conversion_rate
    FROM has_purchases
    GROUP BY category_id
)

SELECT
    p.user_id,
    pp.price_percentile,
    m.main_category_id,
    r.category_conversion_rate AS main_category_conversion_rate,
    COALESCE(c.n_categories, 0) AS n_categories,
    COALESCE(c.n_brands, 0) AS n_brands
FROM population AS p
LEFT JOIN n_categories_brands AS c ON p.user_id = c.user_id
LEFT JOIN price_percentiles AS pp ON p.user_id = pp.user_id
LEFT JOIN rn_1 AS m ON p.user_id = m.user_id
LEFT JOIN category_conversion_rates AS r ON m.main_category_id = r.category_id
