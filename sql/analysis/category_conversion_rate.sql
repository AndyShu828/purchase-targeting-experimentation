-- Session-level purchase conversion by category.
--
-- GRAIN IS ONE ROW PER (CUT SESSION, category_id) PAIR, not per session. A
-- visit that browses three categories and buys in one contributes a
-- denominator to all three and a numerator to one, which is the only way to
-- see "looked but did not buy" per category. The consequence is that the
-- denominators sum to more than the session count -- 1.35 categories per
-- session on the development sample -- so session_category_pairs is not a
-- session total and must never be quoted as one.
--
-- NUMERATOR AND DENOMINATOR SHARE THE GRAIN. has_purchase is MAX() over the
-- pair, so a visit that buys three times in one category counts once. Using
-- purchase events instead would compute the share of events that are
-- purchases, which is 1.51% over the whole stream and is not a conversion
-- rate. The boolean can be checked rather than asserted: pairs_with_purchase
-- summed over all categories must not exceed the purchase total from
-- event_type_distribution.sql, because collapsing can only ever remove
-- purchases. It came to 133,587 against 152,115 on the sample, so 87.8% of
-- purchase events were already alone in their pair.
--
-- SESSIONS ARE CUT ON THE 30-MINUTE TIMEOUT, not taken from user_session,
-- because conclusion 8 measured the provided definition overstating
-- session-level conversion by 1.9% -- small but systematic and always in the
-- same direction. The first four CTEs are the pipeline from
-- sessionization_purchase_rate.sql; see it for why the default RANGE frame is
-- kept and what tied timestamps do under ROWS.
--
-- CATEGORIES ARE KEYED ON category_id, NEVER category_code. The code is
-- missing for 32.21% of events and, per
-- data_quality_category_code_coverage.sql, none of it is recoverable. It is
-- also a coarser grain: 277 labelled ids collapse into 129 codes. So the id is
-- the analysis key and the code is joined back as a display label that will be
-- NULL for 414 of the 691 categories. Labels come from their own scan rather
-- than riding through the window pipeline, which keeps a display attribute out
-- of the keys that fix the grain. Grouping the pair layer on category_code as
-- well would make the pair count depend on every category having exactly one
-- code -- true of this data today, at most one distinct code per id, but a
-- property of it rather than a guarantee, and a silent row split if it ever
-- stops holding.
--
-- is_reportable flags a numerator of at least 100. Every category is emitted
-- anyway: the tail is dominated by categories with a handful of pairs whose
-- rate is 0% or 100% by arithmetic rather than behaviour, and flagging them is
-- more honest than dropping them. Read rates only where the flag is true.
--
-- DIALECT: date_diff(part, start, end) is DuckDB and returns end - start.
-- BigQuery spells it TIMESTAMP_DIFF(end, start, part) -- the timestamps swap
-- places, so porting the arguments verbatim flips the sign silently.
WITH events_with_previous_time AS (
    SELECT
        user_session,
        event_time,
        category_id,
        CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END AS is_purchase,
        LAG(event_time, 1) OVER (
            PARTITION BY user_session ORDER BY event_time
        ) AS previous_event_time
    FROM events
),

events_with_gaps AS (
    SELECT
        *,
        DATE_DIFF('second', previous_event_time, event_time) AS gap_seconds
    FROM events_with_previous_time
),

marked_events AS (
    SELECT
        *,
        CASE WHEN gap_seconds > 1800 THEN 1 ELSE 0 END AS is_session_start
    FROM events_with_gaps
),

indexed_events AS (
    SELECT
        *,
        SUM(is_session_start) OVER (
            PARTITION BY user_session ORDER BY event_time
        ) AS session_index
    FROM marked_events
),

-- One row per (cut session, category). has_purchase is the boolean the rate
-- needs; the rate itself belongs one level up, where the grain is a category.
pair_purchases AS (
    SELECT
        user_session,
        session_index,
        category_id,
        MAX(is_purchase) AS has_purchase
    FROM indexed_events
    GROUP BY user_session, session_index, category_id
),

category_labels AS (
    SELECT
        category_id,
        MAX(category_code) AS category_code
    FROM events
    GROUP BY category_id
)

SELECT
    pairs.category_id,
    labels.category_code,
    COUNT(*) AS session_category_pairs,
    COUNT(CASE WHEN pairs.has_purchase = 1 THEN 1 END) AS pairs_with_purchase,
    ROUND(
        COUNT(CASE WHEN pairs.has_purchase = 1 THEN 1 END) * 100.0 / COUNT(*), 4
    ) AS purchase_rate_pct,
    COUNT(CASE WHEN pairs.has_purchase = 1 THEN 1 END) >= 100 AS is_reportable
FROM pair_purchases AS pairs
LEFT JOIN category_labels AS labels ON pairs.category_id = labels.category_id
GROUP BY pairs.category_id, labels.category_code
ORDER BY session_category_pairs DESC;
