-- How much of the cart abandonment drift is the 2019-11-15 tracking outage.
--
-- GRAIN IS ONE ROW PER TIME WINDOW, the same four windows quoted in
-- cart_abandonment_by_category.sql, each reported twice: over every cart pair,
-- and over the pairs whose session does not touch the outage.
--
-- THE OUTAGE IS A PURCHASE-SIDE BLACKOUT, 2019-11-14 18:09:37 to 2019-11-16
-- 08:32:34 UTC, 38.4 hours with no purchase event recorded while views and
-- carts ran at a record rate (data_quality_daily_event_anomalies.sql). Inside
-- it, a session that carted and then bought looks exactly like a session that
-- carted and left. Abandonment is therefore overstated for any session that
-- overlaps the window, and only for those.
--
-- THE UNIT REMOVED IS A SESSION, NOT AN EVENT OR A DAY. Abandonment is defined
-- within a session, so the corrupted unit is the whole session, and the filter
-- can only be written once sessions exist -- after the 30-minute cut, on
-- session_start and session_end, never on event_time. Filtering events in the
-- innermost layer would be two bugs at once: it changes the gaps the cut is
-- computed from, so sessions are split differently, and it leaves the part of
-- an overlapping session that falls outside the window in place, still missing
-- the purchases that would have landed inside it. Filtering by calendar day
-- would miss the evening of 11-14, where tracking was already down.
--
-- WINDOW BOUNDS ARE OBSERVED, NOT TRUE. 18:09:37 is the last purchase before
-- the blackout and 08:32:34 the first one after it, so the real failure starts
-- somewhat later and ends somewhat earlier. Using the observed bounds drops
-- slightly more sessions than necessary, which is the conservative direction.
--
-- READ BOTH COLUMNS. The raw column is what the metric said before this was
-- known; the gap between them is the size of the defect. Reporting only the
-- corrected figure would hide the finding.
--
-- REGRESSION VALUES, --full, 4 rows (raw column reproduces the four figures in
-- cart_abandonment_by_category.sql exactly, which is what says this pipeline
-- is the same one):
--   window        cart_pairs   raw %    dropped    kept       corrected %
--   1. 10/01-07      112,445  49.0222         0    112,445        49.0222
--   2. 10/08-31      473,733  49.6214         0    473,733        49.6214
--   3. 11/01-14      470,622  58.8976    47,530    423,092        54.2986
--   4. 11/15-30    1,392,150  65.0009   385,154  1,006,996        51.7380
--
-- WHAT IT SAYS: most of the drift was the outage. 65.00% against October's
-- 49.62% is a 15.4pp gap; after removing the sessions that could not record a
-- purchase, 51.74% leaves 2.1pp. The remaining 11/01-14 figure, 54.30%, is
-- still the highest of the four, so November does differ from October -- but
-- by a few points, not by fifteen, and the earlier reading of a large genuine
-- behavioural shift does not survive.
--
-- The four CTEs before session_bounds are the sessionisation pipeline from
-- cart_abandonment_by_category.sql, minus the columns this question does not
-- need (category_code, the NULL-code flag).
WITH previous_event_times AS (
    SELECT
        event_time,
        user_session,
        category_id,
        LAG(event_time, 1)
            OVER (PARTITION BY user_session ORDER BY event_time)
            AS previous_event_time,
        CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END AS is_cart,
        CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END AS is_purchase
    FROM events
),

is_new_sessions AS (
    SELECT
        event_time,
        user_session,
        category_id,
        is_cart,
        is_purchase,
        CASE
            WHEN TIMESTAMP_DIFF(event_time, previous_event_time, SECOND) > 1800 THEN 1
            ELSE 0
        END AS is_new_session
    FROM previous_event_times
),

session_events AS (
    SELECT
        event_time,
        user_session,
        category_id,
        is_cart,
        is_purchase,
        SUM(is_new_session) OVER (PARTITION BY user_session ORDER BY event_time) AS new_index
    FROM is_new_sessions
),

-- The session grain has to exist before the outage filter can be written, and
-- this is where it first does: one row per cut session, with the two timestamps
-- the overlap test needs.
session_bounds AS (
    SELECT
        user_session,
        new_index,
        MIN(event_time) AS session_start,
        MAX(event_time) AS session_end
    FROM session_events
    GROUP BY user_session, new_index
),

-- Overlap, not containment: a session counts as touched if it starts before
-- the blackout ends AND ends after the blackout starts. The flag is carried on
-- the pair rather than filtered here, so the same pass can report both columns.
cart_pairs AS (
    SELECT
        session_events.user_session,
        session_events.new_index,
        session_events.category_id,
        MIN(session_bounds.session_start) AS session_start,
        MAX(session_events.is_cart) AS has_cart,
        MAX(session_events.is_purchase) AS has_purchase,
        MAX(
            CASE
                WHEN
                    session_bounds.session_start < TIMESTAMP '2019-11-16 08:32:34'
                    AND session_bounds.session_end > TIMESTAMP '2019-11-14 18:09:37'
                    THEN 1
                ELSE 0
            END
        ) AS is_outage_session
    FROM session_events
    INNER JOIN session_bounds
        ON
            session_events.user_session = session_bounds.user_session
            AND session_events.new_index = session_bounds.new_index
    GROUP BY
        session_events.user_session, session_events.new_index, session_events.category_id
)

SELECT
    -- Windows are labelled by the session's start, so a session that crosses a
    -- boundary is counted once, in the window it began in.
    CASE
        WHEN session_start < TIMESTAMP '2019-10-08' THEN '1. 10/01-07'
        WHEN session_start < TIMESTAMP '2019-11-01' THEN '2. 10/08-31'
        WHEN session_start < TIMESTAMP '2019-11-15' THEN '3. 11/01-14'
        ELSE '4. 11/15-30'
    END AS window_label,
    COUNT(*) AS cart_pairs,
    ROUND(COUNT(CASE WHEN has_purchase = 0 THEN 1 END) * 100.0 / COUNT(*), 4) AS rate_raw_pct,
    COUNT(CASE WHEN is_outage_session = 1 THEN 1 END) AS pairs_dropped,
    COUNT(CASE WHEN is_outage_session = 0 THEN 1 END) AS pairs_kept,
    ROUND(
        COUNT(CASE WHEN is_outage_session = 0 AND has_purchase = 0 THEN 1 END) * 100.0
        / NULLIF(COUNT(CASE WHEN is_outage_session = 0 THEN 1 END), 0),
        4
    ) AS rate_corrected_pct
FROM cart_pairs
WHERE has_cart = 1
GROUP BY window_label
ORDER BY window_label;
