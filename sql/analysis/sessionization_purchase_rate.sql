-- Session purchase conversion rate under two session definitions.
--
-- The question is not "how many sessions convert" but "how much does the choice
-- of session definition move that number". Check five established that
-- `user_session` never applies a timeout -- the largest gap inside one is 57
-- days and grows in proportion to the observation window -- so treating it as a
-- session is a measurement error, and this query prices that error.
--
-- THE NUMERATOR IS BOOLEAN, NOT A COUNT. A session holding three purchases
-- contributes 1, not 3. This is not a style preference. Purchases are conserved
-- by the cut: splitting a session moves purchase events between sessions, it
-- never creates or destroys one. So a count numerator would be byte-identical
-- under both definitions and the rate would fall by exactly the denominator's
-- growth -- an arithmetic identity that measures nothing. Only the boolean
-- numerator can move, and it moves precisely when one provided session's
-- purchases land in two different segments, which is the effect under study.
-- purchase_events_per_session is reported alongside to expose that degenerate
-- case: its change IS the denominator's change, so the gap between the two
-- rates is the only real signal in this result.
--
-- Both definitions take the whole population as the denominator, single-event
-- sessions included. They are not dropped because they are not unconvertible --
-- 130 single-event sessions in the development sample are themselves a purchase
-- -- and because the cut produces proportionally more of them, so excluding
-- them would remove a different share from each side and manufacture a
-- difference out of the filter rather than the definition.
--
-- DIALECT: written in BigQuery form, TIMESTAMP_DIFF(end, start, SECOND) with
-- end first. DuckDB has no TIMESTAMP_DIFF; scripts/run_sql.py defines it as a
-- macro that only honours SECOND (docs/sql_dialects.md).
WITH events_with_previous_time AS (
    -- Event grain. A window function adds a column, it never collapses rows.
    -- PARTITION BY stops LAG at the partition boundary, so the first event of
    -- each partition gets NULL -- "start of partition", not "gap of zero".
    SELECT
        user_session,
        event_time,
        event_type,
        LAG(event_time, 1) OVER (
            PARTITION BY user_session ORDER BY event_time
        ) AS previous_event_time
    FROM events
),

events_with_gaps AS (
    SELECT
        *,
        TIMESTAMP_DIFF(event_time, previous_event_time, SECOND) AS gap_seconds
    FROM events_with_previous_time
),

marked_events AS (
    -- NULL > 1800 evaluates to NULL rather than TRUE, so the first event of a
    -- partition falls through to ELSE 0 and never opens a segment by itself.
    SELECT
        *,
        CASE WHEN gap_seconds > 1800 THEN 1 ELSE 0 END AS is_session_start
    FROM events_with_gaps
),

indexed_events AS (
    -- The running sum turns the 0/1 marks into a 0-based segment number.
    -- The window frame stays at its default (RANGE) on purpose. Under ROWS,
    -- events sharing an event_time would be split across two segments in an
    -- order the engine does not guarantee, making the result irreproducible
    -- run to run; the development sample holds 4,732 tied groups, so this is
    -- a live risk rather than a hypothetical one.
    SELECT
        *,
        SUM(is_session_start) OVER (
            PARTITION BY user_session ORDER BY event_time
        ) AS session_index
    FROM marked_events
),

native_sessions AS (
    -- Session grain, definition one: one row per user_session as provided.
    -- The grouping key is deliberately absent from the SELECT list -- the only
    -- thing this block owes downstream is one fact about each session.
    SELECT COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) AS purchase_event_count
    FROM indexed_events
    GROUP BY user_session
),

cut_sessions AS (
    -- Session grain, definition two: one row per 30-minute-timeout session.
    -- The ONLY difference from native_sessions is the extra grouping key.
    SELECT COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) AS purchase_event_count
    FROM indexed_events
    GROUP BY user_session, session_index
)

-- Dataset grain, one row per definition. Definitions are rows rather than
-- columns so that adding a third threshold later costs a row instead of a
-- schema change. COUNT(*) counts sessions here because each input row IS a
-- session -- the same COUNT(*) over indexed_events would count events.
SELECT
    'user_session (as provided)' AS session_definition,
    COUNT(*) AS session_count,
    COUNT(CASE WHEN purchase_event_count > 0 THEN 1 END) AS sessions_with_purchase,
    ROUND(
        COUNT(CASE WHEN purchase_event_count > 0 THEN 1 END) * 100.0 / COUNT(*), 4
    ) AS purchase_rate_pct,
    -- Reconciliation and contrast in one column: this total is conserved, so
    -- the two rows must agree on it or the pipeline has lost or duplicated
    -- events somewhere.
    SUM(purchase_event_count) AS purchase_events,
    ROUND(SUM(purchase_event_count) * 1.0 / COUNT(*), 6) AS purchase_events_per_session
FROM native_sessions

UNION ALL

SELECT
    '30-minute inactivity timeout' AS session_definition,
    COUNT(*) AS session_count,
    COUNT(CASE WHEN purchase_event_count > 0 THEN 1 END) AS sessions_with_purchase,
    ROUND(
        COUNT(CASE WHEN purchase_event_count > 0 THEN 1 END) * 100.0 / COUNT(*), 4
    ) AS purchase_rate_pct,
    SUM(purchase_event_count) AS purchase_events,
    ROUND(SUM(purchase_event_count) * 1.0 / COUNT(*), 6) AS purchase_events_per_session
FROM cut_sessions

-- Cutting only ever splits sessions, so the provided definition always has the
-- smaller count. Sorting on it puts the baseline first without inventing a key.
ORDER BY session_count;
