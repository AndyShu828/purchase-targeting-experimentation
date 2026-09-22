-- Does `user_session` delimit a time-bounded session?
--
-- The criterion is the largest gap BETWEEN CONSECUTIVE EVENTS INSIDE a session,
-- not the session's total span. A visitor who is active for 80 minutes but
-- clicks every 20 minutes is a single legal session under a 30-minute timeout;
-- judging by span would flag it as an anomaly. Only an internal gap longer than
-- the timeout is evidence that no timeout is being applied.
--
-- MAX turns that existential question into a scalar comparison:
-- `MAX(gap) > 1800` is exactly equivalent to "some gap exceeds 1800", because
-- if the largest one does not, none can.
--
-- Three query blocks are required, one per grain -- event, session, dataset.
-- The boundaries are forced, not stylistic:
--   * a window function cannot sit inside an aggregate, so LAG must be resolved
--     before MAX can consume it;
--   * aggregates cannot be nested, so counting the sessions whose MAX exceeds a
--     threshold cannot happen in the block that computes that MAX.
--
-- DIALECT: written in BigQuery form, TIMESTAMP_DIFF(end, start, SECOND) with
-- end first. DuckDB has no TIMESTAMP_DIFF; scripts/run_sql.py defines it as a
-- macro that only honours SECOND (docs/sql_dialects.md).
-- Swapping the two timestamps flips the sign silently, which MAX then reads as
-- the shortest gap rather than the longest, producing a clean-looking "every
-- session respects the timeout".
WITH event_gaps AS (
    -- Event grain: 109,950,743 rows in, 109,950,743 rows out. A window function
    -- adds a column, it never collapses rows. PARTITION BY keeps LAG from
    -- reaching across a session boundary, so the first event of each session
    -- gets NULL -- meaning "start of session", not "gap of zero".
    SELECT
        user_session,
        event_time,
        LAG(event_time, 1) OVER (
            PARTITION BY user_session ORDER BY event_time
        ) AS previous_event_time
    FROM events
),

session_gaps AS (
    -- Session grain: one row per session. The individual timestamps are gone
    -- from here on, which is why the gap has to be computed above.
    -- A single-event session has no adjacent pair, so MAX skips its only NULL
    -- and returns NULL -- an abstention, not a zero.
    SELECT
        user_session,
        COUNT(*) AS event_count,
        MAX(TIMESTAMP_DIFF(event_time, previous_event_time, SECOND)) AS max_event_gap_seconds
    FROM event_gaps
    GROUP BY user_session
)

-- Dataset grain: aggregates with no GROUP BY treat the whole input as one group,
-- which is what collapses 23 million rows to one.
SELECT
    -- Reconciliation columns, not findings. total_events restores the original
    -- row count, proving the two collapses lost and duplicated nothing; the same
    -- technique as SUM(occurrences) in data_quality_duplicates.sql.
    SUM(event_count) AS total_events,
    COUNT(*) AS total_sessions,
    -- These two must sum to total_sessions. The identity holds only if event_time
    -- is never NULL, so agreement between the two routes is also a null check on
    -- a column the missing-value query never covered.
    COUNT(CASE WHEN event_count = 1 THEN 1 END) AS single_event_sessions,
    COUNT(max_event_gap_seconds) AS sessions_with_gap,
    -- The decisive number. Read it against the width of the observation window,
    -- not against intuition: a maximum that tracks the window is censored by the
    -- data, which means unbounded.
    MAX(max_event_gap_seconds) AS max_gap_seconds,
    -- No ELSE: non-matching rows become NULL and COUNT skips them. Single-event
    -- sessions drop out for free, since NULL > 1800 is NULL rather than TRUE.
    -- The rate for this column takes sessions_with_gap as its denominator.
    COUNT(CASE WHEN max_event_gap_seconds > 1800 THEN 1 END) AS sessions_over_30min
FROM session_gaps;
