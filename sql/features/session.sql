-- Session feature group: the DuckDB side of src/pte/features/session.py.
--
-- Andy's query (2026-09-26); Claude parameterised it, renamed per the naming rules and
-- formatted it for sqlfluff. Run through scripts/bench_features.py, which sets the
-- variables: cutoff (the first label day), window_start (cutoff - window_days).
-- One row per population user.
--
-- Sessions are cut as in intent.sql: a gap over 1800 s within (user_id, user_session),
-- on events already filtered to the window, so a session crossing the cutoff ends there.
-- Decided 2026-09-26: outage sessions stay in (no purchase is involved here), and
-- single-event sessions count, with 0 seconds: the span from first to last event, not
-- the time on the site, which is unrecorded after the last event of every session.
-- The session index sums the new-session flags under the default RANGE frame, so events
-- tied on event_time share a session; a ROWS frame splits them, differently each run.
WITH population AS (
    SELECT DISTINCT user_id
    FROM events
    WHERE
        event_date >= GETVARIABLE('cutoff') - INTERVAL 14 DAY AND event_date < GETVARIABLE('cutoff')
),

previous_event_times AS (
    SELECT
        user_id,
        user_session,
        event_time,
        LAG(event_time, 1)
            OVER (PARTITION BY user_session, user_id ORDER BY event_time)
            AS previous_event_time
    FROM events
    WHERE event_date >= GETVARIABLE('window_start') AND event_date < GETVARIABLE('cutoff')
),

event_gaps AS (
    SELECT
        user_id,
        user_session,
        event_time,
        DATE_DIFF('second', previous_event_time, event_time) AS gap_seconds
    FROM previous_event_times
),

is_new_sessions AS (
    SELECT
        user_id,
        user_session,
        event_time,
        CASE WHEN gap_seconds > 1800 THEN 1 ELSE 0 END AS is_new_session
    FROM event_gaps
),

session_events AS (
    SELECT
        user_id,
        user_session,
        event_time,
        SUM(is_new_session)
            OVER (PARTITION BY user_session, user_id ORDER BY event_time)
            AS session_index
    FROM is_new_sessions
),

session_bounds AS (
    SELECT
        user_id,
        user_session,
        session_index,
        MIN(event_time)
            OVER (PARTITION BY user_session, session_index, user_id)
            AS session_start,
        MAX(event_time)
            OVER (PARTITION BY user_session, session_index, user_id)
            AS session_end
    FROM session_events
),

sessions AS (
    SELECT
        user_id,
        user_session,
        session_index,
        COUNT(*) AS event_count,
        DATE_DIFF('second', MAX(session_start), MAX(session_end)) AS session_seconds
    FROM session_bounds
    GROUP BY user_id, user_session, session_index
),

user_sessions AS (
    SELECT
        user_id,
        COUNT(*) AS n_sessions,
        AVG(session_seconds) AS avg_session_seconds,
        SUM(event_count) * 1.0 / COUNT(*) AS events_per_session
    FROM sessions
    GROUP BY user_id
)

SELECT
    p.user_id,
    u.avg_session_seconds,
    u.events_per_session,
    COALESCE(u.n_sessions, 0) AS n_sessions
FROM population AS p LEFT JOIN user_sessions AS u ON p.user_id = u.user_id
