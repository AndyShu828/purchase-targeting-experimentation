-- Does one `user_session` ever belong to more than one `user_id`?
--
-- GRAIN IS ONE ROW PER USERS-PER-SESSION VALUE: the distribution of distinct
-- user_id counts over sessions. Two blocks, because the count per session has
-- to exist before sessions can be grouped by it.
--
-- NULL user_session IS EXCLUDED ON PURPOSE. GROUP BY folds the 12 rows with no
-- session into one group, and those 12 rows belong to 12 different users, so
-- keeping them adds a row reading "one session shared by 12 users" -- a
-- finding that does not exist, with no error to flag it. Without the filter
-- the counts sum to 23,016,651; with it, to 23,016,650, which is
-- COUNT(DISTINCT user_session), the figure that excludes NULL by definition.
--
-- REGRESSION VALUES, --full, 3 rows:
--   users_per_session  session_count
--   1                     23,015,711
--   2                            938
--   3                              1
-- 939 multi-user sessions, 0.00408%. The mechanism behind them was looked at on
-- 9 examples only and is left as an impression, not a finding
-- (docs/data_quality.md, check five).
WITH session_users AS (
    SELECT
        user_session,
        COUNT(DISTINCT user_id) AS users_per_session
    FROM events
    WHERE user_session IS NOT NULL
    GROUP BY user_session
)

SELECT
    users_per_session,
    COUNT(*) AS session_count
FROM session_users
GROUP BY users_per_session
ORDER BY users_per_session;
