-- Distribution of event_type over the whole event stream.
--
-- Two things this settles:
--   1. The vocabulary is exactly view / cart / purchase. There is no
--      remove_from_cart, so a cart event is never retracted and "added to cart
--      but did not purchase" needs no undo handling.
--   2. The cart share is the baseline used to judge the 12 NULL user_session
--      rows, all of which are cart events.
--
-- The counts are NOT funnel conversion rates: one user can view or cart the
-- same product many times, so these are event-level totals only. Session-level
-- funnel work belongs in a separate query built on the sessionisation step.
SELECT
    event_type,
    COUNT(*) AS event_count
FROM events
GROUP BY event_type
ORDER BY event_count DESC;
