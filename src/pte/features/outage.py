"""The 2019-11-14 .. 16 purchase-tracking outage (docs/incident_2019_11_15.md).

No purchase event was recorded between these two instants; views and carts were.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

# Last purchase before the gap and first purchase after it, UTC: 38.3825 hours.
OUTAGE_START = datetime(2019, 11, 14, 18, 9, 37)
OUTAGE_END = datetime(2019, 11, 16, 8, 32, 34)


def effective_days(cutoff: date, window_days: int) -> float:
    """Days of [cutoff - window_days, cutoff) during which purchases were being recorded.

    Decided 2026-09-25: purchase counts are divided by this, not by window_days, so the
    test window (cutoff 11-18, 14 days -> 12.40) is on the same scale as training (14).
    """
    end = datetime.combine(cutoff, time())
    start = end - timedelta(days=window_days)
    overlap = min(end, OUTAGE_END) - max(start, OUTAGE_START)
    return window_days - max(overlap, timedelta(0)) / timedelta(days=1)
