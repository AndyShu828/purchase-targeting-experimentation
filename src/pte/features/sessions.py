"""Cut sessions on the 30-minute timeout, as in P1 (docs/sessionization.md).

user_session is a cookie, not a session (conclusion 5): a new session starts after a gap
of more than 30 minutes within one (user_id, user_session). P1 partitioned by
user_session alone; user_id is added here because features belong to one user, and 939
cookies (0.004%) carry more than one.
"""

from __future__ import annotations

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from pte.features.outage import OUTAGE_END, OUTAGE_START

SESSION_GAP_SECONDS = 1800


def _ntz(moment) -> Column:
    # event_time is timestamp_ntz; a string cast keeps the literal free of any time zone.
    return F.lit(moment.isoformat(sep=" ")).cast("timestamp_ntz")


def cut_sessions(events: DataFrame) -> DataFrame:
    """Every event, plus session_index, session_start, session_end and is_outage_session.

    Filter events to the feature window BEFORE calling this: a session that crosses the
    cutoff must end at the cutoff, or its later events leak into the features.
    """
    by_cookie = Window.partitionBy("user_id", "user_session").orderBy("event_time")
    gap = F.timestamp_diff("SECOND", F.lag("event_time").over(by_cookie), F.col("event_time"))
    # One session keeps every row and gains its bounds: a window, not a GROUP BY + join.
    by_session = Window.partitionBy("user_id", "user_session", "session_index")
    return (
        events.withColumn("is_new_session", F.when(gap > SESSION_GAP_SECONDS, 1).otherwise(0))
        .withColumn("session_index", F.sum("is_new_session").over(by_cookie))
        .drop("is_new_session")
        .withColumn("session_start", F.min("event_time").over(by_session))
        .withColumn("session_end", F.max("event_time").over(by_session))
        .withColumn(
            "is_outage_session",
            (F.col("session_start") < _ntz(OUTAGE_END))
            & (F.col("session_end") > _ntz(OUTAGE_START)),
        )
    )
