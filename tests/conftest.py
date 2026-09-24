import sys
from datetime import datetime
from pathlib import Path

import pytest

# scripts/ is not a package, so tests reach it through sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture(scope="session")
def spark():
    """One local Spark session for the whole run; skipped where Spark cannot run."""
    if sys.platform == "win32":
        pytest.skip("Spark runs under WSL / Linux only")
    pytest.importorskip("pyspark")
    from pte.spark import build_session

    session = build_session(
        "pte-tests",
        master="local[1]",
        # Tiny test frames: one shuffle partition and no UI keep each test fast.
        conf={"spark.sql.shuffle.partitions": "1", "spark.ui.enabled": "false"},
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture
def make_events(spark):
    """Build an events DataFrame with the real schema from a few hand-written rows.

    Each row is a dict; missing columns are NULL, event_time may be a string such as
    "2019-11-17 23:59:59", and event_date is derived from event_time unless given.
    """
    from pyspark.sql.types import StructType

    from pte.spark import EVENTS_SCHEMA

    schema = StructType.fromDDL(EVENTS_SCHEMA)
    columns = schema.fieldNames()

    def _make(rows):
        records = []
        for row in rows:
            unknown = set(row) - set(columns)
            if unknown:
                raise KeyError(f"not an events column: {sorted(unknown)}")
            record = dict.fromkeys(columns)
            record.update(row)
            if isinstance(record["event_time"], str):
                record["event_time"] = datetime.fromisoformat(record["event_time"])
            if record["event_date"] is None and record["event_time"] is not None:
                record["event_date"] = record["event_time"].date()
            records.append(tuple(record[c] for c in columns))
        return spark.createDataFrame(records, schema)

    return _make
