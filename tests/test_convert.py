from pathlib import Path

import duckdb

from pte.ingest.convert import convert, count_rows, extract_sample, parquet_source

HEADER = (
    "event_time,event_type,product_id,category_id,category_code,brand,price,user_id,user_session"
)
ROWS = [
    "2019-10-01 00:00:00 UTC,view,44600062,2103807459595387724,,shiseido,35.79,541312140,s1",
    "2019-10-01 23:59:59 UTC,purchase,3900821,2053013552326770905,appliances.kettle,aqua,33.20,"
    "554748717,s2",
    "2019-10-02 00:00:01 UTC,cart,17200506,2053013559792632471,furniture.sofa,,543.10,519107250,s3",
]


def _write_csv(tmp_path: Path) -> Path:
    path = tmp_path / "2019-Oct.csv"
    path.write_text("\n".join([HEADER, *ROWS]) + "\n", encoding="utf-8")
    return path


def test_convert_partitions_by_utc_date(tmp_path):
    csv = _write_csv(tmp_path)
    out = tmp_path / "events"
    convert([csv], out)

    partitions = sorted(p.name for p in out.iterdir())
    assert partitions == ["event_date=2019-10-01", "event_date=2019-10-02"]

    per_day = duckdb.sql(
        f"SELECT CAST(event_date AS VARCHAR), count(*) FROM {parquet_source(out)} "
        "GROUP BY 1 ORDER BY 1"
    ).fetchall()
    # 23:59:59 UTC stays on 10-01; the boundary must not shift a row to the next day.
    assert per_day == [("2019-10-01", 2), ("2019-10-02", 1)]


def test_convert_keeps_types(tmp_path):
    csv = _write_csv(tmp_path)
    out = tmp_path / "events"
    convert([csv], out)

    event_time, category_id = duckdb.sql(
        f"SELECT event_time, category_id FROM {parquet_source(out)} WHERE user_session = 's1'"
    ).fetchone()
    assert str(event_time) == "2019-10-01 00:00:00"
    assert category_id == 2103807459595387724  # no float precision loss


def test_sample_and_row_count(tmp_path):
    csv = _write_csv(tmp_path)
    events = tmp_path / "events"
    convert([csv], events)

    assert count_rows([csv], events) == (3, 3)

    sample = tmp_path / "sample"
    extract_sample(events, sample, "2019-10-01", "2019-10-01")
    assert duckdb.sql(f"SELECT count(*) FROM {parquet_source(sample)}").fetchone()[0] == 2
