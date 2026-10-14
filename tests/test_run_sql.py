import duckdb
import pytest
from run_sql import build_parser, read_sql, save_parquet


def test_execute_flag_supplies_the_statement():
    args = build_parser().parse_args(["-e", "SELECT 1"])
    assert read_sql(args) == ("-e", "SELECT 1")


def test_sql_file_is_read_from_disk(tmp_path):
    path = tmp_path / "q.sql"
    path.write_text("SELECT 2\n", encoding="utf-8")
    args = build_parser().parse_args([str(path)])
    assert read_sql(args) == (str(path), "SELECT 2\n")


def test_missing_file_exits_with_a_message_not_a_traceback(tmp_path):
    args = build_parser().parse_args([str(tmp_path / "nope.sql")])
    with pytest.raises(SystemExit) as exc:
        read_sql(args)
    assert "nope.sql" in str(exc.value)


@pytest.mark.parametrize("argv", [[], ["q.sql", "-e", "SELECT 1"]])
def test_parser_rejects_neither_or_both_sources(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(argv)
    assert exc.value.code == 2


def test_save_parquet_keeps_types_and_runs_the_query_once(tmp_path):
    con = duckdb.connect()
    con.execute("CREATE SEQUENCE calls")
    result = con.sql("SELECT nextval('calls') AS call, DATE '2019-10-01' AS week, 0.25 AS share")
    path = tmp_path / "sub" / "out.parquet"
    stored = save_parquet(con, result, path)
    assert stored.fetchall() == duckdb.sql(f"SELECT * FROM '{path.as_posix()}'").fetchall()
    assert stored.fetchall()[0][0] == 1
    back = duckdb.sql(f"SELECT * FROM '{path.as_posix()}'")
    assert [str(t) for t in back.types][1:] == ["DATE", "DECIMAL(3,2)"]


def test_save_parquet_writes_integer_sums_as_integers(tmp_path):
    con = duckdb.connect()
    result = con.sql("SELECT SUM(x) AS total FROM range(4) t(x)")
    assert str(result.types[0]) == "HUGEINT"
    path = tmp_path / "sum.parquet"
    save_parquet(con, result, path)
    back = duckdb.sql(f"SELECT * FROM '{path.as_posix()}'")
    assert [str(t) for t in back.types] == ["BIGINT"]
    assert back.fetchall() == [(6,)]
