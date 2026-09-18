import pytest
from run_sql import build_parser, read_sql


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
