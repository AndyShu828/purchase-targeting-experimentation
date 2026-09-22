"""sql/analysis/ is BigQuery SQL; these checks keep the DuckDB shims honest.

The shims in run_sql.py ignore their date-part argument, so a file that asks for
TIMESTAMP_DIFF(..., DAY) would silently get seconds locally. Nothing but this
test stands between that and a green run.
"""

import re
from pathlib import Path

import duckdb
import pytest
from run_sql import BIGQUERY_SHIMS

SQL_FILES = sorted((Path(__file__).resolve().parents[1] / "sql" / "analysis").glob("*.sql"))


def strip_comments(text: str) -> str:
    return re.sub(r"--[^\n]*", "", text)


def call_arguments(text: str, function: str) -> list[list[str]]:
    """Top-level argument lists of every call to `function` in `text`."""
    calls = []
    for match in re.finditer(rf"\b{function}\s*\(", text, flags=re.IGNORECASE):
        depth, args, current = 1, [], ""
        for char in text[match.end() :]:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    break
            if char == "," and depth == 1:
                args.append(current.strip())
                current = ""
            else:
                current += char
        calls.append(args + [current.strip()])
    return calls


@pytest.mark.parametrize("path", SQL_FILES, ids=lambda p: p.name)
def test_shimmed_functions_use_only_the_supported_part(path):
    text = strip_comments(path.read_text(encoding="utf-8"))
    for function, (part, _macro) in BIGQUERY_SHIMS.items():
        for args in call_arguments(text, function):
            assert args[-1].upper() == part, f"{function}({', '.join(args)}): only {part}"


@pytest.mark.parametrize("path", SQL_FILES, ids=lambda p: p.name)
def test_no_duckdb_string_date_parts(path):
    text = strip_comments(path.read_text(encoding="utf-8"))
    found = re.findall(r"\b(?:DATE_DIFF|DATE_TRUNC|DATE_PART)\s*\(\s*'", text, re.IGNORECASE)
    assert not found, "DuckDB form with a quoted part; write the BigQuery form"


@pytest.fixture
def con():
    con = duckdb.connect()
    for _part, macro in BIGQUERY_SHIMS.values():
        con.execute(macro)
    return con


def test_timestamp_diff_is_end_minus_start_in_seconds(con):
    sql = "SELECT TIMESTAMP_DIFF(TIMESTAMP '2019-10-01 00:01:00', TIMESTAMP '2019-10-01', SECOND)"
    assert con.sql(sql).fetchone()[0] == 60


def test_timestamp_trunc_isoweek_starts_on_monday(con):
    # 2019-10-06 is a Sunday; BigQuery's WEEK would return it unchanged.
    sql = "SELECT TIMESTAMP_TRUNC(TIMESTAMP '2019-10-06 12:00:00', ISOWEEK)::DATE"
    assert str(con.sql(sql).fetchone()[0]) == "2019-09-30"
