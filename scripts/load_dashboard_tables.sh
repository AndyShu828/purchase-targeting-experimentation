#!/usr/bin/env bash
# Load the dashboard's result tables into BigQuery, one table per Parquet file.
#
# The P6 dashboard reads small result tables, never the raw events
# (docs/dashboard.md section 3): the queries run locally on the full data,
# where both months fit, and only their results go up. scripts/run_sql.py
# --out writes the Parquet, so the column types are DuckDB's own rather than
# whatever a CSV would be guessed back into.
#
# Each file <name>.parquet becomes <dataset>.<name> and replaces it whole
# (--replace). The tables are results, rebuilt from the repository at any
# time, so there is no DDL for them and nothing to append to. The sandbox still
# drops every table 60 days after it is created; rerunning this script is the
# rebuild.
#
# The dataset is created here if it is missing, in the same location as pte,
# so that nothing about it exists only in a console click.
#
# Runs in WSL, where gcloud and bq are installed.
#
# Usage:
#   scripts/load_dashboard_tables.sh <parquet_dir> [dataset] [project]
#
# Example:
#   scripts/load_dashboard_tables.sh /home/andy/pte-data/dashboard
set -euo pipefail

usage="usage: load_dashboard_tables.sh <parquet_dir> [dataset] [project]"
parquet_dir=${1:?$usage}
dataset=${2:-pte_dashboard}
project=${3:-purchase-targeting-exp}
location=asia-northeast1

shopt -s nullglob
files=("$parquet_dir"/*.parquet)
if [ ${#files[@]} -eq 0 ]; then
    echo "no .parquet under $parquet_dir" >&2
    exit 1
fi

if ! bq --project_id="$project" show "$dataset" >/dev/null 2>&1; then
    echo "creating dataset $dataset in $location"
    bq --project_id="$project" --location="$location" mk --dataset "$dataset" >/dev/null
fi

for file in "${files[@]}"; do
    table="$dataset.$(basename "$file" .parquet)"
    printf '%s ' "$table"
    # stderr is kept: a failed load must say why (see load_bq_month.sh).
    if ! bq --project_id="$project" load \
        --source_format=PARQUET \
        --replace \
        "$table" "$file" >/dev/null; then
        echo "FAILED"
        echo "load failed on $file" >&2
        exit 1
    fi
    echo "ok"
done

# Row counts and column types as BigQuery holds them, to check against the
# local run: a column that left as an integer must not arrive as FLOAT64.
bq --project_id="$project" query --use_legacy_sql=false --format=pretty \
    "SELECT table_name, column_name, data_type
     FROM \`$project.$dataset\`.INFORMATION_SCHEMA.COLUMNS
     ORDER BY table_name, ordinal_position"
for file in "${files[@]}"; do
    table="$dataset.$(basename "$file" .parquet)"
    bq --project_id="$project" query --use_legacy_sql=false --format=csv \
        "SELECT '$table' AS table_name, COUNT(*) AS rows_loaded FROM \`$project.$table\`"
done
