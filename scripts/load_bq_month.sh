#!/usr/bin/env bash
# Append one exported month to an existing BigQuery table, one day per call.
#
# The table must already exist: `bq load` into a missing table would create an
# unpartitioned, unclustered one from the file schema and the DDL in sql/ddl/
# would be silently bypassed. Partitioning and clustering are properties of the
# table, not of the load.
#
# Runs in WSL, where gcloud and bq are installed. The export comes from
# scripts/export_bq_parquet.py, which writes one file per day with event_date
# as a real column -- a hive-partitioned tree would lose that column here,
# since a local load does no path inference.
#
# Usage:
#   scripts/load_bq_month.sh <dataset.table> <export_dir> [project]
#
# Example:
#   scripts/load_bq_month.sh pte.events \
#     /mnt/c/Users/AndyS/purchase-targeting-experimentation/data/bq_export/events_201910
set -euo pipefail

table=${1:?usage: load_bq_month.sh <dataset.table> <export_dir> [project]}
export_dir=${2:?usage: load_bq_month.sh <dataset.table> <export_dir> [project]}
project=${3:-purchase-targeting-exp}

if ! bq --project_id="$project" show "$table" >/dev/null 2>&1; then
    echo "$table does not exist -- create it from sql/ddl/ first" >&2
    exit 1
fi

shopt -s nullglob
files=("$export_dir"/events_*.parquet)
if [ ${#files[@]} -eq 0 ]; then
    echo "no events_*.parquet under $export_dir" >&2
    exit 1
fi

echo "loading ${#files[@]} files into $table"
start=$SECONDS
for file in "${files[@]}"; do
    printf '%s ' "$(basename "$file")"
    # Only bq's progress chatter is dropped. An earlier version sent stderr to
    # /dev/null too and a schema mismatch came out as a bare filename with no
    # reason; the run also looked successful from outside, because it was piped
    # into tail and a pipeline reports the exit status of its last command.
    if ! bq --project_id="$project" load \
        --source_format=PARQUET \
        --noreplace \
        "$table" "$file" >/dev/null; then
        echo "FAILED"
        echo "load failed on $file -- table may hold a partial month" >&2
        exit 1
    fi
    echo "ok"
done
echo "done in $((SECONDS - start))s"

# Row count per day, to reconcile against the local export.
bq --project_id="$project" query --use_legacy_sql=false --format=csv \
    "SELECT event_date, COUNT(*) AS rows_loaded
     FROM \`$table\`
     GROUP BY event_date
     ORDER BY event_date"
