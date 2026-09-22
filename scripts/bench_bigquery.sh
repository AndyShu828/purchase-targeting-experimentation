#!/usr/bin/env bash
# Measure pruning on pte.events: partition key, correlated column, cluster key.
#
# Every variant asks the same shape of question -- count and revenue over the
# week of 2019-10-15 -- and differs only in what the WHERE clause lets BigQuery
# skip:
#   1. no filter, the whole table, the baseline for everything below
#   2. event_date, an ordinary column: NOT the partition key
#   3. day_index, the partition key
#   4. day_index plus event_type, the leading cluster column
#
# Variants 2 and 3 must return identical rows. That is what makes any byte
# difference between them attributable to how the filter is written rather than
# to a different question, and the script checks it instead of assuming it.
#
# Both byte figures are reported, because they disagree and the disagreement is
# the point: a dry run states what would be read, the job states what was
# billed, and billing rounds up -- 82,174,774 processed bills as 82,837,504,
# exactly 79 MiB. --nouse_cache is on every run, since a cached result bills
# zero bytes and would make every variant look free.
#
# Runs in WSL. Usage: scripts/bench_bigquery.sh [project]
set -euo pipefail

project=${1:-purchase-targeting-exp}
# Job metadata is looked up by location; without it bq finds nothing and prints
# no error, which is how an earlier version of this script produced an empty
# table with a successful exit status.
location=asia-northeast1
stamp=$(date +%s)

run_variant() {
    local label=$1 filter=$2 index=$3
    local where=""
    [ -n "$filter" ] && where="WHERE $filter"
    local sql="SELECT COUNT(*) AS events, SUM(price) AS revenue
               FROM \`$project.pte.events\` $where"
    local job="bench_${stamp}_${index}"

    local dry
    dry=$(bq --project_id="$project" query --use_legacy_sql=false --dry_run "$sql" 2>&1 |
        grep -o '[0-9]\+ bytes' | grep -o '[0-9]\+')

    bq --project_id="$project" query --use_legacy_sql=false --nouse_cache \
        --job_id="$job" --format=csv "$sql" >"/tmp/$job.csv" 2>/dev/null

    local info billed elapsed
    info=$(bq --project_id="$project" --location="$location" show --format=prettyjson -j "$job")
    billed=$(echo "$info" | grep -o '"totalBytesBilled": "[0-9]*"' | grep -o '[0-9]*' | head -1)
    elapsed=$(echo "$info" | grep -o '"finalExecutionDurationMs": "[0-9]*"' | grep -o '[0-9]*' | head -1)

    printf '%-30s %13s %13s %8s   %s\n' \
        "$label" "$dry" "$billed" "${elapsed}ms" "$(tail -1 "/tmp/$job.csv")"
}

printf '%-30s %13s %13s %8s   %s\n' variant dry_run billed elapsed "events,revenue"
run_variant "1. no filter" "" 1
run_variant "2. event_date (plain column)" \
    "event_date BETWEEN DATE '2019-10-15' AND DATE '2019-10-21'" 2
run_variant "3. day_index (partition key)" "day_index BETWEEN 14 AND 20" 3
run_variant "4. day_index + event_type" \
    "day_index BETWEEN 14 AND 20 AND event_type = 'purchase'" 4
run_variant "5. brand (no index at all)" "brand = 'samsung'" 5

echo
echo "rows 2 vs 3 (must be identical):"
if diff "/tmp/bench_${stamp}_2.csv" "/tmp/bench_${stamp}_3.csv"; then
    echo "identical"
fi
