# Purchase Propensity Targeting & Experimentation Platform

A BNPL data-science case study on public e-commerce event data: predict who will purchase in the
next 14 days, test whether a zero-fee instalment coupon creates incremental purchases, and choose
how many coupons to send.

> **Status:** P0 and P1 complete — data quality, sessionization, six analysis queries, BigQuery
> load and pruning experiments. Next: P2 feature pipeline.

| Phase | Scope | Status |
| --- | --- | --- |
| P0 | Environment, CSV → Parquet ingestion, development sample | Done (`v0.1-p0`) |
| P1 | Data quality, 30-minute sessionization, analysis SQL, BigQuery load, pruning experiments | Done |
| P2 | Feature pipeline | Next |
| P3 | Purchase propensity model | Planned |
| P4 | Pre-registered coupon experiment, power analysis, CUPED, profit curve | Planned |
| P5 | Airflow daily scoring, monitoring | Planned |
| P6 | Dashboard and write-up | Planned |

## Key findings so far

All figures are on the full data (109,950,743 events) and link to the query that produces them.

**1. The data has a 38.4-hour purchase blackout, and it overturned an earlier conclusion.**
No purchase event was recorded from 2019-11-14 18:09:37 to 2019-11-16 08:32:34 UTC, while views and
carts ran at a record rate. Every earlier check reconciled totals, and a missing day breaks no
total, so it went unnoticed until a per-day, per-event-type check was added. Removing the sessions
that overlap it shrinks the apparent rise in cart abandonment from October to late November from
15.4pp to about 2.1pp,
moves strict cart → purchase conversion from 40.45% to 48.50%, and moved the experiment's
observation window to 11/18–11/30.
→ [docs/incident_2019_11_15.md](docs/incident_2019_11_15.md)

**2. `user_session` is a cookie, not a session, so sessions are re-cut on a 30-minute timeout.**
99.996% of `user_session` values map to one user, but the longest gap inside one is 57 days, and
that maximum grows in step with the observation window (8.62× for an 8.71× longer window) — no
timeout is being applied. Using it as a session undercounts sessions by 2.13% and overstates
session purchase conversion by 1.9% relative, in a fixed direction.
→ [docs/data_quality.md](docs/data_quality.md) (check five), [docs/sessionization.md](docs/sessionization.md)

**3. "How soon do customers buy again" depends on what counts as one purchase — by a factor of two.**
The 80th-percentile repurchase interval is 5.96 days if every purchase event counts, and 10.88 days
if purchases less than 30 minutes apart are one shopping occasion. Both are upper bounds because of
the blackout; dropping the intervals that span it gives 3.63 and 6.95 days. The absolute figures
move, the ratio does not (1.83 and 1.91), so a coupon timing rule has to name its definition.
→ [repurchase_interval_distribution.sql](sql/analysis/repurchase_interval_distribution.sql),
[repurchase_interval_outage_sensitivity.sql](sql/analysis/repurchase_interval_outage_sensitivity.sql)

Also measured:

- **A third of purchase sessions never pass through a cart.** 29.56% of purchase sessions bought a
  product the user never carted anywhere in the 61 days, so the view → cart → purchase funnel is not
  the path most purchases take here.
  → [session_funnel_purchase_decomposition.sql](sql/analysis/session_funnel_purchase_decomposition.sql)
- **Category conversion spans 39.5×**, 0.21% to 8.19% across the 338 categories with at least
  100 purchasing sessions. → [category_conversion_rate.sql](sql/analysis/category_conversion_rate.sql)
- **On BigQuery, filtering on a non-clustered column cost more than no filter** (967 MB against
  679 MB), because a columnar engine must read every column a `WHERE` clause names. Clustering on
  `event_type` cut a partition-pruned scan by a further 3.0×.
  → [docs/sql_performance.md](docs/sql_performance.md)
- **A green dry run is not a working query.** Two-argument `LOG` takes its arguments in opposite
  orders on DuckDB and BigQuery; every file dry-ran clean, and only running them for real caught it.
  → [docs/sql_dialects.md](docs/sql_dialects.md)

## Data

[REES46 eCommerce behavior data](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store),
October–November 2019 (109,950,743 events), treated as a partner merchant's web events.
Treatment effects in the experiment module (P4) will be simulated and labelled as such; they are
never reported as findings.

Known defects, documented rather than silently cleaned:

- **Purchase tracking outage**, 2019-11-14 18:09:37 → 2019-11-16 08:32:34 UTC (finding 1).
- **Exact duplicate events**, zero prices, and missing `brand` / `category_code` —
  see [docs/data_quality.md](docs/data_quality.md) for each rule and its reason.

## Reproduce

Raw data is not committed. Download `2019-Oct.csv` and `2019-Nov.csv` into `data/raw/`, then:

```bash
pip install -e ".[dev]"
python -m pte.ingest.convert convert   # CSV -> Parquet partitioned by event_date (UTC)
python -m pte.ingest.convert verify    # row-count reconciliation
python -m pte.ingest.convert sample    # 2019-10-01..07 development sample
```

Run any analysis query locally with DuckDB. Queries are written in BigQuery SQL; the runner exposes
the data as a view named `events`.

```bash
python scripts/run_sql.py sql/analysis/session_funnel.sql          # development sample
python scripts/run_sql.py sql/analysis/session_funnel.sql --full   # all 61 days
python scripts/run_sql.py -e "SELECT event_type, COUNT(*) FROM events GROUP BY 1"
```

BigQuery (October only, sandbox project): create the table with
[sql/ddl/create_events.sql](sql/ddl/create_events.sql), export with
`scripts/export_bq_parquet.py --month 2019-10`, load with `scripts/load_bq_month.sh`, then run
[sql/ddl/assert_events_loaded.sql](sql/ddl/assert_events_loaded.sql) — zero rows means the load
is correct.

CI runs `ruff check`, `ruff format --check`, `sqlfluff lint sql/` (BigQuery dialect) and `pytest`
on every pull request.

## Repository layout

```
configs/         environment settings (dev / staging / prod / wsl)
sql/analysis/    analysis and data-quality queries, BigQuery dialect
sql/ddl/         BigQuery table definition and post-load assertions
src/pte/         ingestion and config (feature, model and experiment modules to come)
scripts/         query runner, BigQuery export/load, benchmarks
tests/           unit tests
docs/            design notes and findings
```

| Document | What it answers |
| --- | --- |
| [data_quality.md](docs/data_quality.md) | Eight checks on the raw events and the cleaning rules they led to |
| [sessionization.md](docs/sessionization.md) | Why sessions are re-cut, and what the choice changes |
| [incident_2019_11_15.md](docs/incident_2019_11_15.md) | The purchase blackout: detection, impact, corrected figures |
| [sql_performance.md](docs/sql_performance.md) | BigQuery pruning experiments: bytes scanned and billed per variant |
| [sql_dialects.md](docs/sql_dialects.md) | DuckDB vs BigQuery differences, each tested on both engines |
| [storage.md](docs/storage.md) | CSV vs Parquet: 14.7 GB of CSV becomes 2.0 GB of Parquet |

Design notes are currently written in Chinese; English versions will follow with the P6 write-up.
