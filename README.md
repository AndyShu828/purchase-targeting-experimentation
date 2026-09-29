# Purchase Propensity Targeting & Experimentation Platform

A BNPL data-science case study on public e-commerce event data: predict who will purchase in the
next 14 days, test whether a zero-fee instalment coupon creates incremental purchases, and choose
how many coupons to send.

> **Status:** P0–P3 complete — data quality and sessionization, a Spark feature pipeline, and a
> calibrated LightGBM propensity model evaluated once on a held-out, pre-registered test set.
> Next: P4 coupon experiment.

| Phase | Scope | Status |
| --- | --- | --- |
| P0 | Environment, CSV → Parquet ingestion, development sample | Done (`v0.1-p0`) |
| P1 | Data quality, 30-minute sessionization, analysis SQL, BigQuery load, pruning experiments | Done |
| P2 | Feature pipeline: 20 features at three cutoffs, PySpark, leakage checks | Done |
| P3 | Purchase propensity: RFM and logistic baselines, LightGBM, Platt calibration, SHAP, MLflow, batch scoring | Done (`v0.3-p3`) |
| P4 | Pre-registered coupon experiment, power analysis, CUPED, profit curve | Next |
| P5 | Airflow daily scoring, monitoring | Planned |
| P6 | Dashboard and write-up | Planned |

## Key findings so far

### The model (P3)

Test set: users active in the 14 days before 2019-11-18, labelled on purchases 11/18–11/30,
evaluated once against a checklist fixed before any label was read.
→ [docs/model_report.md](docs/model_report.md)

**1. The ranking survives the move to the November sale; the probability level does not.**
LightGBM's ROC-AUC goes from 0.7657 on validation to 0.7614 on test, but after Platt calibration
on validation it still overstates the test-period purchase rate 1.59× (adjusted for the 13-day
label window) and 1.75× in the top decile, where coupons go. So P4 targets by share, which needs
only the ranking, and takes the purchase level from its control group.

**2. The simpler model wins the top decile.** LightGBM leads logistic regression on ROC-AUC, AP and
lift@20%, but in the test period's top decile the logistic is ahead: lift@10% +0.027, paired
bootstrap 95% interval [+0.008, +0.046]. The model was frozen before the test, so it stays
LightGBM.

**3. Users who have never bought rank behind lapsed buyers.** Against a past buyer who does not
come back, a future first-time buyer in the window is ranked higher only 10.7% of the time, and
first-time buyers are 57% of test buyers. RFM cannot rank them at all (ROC-AUC 0.5000); the
model's gain over RFM comes mostly from them.

Also found along the way: a category id above int32 that LightGBM silently turned into NaN when
declared categorical, and outage-corrected `*_per_day` features that the trees never used because
they tied with the raw counts on the training data (§8 of the report).

### The data (P1)

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

Features and models run under WSL / Linux (Spark needs Linux here), with `PTE_ENV=wsl`:

```bash
python scripts/build_features.py                      # feature and label tables at three cutoffs
python scripts/compare_models.py                      # baselines, LightGBM, calibration, segments on valid
python scripts/register_model.py                      # LightGBM + Platt -> MLflow, alias champion
python scripts/evaluate_test.py                       # the one test-set evaluation
python scripts/score_users.py --cutoff 2019-12-01     # batch scoring: probability and decile per user
```

CI runs `ruff check`, `ruff format --check`, `sqlfluff lint sql/` (BigQuery dialect) and `pytest`
on every pull request.

## Repository layout

```
configs/         environment settings (dev / staging / prod / wsl)
sql/analysis/    analysis and data-quality queries, BigQuery dialect
sql/ddl/         BigQuery table definition and post-load assertions
src/pte/         ingestion, features (PySpark), model (baselines, LightGBM, calibration, scoring)
scripts/         query runner, BigQuery export/load, feature build, model and scoring scripts
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
| [features.md](docs/features.md) | The three cutoffs and the 20 features |
| [leakage.md](docs/leakage.md) | Every leakage rule, and the test-set pre-registration |
| [model_report.md](docs/model_report.md) | P3 results: baselines, decay, SHAP, calibration, segments |

The model report is in English; the other design notes are currently in Chinese, and English
versions will follow with the P6 write-up.
