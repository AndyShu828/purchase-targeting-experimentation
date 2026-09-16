# Purchase Propensity Targeting & Experimentation Platform

A BNPL data-science case study built on public e-commerce event data: predict who will purchase
in the next 14 days, test whether a zero-fee instalment coupon creates incremental purchases,
choose how many coupons to send, and deliver results daily.

> Status: **P0 – environment and ingestion** (in progress)

## Data

[REES46 eCommerce behavior data](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store),
October–November 2019 (109,950,743 events), treated as a partner merchant's web events.
Treatment effects in the experiment module are simulated and labelled as such.

Raw data is not committed. To rebuild locally:

1. Download `2019-Oct.csv` and `2019-Nov.csv` into `data/raw/`.
2. Run:

```bash
pip install -e ".[dev]"
python -m pte.ingest.convert convert   # CSV -> Parquet partitioned by event_date (UTC)
python -m pte.ingest.convert verify    # row-count reconciliation
python -m pte.ingest.convert sample    # 2019-10-01..07 development sample
```

See [docs/storage.md](docs/storage.md) for the CSV vs Parquet comparison.

## Repository layout

```
configs/      environment settings (dev / staging / prod)
sql/          BigQuery DDL and analysis queries
src/pte/      ingest, quality, features, model, experiment, monitoring
dags/         Airflow DAGs
notebooks/    EDA and reports
tests/        unit tests
docs/         design notes, pre-registration, reports
```
