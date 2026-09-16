# Storage: CSV vs Parquet

Source: REES46, 2019-Oct and 2019-Nov.

## Row counts

| File | Rows (excluding header) |
| --- | --- |
| 2019-Oct.csv | 42,448,764 |
| 2019-Nov.csv | 67,501,979 |
| **Total** | **109,950,743** |

Parquet row count matches exactly (verified per month).

## Size

| Format | Bytes | Bytes per row | vs CSV |
| --- | ---: | ---: | ---: |
| CSV | 14,675,375,250 | 133.5 | 1x |
| Download archives (Oct `.csv.gz` + Nov `.csv.zip`) | 4,616,053,647 | 42.0 | 3.2x smaller |
| Parquet (zstd, 61 daily partitions) | 2,006,394,515 | 18.2 | 7.3x smaller |

Each archive holds a single month's CSV. Parquet is 2.3x smaller than the gzip/zip archives
and remains directly queryable: generic compressors see one long text stream, while Parquet
encodes each column separately (dictionary, run-length, delta) before applying zstd.

## Size by column (Parquet, both months)

| Column | Compressed (MB) | Uncompressed (MB) |
| --- | ---: | ---: |
| user_session | 825.4 | 4,303.1 |
| user_id | 287.2 | 744.8 |
| product_id | 274.5 | 666.4 |
| price | 211.8 | 259.5 |
| brand | 131.9 | 163.3 |
| category_id | 113.6 | 142.0 |
| category_code | 70.7 | 83.3 |
| event_time | 31.4 | 63.7 |
| event_type | 7.3 | 26.4 |

## Observations

- `user_session` (a UUID string) is 42% of the column data. Its values are almost unique, so dictionary
  encoding does not help. A custom integer session ID (see sessionization) would be far cheaper;
  the original column is kept for comparison.
- `event_type` has a handful of values; dictionary + run-length encoding shrinks it to <0.1 byte/row.
- `event_time` compresses well because rows are in time order and repeated timestamps sit together:
  sort order directly affects compression.
- `price` (DOUBLE) barely compresses (1.2x); storing integer cents or DECIMAL is a possible follow-up.

## Query speed

Daily purchase counts for October 2019 (`python scripts/bench_storage.py`),
DuckDB 1.5.5, Intel Core i5-12600K, best of 3 warm runs (files already in the OS cache),
repeated in two sessions. Both formats return identical results (31 days).

| Format | Time |
| --- | ---: |
| CSV (5.7 GB) | 2.23–2.25 s |
| Parquet | 0.039–0.047 s |
| Speed-up | ~50x (48–58x) |

Even with DuckDB's parallel CSV reader, Parquet is over an order of magnitude faster: the query
reads only 2 of 10 columns (`event_type`, `event_date`), skips the November partitions
entirely, and parses no text. At tens of milliseconds, run-to-run noise is ~20% of the Parquet
timing, so the ratio is reported as a range. Cold-cache timings were not measured.
