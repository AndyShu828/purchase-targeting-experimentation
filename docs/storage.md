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

| Format | Size | Bytes per row | Ratio |
| --- | --- | --- | --- |
| zip (download) | TODO | | |
| CSV | ~14.7 GB | ~134 | 1x |
| Parquet (zstd, 61 daily partitions) | 2.01 GB | ~20 | ~7.3x smaller |

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

- `user_session` (a UUID string) is 42% of the dataset. Its values are almost unique, so dictionary
  encoding does not help. A custom integer session ID (see sessionization) would be far cheaper;
  the original column is kept for comparison.
- `event_type` has a handful of values; dictionary + run-length encoding shrinks it to <0.1 byte/row.
- `event_time` compresses well because rows are in time order and repeated timestamps sit together:
  sort order directly affects compression.
- `price` (DOUBLE) barely compresses (1.2x); storing integer cents or DECIMAL is a possible follow-up.

## Query speed

| Query | CSV | Parquet |
| --- | --- | --- |
| Daily purchases, October | TODO s | TODO s |
