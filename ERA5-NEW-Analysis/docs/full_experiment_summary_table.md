# Full NEW WAY experiment summary (SECE v2 Phase 3)

Master reference for the results section. **SECE rank-1** = count of seeds where SECE has lowest test median km; **mean median km** = mean of SECE test medians across seeds.

Dev experiments use seeds `[3,4,5,6,8,9,10,11,14]`; held-out uses `HELDOUT_SEEDS` only.

CSV: `docs\full_experiment_summary_table.csv`

## SECE rank-1 by experiment

| Experiment | 3h | 12h | 24h | 48h |
|------------|----|-----|-----|-----|
| Track-only (no ERA5) | 5/9 | 3/9 | 3/9 | 0/9 |
| Point ERA5 | 5/9 | 7/9 | 2/9 | 1/9 |
| Environ subset | 5/9 | 6/9 | 3/9 | 2/9 |
| Spatial 5deg | 7/9 | 7/9 | 1/9 | 1/9 |
| Spatial 3deg | 7/9 | 7/9 | 2/9 | 3/9 |
| Held-out (locked environ) | 6/9 | 6/9 | 2/9 | 1/9 |

## SECE mean median km by experiment

| Experiment | 3h | 12h | 24h | 48h |
|------------|---:|----:|----:|----:|
| Track-only (no ERA5) | 4.84 | 35.48 | 100.48 | 250.26 |
| Point ERA5 | 4.93 | 35.59 | 99.86 | 246.80 |
| Environ subset | 4.93 | 35.56 | 99.43 | 246.94 |
| Spatial 5deg | 4.93 | 35.56 | 99.74 | 246.55 |
| Spatial 3deg | 4.93 | 35.66 | 99.42 | 247.00 |
| Held-out (locked environ) | 4.90 | 35.63 | 101.21 | 246.99 |

## Long format

| Experiment | Horizon | SECE rank-1 | n seeds | Mean median km |
|------------|---------|------------:|--------:|---------------:|
| Track-only (no ERA5) | 3h | 5/9 | 9 | 4.84 |
| Track-only (no ERA5) | 12h | 3/9 | 9 | 35.48 |
| Track-only (no ERA5) | 24h | 3/9 | 9 | 100.48 |
| Track-only (no ERA5) | 48h | 0/9 | 9 | 250.26 |
| Point ERA5 | 3h | 5/9 | 9 | 4.93 |
| Point ERA5 | 12h | 7/9 | 9 | 35.59 |
| Point ERA5 | 24h | 2/9 | 9 | 99.86 |
| Point ERA5 | 48h | 1/9 | 9 | 246.80 |
| Environ subset | 3h | 5/9 | 9 | 4.93 |
| Environ subset | 12h | 6/9 | 9 | 35.56 |
| Environ subset | 24h | 3/9 | 9 | 99.43 |
| Environ subset | 48h | 2/9 | 9 | 246.94 |
| Spatial 5deg | 3h | 7/9 | 9 | 4.93 |
| Spatial 5deg | 12h | 7/9 | 9 | 35.56 |
| Spatial 5deg | 24h | 1/9 | 9 | 99.74 |
| Spatial 5deg | 48h | 1/9 | 9 | 246.55 |
| Spatial 3deg | 3h | 7/9 | 9 | 4.93 |
| Spatial 3deg | 12h | 7/9 | 9 | 35.66 |
| Spatial 3deg | 24h | 2/9 | 9 | 99.42 |
| Spatial 3deg | 48h | 3/9 | 9 | 247.00 |
| Held-out (locked environ) | 3h | 6/9 | 9 | 4.90 |
| Held-out (locked environ) | 12h | 6/9 | 9 | 35.63 |
| Held-out (locked environ) | 24h | 2/9 | 9 | 101.21 |
| Held-out (locked environ) | 48h | 1/9 | 9 | 246.99 |
