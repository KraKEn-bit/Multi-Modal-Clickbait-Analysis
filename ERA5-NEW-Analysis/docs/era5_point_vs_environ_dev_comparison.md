# Point-ERA5 vs environ-subset Phase 3 (controlled)

Same 1940-2024 rows, 9 DEV seeds, 63 features. Only change: environ subset expert added.

## SECE rank-1

| Horizon | Point-ERA5 | +Environ subset |
|---------|----------:|----------------:|
| 3h | 5/9 | 5/9 |
| 12h | 7/9 | 6/9 |
| 24h | 2/9 | 3/9 |
| 48h | 1/9 | 2/9 |

## SECE mean median km

| Horizon | Point-ERA5 | +Environ | Delta |
|---------|----------:|---------:|------:|
| 3h | 4.93 | 4.93 | +0.00 |
| 12h | 35.59 | 35.56 | -0.03 |
| 24h | 99.86 | 99.43 | -0.44 |
| 48h | 246.80 | 246.94 | +0.14 |
