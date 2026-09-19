# ERA5 vs track-only dev comparison (controlled)

Phase 3 SECE, genesis 1940-2024, same CSV rows, 9 DEV seeds.
Only difference: 14 ERA5 feature columns included or excluded.

## SECE rank-1 per horizon

| Horizon | Track-only | +ERA5 |
|---------|----------:|------:|
| 3h | 5/9 | 5/9 |
| 12h | 3/9 | 7/9 |
| 24h | 3/9 | 2/9 |
| 48h | 0/9 | 1/9 |

## SECE mean median km

| Horizon | Track-only | +ERA5 | Delta (ERA5 − track) |
|---------|----------:|------:|---------------------:|
| 3h | 4.84 | 4.93 | +0.09 |
| 12h | 35.48 | 35.59 | +0.11 |
| 24h | 100.48 | 99.86 | -0.62 |
| 48h | 250.26 | 246.80 | -3.46 |
