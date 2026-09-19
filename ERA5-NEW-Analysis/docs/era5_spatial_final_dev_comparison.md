# Spatial patch final dev comparison

Join runtime (spatial extraction): 3.43 h
Current best baseline (24h+48h mean km): **environ_subset**

## SECE rank-1 per horizon

| Horizon | Point ERA5 | Environ | Spatial 5deg | Spatial 3deg |
|---------|----------:|--------:|-------------:|-------------:|
| 3h | 5/9 | 5/9 | 7/9 | 7/9 |
| 12h | 7/9 | 6/9 | 7/9 | 7/9 |
| 24h | 2/9 | 3/9 | 1/9 | 2/9 |
| 48h | 1/9 | 2/9 | 1/9 | 3/9 |

## SECE mean median km

| Horizon | Point | Environ | Spatial 5deg | Spatial 3deg |
|---------|------:|--------:|-------------:|-------------:|
| 3h | 4.93 | 4.93 | 4.93 | 4.93 |
| 12h | 35.59 | 35.56 | 35.56 | 35.66 |
| 24h | 99.86 | 99.43 | 99.74 | 99.42 |
| 48h | 246.80 | 246.94 | 246.55 | 247.00 |

## Delta vs current best (environ_subset)

### Spatial 5deg
- **3h**: +0.01 km (4.93 vs 4.93)
- **12h**: -0.00 km (35.56 vs 35.56)
- **24h**: +0.31 km (99.74 vs 99.43)
- **48h**: -0.39 km (246.55 vs 246.94)

### Spatial 3deg
- **3h**: +0.01 km (4.93 vs 4.93)
- **12h**: +0.10 km (35.66 vs 35.56)
- **24h**: -0.01 km (99.42 vs 99.43)
- **48h**: +0.05 km (247.00 vs 246.94)

