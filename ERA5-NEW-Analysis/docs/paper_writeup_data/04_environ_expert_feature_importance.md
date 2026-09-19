# 4. Environ-subset expert feature importance (LightGBM gain)

Diagnostic retrain on held-out **seed 0** only. Same LGB hyperparameters as locked SECE (n=300, depth=6, lr=0.01).
Importance = mean LightGBM **gain** across the two MultiOutput heads (Δlat, Δlon).
This is the environ-subset expert, not the full SECE stack. Not SHAP (tree gain is the fastest locked-spec diagnostic).

## 24h — ERA5 features in the environ expert

| Rank (within ERA5+environ) | Feature | Gain | Share of environ-gain |
|---:|---------|-----:|----------------------:|
| 3 | `sst` | 20620.8579 | 0.0166 |
| 4 | `u200` | 16686.9691 | 0.0134 |
| 5 | `msl` | 15857.8 | 0.0128 |
| 6 | `steer_u` | 12512.1839 | 0.0101 |
| 8 | `u500` | 10611.8135 | 0.0086 |
| 9 | `v500` | 10592.751 | 0.0085 |
| 10 | `shear_u` | 9682.2439 | 0.0078 |
| 12 | `u850` | 8662.6385 | 0.0070 |
| 13 | `shear_v` | 8337.0716 | 0.0067 |
| 14 | `sst_missing` | 8310.2581 | 0.0067 |
| 15 | `v850` | 6012.0279 | 0.0048 |
| 16 | `v200` | 5429.6074 | 0.0044 |
| 18 | `steer_v` | 4449.3389 | 0.0036 |
| 19 | `shear_mag` | 3926.9569 | 0.0032 |

Top 8 environ-subset features (track + ERA5): `STORM_DIR`, `STORM_SPEED`, `sst`, `u200`, `msl`, `steer_u`, `LANDFALL`, `u500`

## 48h — ERA5 features in the environ expert

| Rank (within ERA5+environ) | Feature | Gain | Share of environ-gain |
|---:|---------|-----:|----------------------:|
| 3 | `sst` | 178498.4888 | 0.0415 |
| 4 | `u200` | 143092.597 | 0.0332 |
| 5 | `msl` | 139107.4402 | 0.0323 |
| 7 | `steer_u` | 69506.0304 | 0.0161 |
| 8 | `v200` | 67023.2343 | 0.0156 |
| 9 | `shear_u` | 65607.4186 | 0.0152 |
| 10 | `u500` | 60622.6944 | 0.0141 |
| 11 | `sst_missing` | 60088.0984 | 0.0140 |
| 12 | `v500` | 59068.7989 | 0.0137 |
| 13 | `steer_v` | 58175.8109 | 0.0135 |
| 15 | `u850` | 36363.5379 | 0.0084 |
| 17 | `v850` | 31782.6063 | 0.0074 |
| 18 | `shear_v` | 27633.8182 | 0.0064 |
| 19 | `shear_mag` | 19971.6811 | 0.0046 |

Top 8 environ-subset features (track + ERA5): `STORM_DIR`, `STORM_SPEED`, `sst`, `u200`, `msl`, `DIST2LAND`, `steer_u`, `v200`

