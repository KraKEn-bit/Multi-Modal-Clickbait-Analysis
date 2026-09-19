# Held-out final confirm — environ subset (locked architecture)

Genesis 1940-2024, Phase 3 SECE + point ERA5 + environ expert.
Seeds: held-out only (`eval_protocol.HELDOUT_SEEDS`). Dev seeds were not used.

## SECE rank-1 (held-out)

| Horizon | SECE rank-1 | n seeds |
|---------|------------:|--------:|
| 3h | 6/9 | 9 |
| 12h | 6/9 | 9 |
| 24h | 2/9 | 9 |
| 48h | 1/9 | 9 |

## SECE mean median km (held-out)

| Horizon | Mean km |
|---------|--------:|
| 3h | 4.90 |
| 12h | 35.63 |
| 24h | 101.21 |
| 48h | 246.99 |

## Rank-1 winners per horizon (held-out, all models)

- **3h**: SECE v2 Phase3(6), Random Forest(2), Stacking Ensemble(1)
- **12h**: SECE v2 Phase3(6), Stacking Ensemble(3)
- **24h**: Stacking Ensemble(3), SECE v2 Phase3(2), LightGBM(2), XGBoost(2)
- **48h**: Persistence Residual Cascade(4), XGBoost(3), SECE v2 Phase3(1), Stacking Ensemble(1)

## Dev reference (architecture selection only — different seeds)

| Horizon | Dev SECE rank-1 | Dev mean km | Held-out mean km |
|---------|----------------:|------------:|-----------------:|
| 3h | 5/9 | 4.93 | 4.90 |
| 12h | 6/9 | 35.56 | 35.63 |
| 24h | 3/9 | 99.43 | 101.21 |
| 48h | 2/9 | 246.94 | 246.99 |
