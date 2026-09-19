# 1. Held-out comparison (locked environ SECE, all evaluated systems)

Generated `2026-09-18T22:42:04Z`. Pooled test predictions across held-out seeds `[0, 1, 2, 7, 13, 99, 123, 2024, 2026]`.
Metric: sample-level median Haversine km; 95% CI from 10,000 storm-resamples.
Wilcoxon signed-rank on paired per-storm medians vs SECE. Negative rank-biserial: SECE lower error.
Bonferroni: α=0.05 × 7 competitors → raw-p threshold 0.0071.

**Not in this table:** CNN-GRU / BLSTM were not trained in the locked held-out run. Do not invent DL numbers.

## Median km + 95% CI

| Horizon | Model | Median km | CI 95% lo | CI 95% hi | n storms | n samples |
|---------|-------|----------:|----------:|----------:|---------:|----------:|
| 3h | SECE v2 Phase3 | 4.895 | 4.714 | 5.079 | 442 | 34114 |
| 3h | Random Forest | 4.927 | 4.734 | 5.125 | 442 | 34114 |
| 3h | Stacking Ensemble | 4.945 | 4.763 | 5.123 | 442 | 34114 |
| 3h | CB+MotionNN | 5.813 | 5.701 | 5.928 | 442 | 34114 |
| 3h | LightGBM | 6.026 | 5.889 | 6.168 | 442 | 34114 |
| 3h | XGBoost | 6.411 | 6.243 | 6.58 | 442 | 34114 |
| 3h | CatBoost | 8.479 | 8.284 | 8.691 | 442 | 34114 |
| 3h | Persistence Residual Cascade | 8.872 | 8.707 | 9.058 | 442 | 34114 |
| 12h | SECE v2 Phase3 | 35.538 | 34.208 | 36.925 | 442 | 21442 |
| 12h | Stacking Ensemble | 36.146 | 34.744 | 37.701 | 442 | 21442 |
| 12h | Random Forest | 36.901 | 35.525 | 38.323 | 442 | 21442 |
| 12h | LightGBM | 36.928 | 35.498 | 38.338 | 442 | 21442 |
| 12h | XGBoost | 36.947 | 35.531 | 38.416 | 442 | 21442 |
| 12h | Persistence Residual Cascade | 37.177 | 35.704 | 38.527 | 442 | 21442 |
| 12h | CB+MotionNN | 37.413 | 36.047 | 38.949 | 442 | 21442 |
| 12h | CatBoost | 42.028 | 40.381 | 43.656 | 442 | 21442 |
| 24h | Stacking Ensemble | 100.771 | 96.887 | 104.703 | 442 | 21442 |
| 24h | XGBoost | 101.029 | 96.932 | 105.286 | 442 | 21442 |
| 24h | SECE v2 Phase3 | 101.186 | 97.301 | 104.808 | 442 | 21442 |
| 24h | LightGBM | 101.632 | 97.778 | 105.589 | 442 | 21442 |
| 24h | Persistence Residual Cascade | 101.941 | 98.131 | 105.824 | 442 | 21442 |
| 24h | Random Forest | 103.511 | 99.652 | 107.437 | 442 | 21442 |
| 24h | CB+MotionNN | 104.034 | 100.239 | 108.074 | 442 | 21442 |
| 24h | CatBoost | 107.191 | 102.555 | 111.839 | 442 | 21442 |
| 48h | XGBoost | 242.702 | 232.596 | 255.403 | 442 | 21442 |
| 48h | Persistence Residual Cascade | 243.229 | 231.65 | 255.158 | 442 | 21442 |
| 48h | LightGBM | 245.975 | 234.926 | 257.214 | 442 | 21442 |
| 48h | Stacking Ensemble | 245.977 | 235.938 | 257.183 | 442 | 21442 |
| 48h | SECE v2 Phase3 | 246.331 | 235.541 | 257.614 | 442 | 21442 |
| 48h | CB+MotionNN | 249.678 | 238.972 | 261.153 | 442 | 21442 |
| 48h | Random Forest | 251.305 | 240.752 | 262.324 | 442 | 21442 |
| 48h | CatBoost | 254.862 | 242.991 | 267.797 | 442 | 21442 |

## Wilcoxon vs SECE

| Horizon | Competitor | p (raw) | p (Bonferroni) | rank-biserial | sig raw | sig Bonf | n paired |
|---------|------------|--------:|---------------:|--------------:|:-------:|:--------:|---------:|
| 3h | Stacking Ensemble | 0.00023 | 0.001609 | -0.2084 | Y | Y | 442 |
| 3h | Persistence Residual Cascade | 0.0 | 0.0 | -1.0 | Y | Y | 442 |
| 3h | Random Forest | 0.00388 | 0.027158 | -0.164 | Y | Y | 442 |
| 3h | LightGBM | 0.0 | 0.0 | -0.9698 | Y | Y | 442 |
| 3h | XGBoost | 0.0 | 0.0 | -0.9927 | Y | Y | 442 |
| 3h | CatBoost | 0.0 | 0.0 | -1.0 | Y | Y | 442 |
| 3h | CB+MotionNN | 0.0 | 0.0 | -0.8327 | Y | Y | 442 |
| 12h | Stacking Ensemble | 0.000199 | 0.001393 | -0.2102 | Y | Y | 442 |
| 12h | Persistence Residual Cascade | 0.0 | 0.0 | -0.3938 | Y | Y | 442 |
| 12h | Random Forest | 2e-06 | 1.1e-05 | -0.2633 | Y | Y | 442 |
| 12h | LightGBM | 0.0 | 0.0 | -0.3058 | Y | Y | 442 |
| 12h | XGBoost | 0.0 | 0.0 | -0.364 | Y | Y | 442 |
| 12h | CatBoost | 0.0 | 0.0 | -0.8062 | Y | Y | 442 |
| 12h | CB+MotionNN | 0.0 | 0.0 | -0.4427 | Y | Y | 442 |
| 24h | Stacking Ensemble | 0.231501 | 1.0 | -0.0706 | N | N | 442 |
| 24h | Persistence Residual Cascade | 0.128216 | 0.897514 | -0.0835 | N | N | 442 |
| 24h | Random Forest | 0.010528 | 0.073695 | -0.1404 | Y | N | 442 |
| 24h | LightGBM | 0.432356 | 1.0 | -0.0443 | N | N | 442 |
| 24h | XGBoost | 0.406911 | 1.0 | -0.0455 | N | N | 442 |
| 24h | CatBoost | 0.0 | 0.0 | -0.4502 | Y | Y | 442 |
| 24h | CB+MotionNN | 1e-06 | 1e-05 | -0.2646 | Y | Y | 442 |
| 48h | Stacking Ensemble | 0.03741 | 0.261869 | -0.1142 | Y | N | 442 |
| 48h | Persistence Residual Cascade | 0.921883 | 1.0 | -0.0054 | N | N | 442 |
| 48h | Random Forest | 0.100564 | 0.703951 | -0.0901 | N | N | 442 |
| 48h | LightGBM | 0.214902 | 1.0 | -0.0681 | N | N | 442 |
| 48h | XGBoost | 0.961562 | 1.0 | -0.0026 | N | N | 442 |
| 48h | CatBoost | 0.0 | 0.0 | -0.3522 | Y | Y | 442 |
| 48h | CB+MotionNN | 0.03741 | 0.261869 | -0.1142 | Y | N | 442 |
