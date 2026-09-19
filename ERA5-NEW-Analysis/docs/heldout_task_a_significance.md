# Held-out Task A — significance vs SECE (environ Phase 3, locked)

Pooled **held-out test** predictions across seeds `[0, 1, 2, 7, 13, 99, 123, 2024, 2026]`. Per-storm median Haversine error (km); bootstrap 95% CI resamples storms; Wilcoxon signed-rank vs SECE on paired storm medians.

Bonferroni: α=0.05 × 4 competitors → threshold 0.0125 on raw p.

Predictions cache: `results\sece_era5_environ_heldout\task_a_test_predictions`
CSV: `results\sece_era5_environ_heldout\task_a_significance.csv`

## All models — median km and bootstrap 95% CI

| Horizon | Model | Median km | CI 95% lo | CI 95% hi | n storms | n samples |
|---------|-------|----------:|----------:|----------:|---------:|----------:|
| 3h | SECE v2 Phase3 | 4.895 | 4.714 | 5.079 | 442 | 34114 |
| 3h | Random Forest | 4.927 | 4.731 | 5.126 | 442 | 34114 |
| 3h | Stacking Ensemble | 4.945 | 4.763 | 5.123 | 442 | 34114 |
| 3h | LightGBM | 6.026 | 5.893 | 6.169 | 442 | 34114 |
| 3h | XGBoost | 6.411 | 6.239 | 6.583 | 442 | 34114 |
| 12h | SECE v2 Phase3 | 35.538 | 34.196 | 36.889 | 442 | 21442 |
| 12h | Stacking Ensemble | 36.146 | 34.761 | 37.689 | 442 | 21442 |
| 12h | Random Forest | 36.901 | 35.511 | 38.323 | 442 | 21442 |
| 12h | LightGBM | 36.928 | 35.537 | 38.366 | 442 | 21442 |
| 12h | XGBoost | 36.947 | 35.538 | 38.429 | 442 | 21442 |
| 24h | Stacking Ensemble | 100.771 | 96.927 | 104.784 | 442 | 21442 |
| 24h | XGBoost | 101.029 | 96.882 | 105.212 | 442 | 21442 |
| 24h | SECE v2 Phase3 | 101.186 | 97.254 | 104.868 | 442 | 21442 |
| 24h | LightGBM | 101.632 | 97.72 | 105.6 | 442 | 21442 |
| 24h | Random Forest | 103.511 | 99.672 | 107.55 | 442 | 21442 |
| 48h | XGBoost | 242.702 | 232.681 | 255.129 | 442 | 21442 |
| 48h | LightGBM | 245.975 | 234.788 | 256.836 | 442 | 21442 |
| 48h | Stacking Ensemble | 245.977 | 235.833 | 256.979 | 442 | 21442 |
| 48h | SECE v2 Phase3 | 246.331 | 235.536 | 258.049 | 442 | 21442 |
| 48h | Random Forest | 251.305 | 240.773 | 262.319 | 442 | 21442 |

## Wilcoxon vs SECE (paired per-storm medians)

| Horizon | Competitor | p (raw) | p (Bonferroni) | rank-biserial | sig raw | sig Bonf | n paired storms |
|---------|------------|--------:|---------------:|--------------:|:-------:|:--------:|----------------:|
| 3h | Stacking Ensemble | 0.00023 | 0.000919 | -0.2084 | Y | Y | 442 |
| 3h | Random Forest | 0.00388 | 0.015519 | -0.164 | Y | Y | 442 |
| 3h | LightGBM | 0.0 | 0.0 | -0.9698 | Y | Y | 442 |
| 3h | XGBoost | 0.0 | 0.0 | -0.9927 | Y | Y | 442 |
| 12h | Stacking Ensemble | 0.000199 | 0.000796 | -0.2102 | Y | Y | 442 |
| 12h | Random Forest | 2e-06 | 6e-06 | -0.2633 | Y | Y | 442 |
| 12h | LightGBM | 0.0 | 0.0 | -0.3058 | Y | Y | 442 |
| 12h | XGBoost | 0.0 | 0.0 | -0.364 | Y | Y | 442 |
| 24h | Stacking Ensemble | 0.231501 | 0.926005 | -0.0706 | N | N | 442 |
| 24h | Random Forest | 0.010528 | 0.042112 | -0.1404 | Y | Y | 442 |
| 24h | LightGBM | 0.432356 | 1.0 | -0.0443 | N | N | 442 |
| 24h | XGBoost | 0.406911 | 1.0 | -0.0455 | N | N | 442 |
| 48h | Stacking Ensemble | 0.03741 | 0.149639 | -0.1142 | Y | N | 442 |
| 48h | Random Forest | 0.100564 | 0.402258 | -0.0901 | N | N | 442 |
| 48h | LightGBM | 0.214902 | 0.859608 | -0.0681 | N | N | 442 |
| 48h | XGBoost | 0.961562 | 1.0 | -0.0026 | N | N | 442 |
