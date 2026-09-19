# 5. Case-study storms (held-out seed 0 trajectories)

- **ERA5-helped:** `2010137N10090` LAILA (2010) (LAILA 2010).
- **Typical (post-1990):** `1996163N08088` UNNAMED (1996) — replaces 1976 case.
- **Trajectories:** ground truth + **all 8 held-out systems** at each origin time × horizon.
- **`best_competitor_at_horizon`:** pooled held-out median winner for figure overlay (3h RF, 12h Stacking, 24h Stacking, 48h XGBoost).
- Column prefix per model: `sece_v2_phase3`, `stacking_ensemble`, `persistence_residual_cascade`, `random_forest`, `lightgbm`, `xgboost`, `catboost`, `cbplusmotionnn`.
- Files: `05_case_study_storm_picks.csv`, `05_case_study_trajectories.csv`.

