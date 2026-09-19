# A Leakage-Aware, ERA5-Augmented Ensemble for Multi-Horizon Cyclone Track Prediction over the Bay of Bengal

**Target venue:** IEEE InGARSS 2026 (Disaster Management & Early Warning Systems track), or IEEE JSTARS special issue

---

## Abstract

We present SECE, a physics-informed ensemble for multi-horizon (3–48h) tropical cyclone track prediction over the Bay of Bengal, combining IBTrACS kinematic features with ERA5 atmospheric reanalysis (wind, pressure, sea surface temperature, derived steering flow and shear) through a dedicated atmospheric expert model. Evaluated under a strict train/validation/held-out protocol — designed specifically to avoid a subtle test-set leakage failure mode we identified and corrected during development — SECE significantly outperforms seven baseline systems (Stacking Ensemble, Random Forest, LightGBM, XGBoost, CatBoost, Persistence Residual Cascade, CB+MotionNN) at 3- and 12-hour horizons (Wilcoxon signed-rank, Bonferroni-corrected, p < 0.0071), and achieves statistical parity with the strongest baselines at 24 and 48 hours. Controlled ablation shows ERA5 features provide genuine, horizon-dependent value, most concentrated at 48 hours on a per-storm basis. We report this as a transparent, reproducible benchmark directly relevant to early-warning system design for Bangladesh.

---

## I. Introduction

Bangladesh's exposure to Bay of Bengal (BoB) tropical cyclones motivates continued development of lightweight, data-driven track-forecasting tools that complement resource-intensive numerical weather prediction. This paper makes three contributions: (1) an eight-system benchmark across four forecast horizons on BoB/North Indian Ocean IBTrACS data (1940–2024, 852 storms); (2) SECE, an ensemble integrating ERA5 atmospheric reanalysis via a dedicated expert model; and (3) a documented correction of test-set leakage encountered during our own development process, offered as a cautionary methodological note for the field.

## II. Data and Features

We use 852 quality-controlled IBTrACS North Indian Ocean storms (27,728 three-hourly points), of which 583 storms (15,605 samples) support valid multi-horizon targets through 48 hours and form our modeling pool. Track/physics features (49) capture storm position, motion kinematics, and derived physical proxies. ERA5 features (14) — wind at 850/500/200 hPa, mean sea level pressure, sea surface temperature, derived steering flow, and vertical wind shear — are extracted at the storm center for all track timestamps (30°N–5°N, 70°E–102°E domain) and added to a dedicated "environ" expert subset within SECE. The 1940–2024 genesis window was selected via a fixed-test-set-controlled ablation over three candidate windows, isolating genuine training-data benefit from test-composition artifacts.

## III. Addressing Test-Set Leakage

Initial architecture iterations were refined by inspecting test-set performance gaps and adding targeted fixes — a form of leakage that produced an apparent (but non-replicating) sweep across all forecast horizons on a single data split. We corrected this via a two-tier seed protocol: nine development seeds for all architecture and fusion-path decisions, and nine independent held-out seeds, generated and evaluated exactly once after the architecture was frozen. We verify this correction produced a stable result by confirming development- and held-out-seed performance are closely matched (Section V).

## IV. Method: SECE

For each horizon, SECE trains four physics-informed feature subsets (3h: position/motion/environ/full; 12–48h: motion/physics/environ/full), each modeled by four tree algorithms (CatBoost, XGBoost, LightGBM, Random Forest) and fused via NNLS into a subset-consensus signal. This signal joins seven baseline "champion" predictions in a second-stage, horizon-specific NNLS fusion; a validation-set-only router selects the best-performing fusion candidate per seed. All hyperparameters (300 estimators, depth 6, learning rate 0.01) are held uniform across systems to isolate architectural effects.

## V. Results

**Table I. Held-out median error (km), 95% bootstrap CI, n = 442 pooled storms**

| Horizon | SECE | Best competitor | Bonferroni-significant vs. |
|---|---|---|---|
| 3h | 4.90 [4.71, 5.08] | Random Forest 4.93 | **All 7 baselines** |
| 12h | 35.54 [34.21, 36.93] | Stacking 36.15 | **All 7 baselines** |
| 24h | 101.19 [97.30, 104.81] | Stacking 100.77 | CatBoost, CB+MotionNN only |
| 48h | 246.33 [235.54, 257.61] | XGBoost 242.70 | CatBoost only |

SECE is never significantly worse than any baseline at any horizon; it sweeps at short range and ties the strongest baselines at long range. Development-ablation (9 seeds) confirms this pattern: ERA5 point features improved SECE's 48h mean error from 250.26 to 246.80 km (8/9 seeds improved); a dedicated environ expert further improved 24h mean error to 99.43 km (best of all tested configurations). Spatial-patch atmospheric features (5°/3°) were tested and improved short-range rank-1 frequency but not long-range accuracy, and were not adopted.

A per-storm LightGBM diagnostic (development seed 0, n=88) shows ERA5's benefit strengthening with lead time: near-parity at 24h (48.9% of storms improved), majority benefit at 48h (61.4% improved, median +10.9 km).

## VI. Case Study

Figure 1 compares ground-truth, SECE, and the horizon-specific best-competitor track (the system with lowest *global* held-out median error at that horizon, per Table I — not necessarily the best performer on this individual storm) for two held-out storms. Cyclone Laila (2010) — separately the storm with the largest single-storm ERA5 benefit in a seed-0 LightGBM diagnostic (25.0 km at 24h, 107.0 km at 48h improvement over track-only features) — shows SECE outperforming the global-best competitor (Stacking Ensemble) locally at 24h (111.3 km vs. 128.9 km) while XGBoost, the global 48h leader, is locally stronger at 48h (204.9 km vs. SECE's 236.1 km). A representative 1996 Bay of Bengal storm (central-basin genesis, 8.4°N/87.7°E) shows a closer local contest at 24h (SECE 107.2 km vs. Stacking 110.7 km) and a wider gap favoring XGBoost at 48h (276.0 km vs. SECE 306.3 km) — consistent with the aggregate finding that SECE's advantage is concentrated at shorter horizons.

## VII. Conclusion

SECE, an ERA5-augmented, leakage-corrected ensemble, provides statistically significant short-range improvement and long-range parity over strong baselines for BoB cyclone track prediction. We release this as a reproducible, honestly-validated benchmark, including negative results, to support further data-driven early-warning research for cyclone-vulnerable regions.

---

*Compute: single workstation, Intel i7-13700HX, 15.7 GB RAM; held-out evaluation 9.4h; development ablations 4.9–10.9h each depending on configuration.*
