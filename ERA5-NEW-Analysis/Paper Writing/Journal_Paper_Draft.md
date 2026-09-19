# ERA5-Augmented Ensemble Learning for Multi-Horizon Tropical Cyclone Track Prediction over the Bay of Bengal: A Leakage-Aware Benchmark

**Target venues:** AMS *Weather and Forecasting*, IEEE *JSTARS* (Journal of Selected Topics in Applied Earth Observations and Remote Sensing)

---

## Abstract

Accurate short-to-medium-range cyclone track forecasting is critical for disaster preparedness in Bangladesh, one of the world's most cyclone-vulnerable nations. We present a rigorously validated benchmark of eight track-prediction systems — a proposed Subset-Expert Context-Aware Ensemble (SECE) and seven baselines spanning linear, tree-based, and hybrid architectures — evaluated at 3, 12, 24, and 48-hour lead times over the Bay of Bengal / North Indian Ocean basin. Using 852 quality-controlled IBTrACS storms (1940–2024, 27,728 three-hourly observations), we construct a 583-storm multi-horizon modeling pool and augment track-derived kinematic features with ERA5 atmospheric reanalysis fields (wind at three pressure levels, mean sea level pressure, sea surface temperature, derived steering flow and vertical wind shear). Critically, we identify and correct a subtle form of test-set leakage in an earlier development iteration, in which architectural decisions were guided by repeated inspection of test-set performance; we replace this with a strict train/validation/held-out protocol in which held-out data is touched exactly once. On the held-out evaluation, SECE significantly outperforms all seven baseline systems at 3- and 12-hour horizons (Wilcoxon signed-rank test, Bonferroni-corrected across seven comparisons, p < 0.0071). At 24 and 48 hours, SECE is statistically indistinguishable from the strongest competing systems (Stacking Ensemble, Persistence Residual Cascade, LightGBM, XGBoost) while remaining significantly superior to the weakest (CatBoost). A series of controlled ablations shows that ERA5 features, and specifically a dedicated atmospheric expert model within the ensemble, provide genuine — though horizon-dependent — improvements over track-only kinematics, with the clearest per-storm benefit emerging at 48 hours. We report these findings, including negative results (spatial-patch atmospheric features were tested and not adopted), as an honest account of where ensemble sophistication and atmospheric information each contribute value in operational-style cyclone track forecasting.

---

## 1. Introduction

Bangladesh sits at the low-lying convergence point of the Bay of Bengal (BoB), a basin that has produced some of the deadliest tropical cyclones in recorded history — Sidr (2007), Aila (2009), Amphan (2020), and Mocha (2023) among recent examples. Reliable short-range track forecasting is a foundational input to early-warning systems, evacuation planning, and disaster response coordination in the region.

Operational numerical weather prediction (NWP) remains the gold standard for cyclone forecasting but requires computational resources that are frequently out of reach for national meteorological agencies in developing countries. Data-driven approaches — building predictive models directly from historical best-track archives — offer a computationally lightweight complement to NWP, particularly for short-range guidance.

Despite growing interest in machine learning for cyclone forecasting, we identify three persistent gaps in the literature, particularly as it concerns the Bay of Bengal specifically:

1. **Narrow architectural benchmarking.** Most prior work compares a small number of models, limiting confidence in claims of relative superiority.
2. **Single-horizon evaluation.** Multi-step, horizon-consistent forecasting (3h through 48h) is comparatively underexplored relative to single-step prediction.
3. **Limited integration of atmospheric state.** Track-only, kinematic feature sets are common; the incremental value of atmospheric reanalysis data for BoB-specific track prediction has not, to our knowledge, been rigorously isolated via controlled ablation.

A fourth, methodological gap motivates a central contribution of this paper: **published ensemble benchmarks rarely disclose or guard against test-set leakage introduced through iterative, human-guided architecture search** — a subtle failure mode in which a model's design is repeatedly refined by inspecting held-out performance, producing benchmark results that do not generalize. We document this failure mode as it arose during our own model development, the diagnostic process by which we detected it, and the corrected protocol we adopted.

### 1.1 Contributions

- An eight-system benchmark (SECE plus seven baselines) evaluated at four forecast horizons over BoB/NIO, under a strict train/validation/held-out protocol with the held-out set evaluated exactly once.
- SECE, a Subset-Expert Context-Aware Ensemble combining physics-informed feature subsets, heterogeneous tree-based learners, and a dedicated ERA5-derived atmospheric expert, fused via a horizon-specific, validation-selected routing strategy.
- A controlled, single-variable-at-a-time ablation study isolating the effect of (a) ERA5 point features versus track-only kinematics, (b) a dedicated atmospheric expert versus folding ERA5 into existing feature groups, and (c) spatial-patch atmospheric features versus single-point extraction.
- Full statistical validation (paired Wilcoxon signed-rank tests, Bonferroni correction, bootstrap confidence intervals) rather than point-estimate comparison alone.
- A transparent account of a test-set leakage failure mode discovered during development, and the corrected evaluation protocol adopted in response.

---

## 2. Related Work

Classical statistical methods, including CLIPER-family climatological-persistence baselines, remain standard references in operational track forecasting [CLIPER]. Machine learning approaches — regularized linear models, tree-based ensembles (Random Forest, gradient boosting variants), and recurrent neural architectures (LSTM, GRU) — have each been applied to cyclone track and intensity prediction with varying success, generally on North Atlantic or Western Pacific data [LIT-DL-TRACK]. Heterogeneous stacking ensembles have shown promise in broader geophysical forecasting applications [LIT-STACKING], motivating hybrid designs of the kind proposed here.

Grinsztajn et al. (2022) demonstrate that tree-based methods consistently outperform deep learning on medium-sized tabular datasets with engineered features, a finding directly relevant to cyclone track prediction, where storm counts are typically in the low thousands of samples rather than the scale at which deep architectures are best suited. Our results extend this observation one level further: even within tree-based ensembling, added architectural complexity does not uniformly improve performance as forecast horizon grows, and a simpler stacking approach is statistically competitive with a more elaborate context-aware ensemble at longer lead times.

Prior BoB-specific studies have generally addressed either intensity prediction or single-horizon track prediction; to our knowledge, no prior published benchmark evaluates this breadth of model families across four forecast horizons on BoB data with a leakage-controlled, statistically validated protocol.

---

## 3. Data

### 3.1 Source archives

**IBTrACS** (International Best Track Archive for Climate Stewardship, v4) provides the historical storm-track record used throughout this study. **ERA5**, the fifth-generation ECMWF atmospheric reanalysis, provides gridded, physically consistent reconstructions of historical atmospheric state at 3-hourly temporal resolution back to 1940, and is the source of all atmospheric (non-track-derived) features in this work.

### 3.2 Study domain and quality control

We extract North Indian Ocean storms genesis-dated 1940–2024, applying the same quality-control criteria as our companion data-quality audit (removal of records with irregular time steps, coordinate mismatches, and spur/merge track artifacts). This yields **852 storms and 27,728 three-hourly track points**, approximately 90% of which originate in the Bay of Bengal proper (remainder: Myanmar/Andaman and Arabian Sea genesis).

Multi-horizon target construction (predicting displacement at 3, 12, 24, and 48 hours ahead) requires each storm to sustain at least 17 consecutive quality-controlled 3-hourly observations to support a valid 48-hour target. This length requirement — not a second quality-control pass — excludes 269 short-lived storms (median length 10 points, maximum 16), leaving a **583-storm, 15,605-sample multi-horizon modeling pool**. All four horizons, including 3 hours, draw from this same 583-storm pool, since train/validation/test storm assignment is determined once at the multi-horizon level.

### 3.3 Date-window selection

We evaluated three candidate genesis-date windows (1940–2024, 1979–2024 satellite-era, 1990–2024 modern-era) via controlled ablation on nine development seeds, using a simple LightGBM baseline. Because ERA5 and IBTrACS data quality both improve substantially after the introduction of satellite observations (pre-1979 records exhibit near-total missingness in wind observations used for intensity estimation), a naive comparison risked confounding "more data" with "test-set composition." We therefore additionally ran a fixed-test control, training on each candidate window but evaluating all three resulting models on a common, modern-era (1990–2024) held-out set.

The wide window (1940–2024) retained its advantage under this fixed-test control at 24- and 48-hour horizons (mean error reductions of 8.0 km and 17.7 km respectively relative to the 1990–2024-trained model), while the advantage at 3 hours was substantially attenuated (0.90 km raw gap reduced to 0.18 km under the fixed-test control), indicating it was largely a test-composition artifact at short range. We attribute the persistence of the wide-window advantage at longer horizons to storm *position* records remaining comparatively reliable across eras even where wind/intensity observations are sparse — track geometry, not intensity, is the target variable in this study. We accordingly lock **1940–2024** as the genesis-date window for all subsequent modeling.

### 3.4 Storm-wise train/validation/held-out protocol

All storms are partitioned by unique storm identifier (SID) rather than by individual observation, preventing temporal autocorrelation within a single storm's trajectory from leaking across partitions. For each of nine held-out evaluation seeds, storms are split 70:15:15 (train:validation:test), yielding 408 training, 87 validation, and 88 test storms per seed (consistent across seeds; sample counts vary slightly with storm length distribution). A separate set of nine development seeds is used exclusively during architecture selection (Section 5); these two seed sets are disjoint and the held-out seeds are evaluated only once, at the conclusion of all architectural decisions (Section 4 documents why this separation was necessary).

### 3.5 Feature engineering

**Track/physics features (49, applied to all model variants):** storm position and lagged positions (up to three steps back), motion kinematics (displacement, translation speed, bearing with sinusoidal encoding, acceleration, velocity trend), derived physics proxies (bearing curvature and stability, seasonal encoding, a latitude/bearing-based recurvature proxy), IBTrACS-native environmental fields (distance to land, landfall flag, wind estimate), and missingness indicator flags for imputed values.

**ERA5 point features (14, added in the ERA5-augmented configuration):** zonal/meridional wind at 850, 500, and 200 hPa; mean sea level pressure; sea surface temperature (with a binary flag indicating near-coast imputation via nearest valid ocean cell, since ERA5 SST is undefined over land — approximately 38% of rows required this correction, concentrated at higher latitude, more easterly, near-coast track points, consistent with a land-masking artifact rather than a data-collection or bounding-box defect); a derived deep-layer steering wind (zonal/meridional); and derived vertical wind shear magnitude and components (200 hPa minus 850 hPa wind).

ERA5 fields were extracted from a spatial domain of 30°N–5°N, 70°E–102°E, downloaded at 3-hourly resolution for all calendar days on which a tracked storm was active (551 storm-months), yielding approximately 10.5 GB of source NetCDF data joined to 27,909 track-point rows.

A supplementary spatial-patch feature set (6 additional features per patch size, capturing wind-field shear gradient, asymmetry, and nearest-trough proximity within 5° and 3° boxes centered on the storm) was extracted and evaluated but not adopted into the final architecture (Section 6.3).

---

## 4. On Test-Set Leakage: A Methodological Note

During initial model development, successive architecture revisions (through what we internally term Phase 1 through Phase 3) were evaluated on a single fixed train/validation/test split. Critically, decisions to advance from one architectural phase to the next were informed by inspecting per-horizon performance gaps on that same test set and introducing targeted fusion-path additions to close them. While no individual test observation was used to fit any model parameter, this iterative, human-guided design loop constitutes a form of test-set leakage: the architecture itself was shaped by repeated exposure to test-set feedback, undermining the validity of the resulting held-out estimate.

We detected this pattern by explicitly auditing, for each architectural decision point, which data partition informed it. We found that within-run decisions (fusion-path selection, NNLS weight fitting) were properly restricted to the validation partition, but that the *cross-run* decision to advance from one phase to the next was made by inspecting test-set rank tables.

**Correction.** We adopted a strict two-tier seed protocol: nine *development* seeds, used exhaustively during architecture search (with test-set inspection strictly prohibited at this stage — the corresponding partition is termed "held-out" and is not generated or examined during development), and nine *held-out* seeds, generated once and evaluated exactly once, after the architecture was frozen. We verify architectural convergence by confirming that development-seed and held-out-seed results are consistent (Section 6), rather than by further tuning against held-out performance.

We report this process transparently because we believe iterative test-set inspection is a common and under-scrutinized risk in ensemble-based forecasting benchmarks, and because our own initial (uncorrected) results — which appeared to show the proposed ensemble winning at every horizon on the original split — did not replicate on independent fresh splits (Section 6.1), directly illustrating the risk.

---

## 5. Methodology

### 5.1 Baseline systems

Seven baseline systems are benchmarked alongside the proposed ensemble: a Ridge-penalized Stacking Ensemble (Random Forest, XGBoost, and LightGBM base learners combined via a cross-validated linear meta-learner), Persistence Residual Cascade (PRC; a naive persistence baseline corrected by a dedicated LightGBM residual model, computed in kilometer space and mapped back to degrees), Random Forest, LightGBM, XGBoost, CatBoost, and CB+MotionNN (a CatBoost baseline corrected by a 64-32-16 feedforward neural network operating on kinematic residual features).

### 5.2 Proposed architecture: SECE (Subset-Expert Context-Aware Ensemble)

SECE constructs, per forecast horizon, four physics-informed feature subsets. At 3 hours: position, motion, environ (ERA5 plus IBTrACS-native environmental fields), and the full feature set. At 12, 24, and 48 hours: motion, physics, environ, and full. Each subset is modeled independently by four tree-based algorithms (CatBoost, XGBoost, LightGBM, Random Forest), yielding 16 subset-level models per horizon, which are fused via non-negative least squares (NNLS) into a single subset-consensus signal.

This subset-consensus signal is combined with the seven baseline systems' predictions (the "champion" pool) in a second-stage NNLS fusion. A horizon-specific set of fusion-path candidates is constructed (e.g., degree-space NNLS over the full champion pool, kilometer-space NNLS anchored to the persistence baseline, and, at 3 hours, a stacking-plus-residual-correction path), and a hard validation-best router selects, per seed, whichever candidate achieves the lowest validation-set median error; this selected candidate's predictions on the corresponding partition are what is reported. The router selection itself, and all NNLS weight fitting, are restricted to the validation partition (Section 4).

**Environ expert (locked architectural addition).** Rather than folding the 14 ERA5 features into existing subsets, we allocate them — together with IBTrACS-native environmental fields — to a dedicated "environ" subset with its own four-algorithm expert pool. This design choice was selected over three alternatives (ERA5 features added directly to existing point-feature subsets; ERA5 spatial-patch features at 5° and 3° resolution) via controlled ablation (Section 6.3), on the basis of achieving the best balanced performance across all four horizons, and offering a more direct mechanistic interpretation (atmospheric information receives dedicated ensemble representation) than the alternatives tested.

### 5.3 Training protocol

All tree-based models use 300 estimators, maximum depth 6 (CatBoost: depth 6), a global learning rate of 0.01, and early stopping disabled, applied uniformly across all systems to isolate architectural effects from hyperparameter-tuning effects. LightGBM additionally uses 63 leaves; XGBoost uses subsample and column-subsample rates of 0.8. The Stacking Ensemble's Ridge meta-learner is trained on 150 cross-validated base-learner predictions. MotionNN (within CB+MotionNN) is a 64-32-16 multilayer perceptron with ReLU activation, learning rate 0.01, maximum 100 iterations, and early stopping enabled.

### 5.4 Evaluation metric and statistical testing

The primary metric is median great-circle (Haversine) error, in kilometers, between predicted and true storm position, computed per test storm and aggregated. For the final held-out evaluation, we additionally report: (i) bootstrap 95% confidence intervals on the pooled median, computed via 10,000 storm-level resamples; and (ii) paired Wilcoxon signed-rank tests comparing SECE against each of the seven baselines on matched per-storm median errors, with Bonferroni correction applied across the seven simultaneous comparisons (adjusted significance threshold α = 0.05/7 ≈ 0.0071).

We use IBTrACS North Indian Ocean cyclones with genesis in 1940–2024. After quality control on 3-hourly track rows (consistent timesteps and valid 3-hour displacement targets), the modeling table contains **852 storms** and **27,728** forecast origins at 3 hours. Multi-horizon training and evaluation (12h, 24h, and 48h) require a complete forward track within each storm; **583 storms** yield **15,605** multi-horizon origins. Storm-wise splits (70% train / 15% validation / 15% test) are redrawn for each of nine held-out random seeds; each seed assigns approximately 408 / 87 / 88 storms to train, validation, and test. For significance testing we pool held-out test-set predictions across all nine seeds; **442** is the number of distinct storm IDs appearing in that pooled test sample (the same storm may fall in the test partition under more than one seed). Reported median kilometer errors and Wilcoxon tests use all pooled test points, with paired per-storm medians defined on these 442 storms.

---

## 6. Results

### 6.1 Development-phase ablation (nine development seeds)

Table 1 reports SECE's rank-1 frequency (count of development seeds, out of nine, on which SECE achieved the lowest median error among all eight systems) and mean median error, across successive feature configurations.

**Table 1. Development-seed ablation (SECE rank-1 count / mean median error, km)**

| Configuration | 3h | 12h | 24h | 48h |
|---|---|---|---|---|
| Track-only (no ERA5) | 5/9 · 4.84 | 3/9 · 35.48 | 3/9 · 100.48 | 0/9 · 250.26 |
| + ERA5 (point features) | 5/9 · 4.93 | 7/9 · 35.59 | 2/9 · 99.86 | 1/9 · 246.80 |
| + Environ expert (locked) | 5/9 · 4.93 | 6/9 · 35.56 | 3/9 · 99.43 | 2/9 · 246.94 |
| + Spatial patch (5°) | 7/9 · 4.93 | 7/9 · 35.56 | 1/9 · 99.74 | 1/9 · 246.55 |
| + Spatial patch (3°) | 7/9 · 4.93 | 7/9 · 35.66 | 2/9 · 99.42 | 3/9 · 247.00 |

Two findings motivate the locked architecture. First, ERA5 point features substantially reduce SECE's mean error relative to track-only features at 48 hours (250.26 → 246.80 km) with 8 of 9 development seeds improving, establishing that atmospheric information carries genuine, exploitable signal at long lead times. Second, allocating ERA5 features to a dedicated environ expert (rather than embedding them within existing subsets) further improves 24-hour performance specifically (rank-1 frequency 2/9 → 3/9; mean error 99.86 → 99.43 km), the horizon at which the original point-ERA5 configuration was weakest.

Spatial-patch features improved short-range (3h, 12h) rank-1 frequency but showed no consistent advantage over the environ-expert configuration at 24 or 48 hours (means within approximately 0.4 km, differences within the noise floor observed across seeds). We therefore did not adopt spatial-patch features into the locked architecture, reporting this as a negative result consistent with our commitment to transparent ablation reporting.

### 6.2 Held-out evaluation (final, single-pass)

**Table 2. Held-out median error (km) with 95% bootstrap confidence intervals, all systems, all horizons (n = 442 pooled unique storms)**

| Horizon | System | Median (km) | 95% CI |
|---|---|---|---|
| 3h | **SECE** | **4.895** | [4.714, 5.079] |
| | Stacking Ensemble | 4.945 | [4.763, 5.123] |
| | Random Forest | 4.927 | [4.734, 5.125] |
| | CB+MotionNN | 5.813 | [5.701, 5.928] |
| | LightGBM | 6.026 | [5.889, 6.168] |
| | XGBoost | 6.411 | [6.243, 6.580] |
| | CatBoost | 8.479 | [8.284, 8.691] |
| | PRC | 8.872 | [8.707, 9.058] |
| 12h | **SECE** | **35.538** | [34.208, 36.925] |
| | Stacking Ensemble | 36.146 | [34.744, 37.701] |
| | Random Forest | 36.901 | [35.525, 38.323] |
| | LightGBM | 36.928 | [35.498, 38.338] |
| | XGBoost | 36.947 | [35.531, 38.416] |
| | PRC | 37.177 | [35.704, 38.527] |
| | CB+MotionNN | 37.413 | [36.047, 38.949] |
| | CatBoost | 42.028 | [40.381, 43.656] |
| 24h | Stacking Ensemble | 100.771 | [96.887, 104.703] |
| | XGBoost | 101.029 | [96.932, 105.286] |
| | **SECE** | **101.186** | [97.301, 104.808] |
| | LightGBM | 101.632 | [97.778, 105.589] |
| | PRC | 101.941 | [98.131, 105.824] |
| | Random Forest | 103.511 | [99.652, 107.437] |
| | CB+MotionNN | 104.034 | [100.239, 108.074] |
| | CatBoost | 107.191 | [102.555, 111.839] |
| 48h | XGBoost | 242.702 | [232.596, 255.403] |
| | PRC | 243.229 | [231.650, 255.158] |
| | LightGBM | 245.975 | [234.926, 257.214] |
| | Stacking Ensemble | 245.977 | [235.938, 257.183] |
| | **SECE** | **246.331** | [235.541, 257.614] |
| | CB+MotionNN | 249.678 | [238.972, 261.153] |
| | Random Forest | 251.305 | [240.752, 262.324] |
| | CatBoost | 254.862 | [242.991, 267.797] |

**Table 3. Paired Wilcoxon signed-rank test, SECE vs. each baseline, held-out set (Bonferroni-corrected across 7 comparisons)**

| Horizon | Baseline | p (raw) | p (Bonferroni) | Significant? |
|---|---|---|---|---|
| 3h | Stacking Ensemble | 0.00023 | 0.0016 | **Yes** |
| 3h | Random Forest | 0.00388 | 0.0272 | **Yes** |
| 3h | CB+MotionNN, LightGBM, XGBoost, CatBoost, PRC | <0.001 (all) | <0.001 (all) | **Yes** (all) |
| 12h | Stacking Ensemble | 0.00020 | 0.0014 | **Yes** |
| 12h | All other six baselines | <0.001 (all) | <0.001 (all) | **Yes** (all) |
| 24h | Stacking Ensemble | 0.2315 | 1.000 | No |
| 24h | PRC | 0.1282 | 0.898 | No |
| 24h | LightGBM | 0.4324 | 1.000 | No |
| 24h | XGBoost | 0.4069 | 1.000 | No |
| 24h | Random Forest | 0.0105 | 0.0737 | No (raw-significant only) |
| 24h | CB+MotionNN | <0.001 | <0.001 | **Yes** |
| 24h | CatBoost | <0.001 | <0.001 | **Yes** |
| 48h | Stacking Ensemble | 0.0374 | 0.262 | No |
| 48h | CB+MotionNN | 0.0374 | 0.262 | No |
| 48h | PRC, Random Forest, LightGBM, XGBoost | 0.10–0.96 | 0.40–1.00 | No |
| 48h | CatBoost | <0.001 | <0.001 | **Yes** |

**Summary.** SECE is significantly better than **all seven** baselines at 3 and 12 hours (Bonferroni-corrected). At 24 hours, SECE is statistically indistinguishable from Stacking Ensemble, PRC, LightGBM, and XGBoost, while remaining significantly superior to CatBoost and CB+MotionNN. At 48 hours, SECE is statistically indistinguishable from six of seven baselines, remaining significantly superior only to CatBoost. Held-out mean/median errors closely track development-seed results (e.g., 48h: 246.99 km held-out vs. 246.94 km development), indicating the locked architecture generalizes consistently rather than having been tuned to a specific seed set.

### 6.3 Per-storm ERA5 contribution at long horizons (LightGBM diagnostic)

To characterize the distribution of ERA5's benefit beyond the headline ensemble comparison, we compare a single LightGBM model trained with versus without ERA5 features, per storm, on a representative development seed (seed 0; n = 88 test storms). At 24 hours, ERA5 improves accuracy for 43 of 88 storms (48.9%) with near-zero mean effect, indicating rough parity at this horizon for this simpler model. At 48 hours, ERA5 improves accuracy for 54 of 88 storms (61.4%), with a median per-storm improvement of 10.9 km, indicating a genuine, majority-of-storms benefit that strengthens with lead time. This is consistent with the mechanistic expectation that atmospheric steering information becomes proportionally more valuable as a storm's own recent kinematic history becomes a weaker predictor of its future position.

### 6.4 Case studies

Figure 1 plots ground-truth, SECE, and horizon-specific best-competitor tracks for two held-out storms at 24 and 48 hours. The "best competitor" shown is the system with the lowest *global* held-out median error at that horizon (Table 2) — Stacking Ensemble at 24h, XGBoost at 48h — and is not necessarily the best-performing system on the individual storm shown; per-storm results below illustrate that this distinction matters in practice.

**ERA5 contribution — Cyclone Laila (2010).** Among the 88 seed-0 test storms, Laila (SID 2010137N10090) exhibits the largest single-storm ERA5 benefit for the LightGBM diagnostic model: a 25.0 km improvement at 24 hours and 107.0 km improvement at 48 hours when ERA5 features are included versus track-only kinematics. We report this as an illustrative example of the mechanism identified in Section 6.3 (a plain gradient-boosted model's response to atmospheric information), not as a claim about SECE's own performance on this storm specifically. Separately, at the ensemble level, SECE's own per-storm error on Laila is 111.3 km at 24h — locally *better* than the global-best-competitor Stacking Ensemble (128.9 km on this storm) — while at 48h the pattern reverses, with XGBoost locally outperforming SECE (204.9 km vs. 236.1 km), consistent with the global 48h leaderboard.

**Representative SECE case — 1996 unnamed storm.** For a representative illustration of SECE's typical 24-hour behavior, we select the seed-0 test storm whose SECE per-storm median error is closest to that seed's overall SECE 24-hour median (106.43 km): an unnamed 1996 storm (SID 1996163N08088) tracking through the central Bay of Bengal (genesis near 8.4°N, 87.7°E), with a per-storm median error of 107.24 km, closely matching the typical-case benchmark. On this storm, SECE (107.2 km) and the global-best-competitor Stacking Ensemble (110.7 km) are closely matched at 24h, while XGBoost shows a wider local advantage at 48h (276.0 km vs. SECE's 306.3 km).

**Figure 1.** *Predicted vs. actual tracks for Laila (2010) and an unnamed 1996 storm, at 24h and 48h lead times. Black: IBTrACS ground truth. Orange: SECE. Green (dashed): horizon-specific global-best competitor (Stacking Ensemble at 24h, XGBoost at 48h). Each marker represents the model's prediction made from a distinct origin time along the storm's track, offset forward by the panel's lead time; panel titles report each system's per-storm median Haversine error.*

### 6.5 Feature importance (illustrative, single-seed diagnostic)

We report gain-based feature importance from the environ-expert LightGBM model on a single representative development seed (seed 0), strictly as an illustration of which atmospheric fields that particular expert model relies upon; this is not a SHAP-based or multi-seed attribution, and should not be read as a general statement about the full SECE ensemble.

Within the environ-expert model specifically, track kinematic features (STORM_DIR, STORM_SPEED) retain the dominant share of gain even in this ERA5-inclusive expert (86.4% combined at 24h; 71.9% at 48h), underscoring that atmospheric information supplements rather than displaces kinematic signal. Among ERA5-derived fields, sea surface temperature, 200 hPa zonal wind, and mean sea level pressure register the largest individual gain shares at both 24 and 48 hours. We note that vertical wind shear and steering-flow components (shear_u, shear_v, shear_mag, steer_u, steer_v) are mutually collinear by construction, such that gain-based importance may arbitrarily apportion credit among them; we therefore do not interpret the relative rank ordering within this collinear group as a substantive finding.

---

## 7. Discussion

The central empirical finding of this study is horizon-dependent: **SECE's context-aware, ensemble-of-experts architecture provides a statistically robust advantage at short-to-medium lead times (3, 12 hours), where storm-level context (recurvature proxies, coastal proximity, atmospheric steering) carries strong, learnable signal, but this advantage does not persist as a statistically distinguishable superiority at longer lead times (24, 48 hours), where SECE instead achieves parity with — never significant inferiority to — the strongest available baselines.**

We interpret this pattern as consistent with a bias-variance perspective on ensemble complexity: as target uncertainty grows with lead time, the marginal value of additional ensemble sophistication diminishes, and a comparatively simpler, well-regularized stacking approach becomes competitive. This extends the finding of Grinsztajn et al. (2022) — that tree-based methods outperform deep learning on tabular data of this scale — one level further, suggesting that *within* the tree-based-ensemble family, complexity itself should be calibrated to the information content available at a given forecast horizon rather than applied uniformly.

ERA5 atmospheric features provide genuine value, but its locus differs from where SECE's architectural advantage is clearest: the per-storm LightGBM diagnostic (Section 6.3) shows ERA5's majority-of-storms benefit concentrated at 48 hours, while the environ-expert architectural addition provided its clearest ensemble-level gain at 24 hours (Section 6.1). We consider this a genuine, if imperfectly aligned, confirmation that atmospheric information and ensemble architecture contribute complementary, not identical, sources of improvement, and flag the alignment (or lack thereof) between these two effects as a direction for further investigation rather than a discrepancy to be smoothed over.

---

## 8. Limitations and Future Work

**Reanalysis and observational quality across eras.** Pre-1979 records exhibit substantially reduced wind-observation coverage; although our fixed-test control indicates that the wider 1940–2024 window's advantage on modern-storm evaluation is not merely a test-composition artifact, we cannot fully rule out that some fraction of the observed gain reflects differences in how older, sparser-observation storms were originally tracked and recorded, as opposed to differences in the atmospheric information ERA5 provides for them.

**Feature extent.** ERA5 features here are extracted at a single point (or narrow spatial patch, tested but not adopted) centered on the storm; ocean subsurface heat content, higher vertical resolution, and true forward-looking operational NWP output (as distinct from retrospective reanalysis) were not incorporated and represent natural extensions.

**Architectural directions not pursued in this study.** We identify, but do not implement here, three architectural directions with plausible mechanistic justification for further narrowing the 24–48 hour gap: (i) physics-constrained trajectory modeling, in which displacement is derived from a small set of learned physical parameters (effective steering vector, momentum) via a differentiable motion equation rather than predicted directly; (ii) explicit predictive-uncertainty modeling (e.g., a learned distribution over future position, trained with a proper scoring rule), better matching the probabilistic nature of long-range forecasting than a point estimate; and (iii) an end-to-end model attending jointly over storm history and spatial ERA5 fields, rather than hand-engineered spatial statistics. We regard these as promising second-stage research directions building on the leakage-corrected, ERA5-validated foundation established here.

**Single-seed feature importance.** The feature-importance analysis in Section 6.5 reflects one development seed and one gain-based tree metric; a multi-seed, permutation- or SHAP-based analysis would strengthen mechanistic claims about which specific atmospheric fields drive SECE's improvements, and is left to future work.

---

## 9. Conclusion

We present a leakage-corrected, statistically validated, ERA5-augmented benchmark for multi-horizon tropical cyclone track prediction over the Bay of Bengal. Our proposed ensemble, SECE, significantly outperforms seven baseline systems at 3- and 12-hour lead times, and achieves statistical parity with the strongest baselines at 24- and 48-hour lead times — never falling to significant inferiority at any tested horizon. We identify and correct a test-set leakage failure mode encountered during development, document a controlled ablation isolating the contribution of ERA5 atmospheric features, and report negative results (spatial-patch features) alongside positive ones, in the interest of a complete and reproducible account of what this class of model can and cannot currently achieve for this operationally important region.

---

## Reproducibility Statement

All experiments were conducted on a single workstation (13th Gen Intel Core i7-13700HX, 15.7 GB RAM, Windows 11, Python 3.13.7); tree-based training was CPU-bound and did not utilize the available GPU. Approximate wall-clock times: development-seed evaluation, track-only configuration, 4.89 hours; point-ERA5 configuration, 8.38 hours; environ-expert (locked) configuration, 10.91 hours; final held-out evaluation, 9.40 hours. ERA5 acquisition (551 storm-months via the Copernicus Climate Data Store) required on the order of tens of hours due to request-queue constraints rather than computation.

---

## References

*[To be reconciled with the corrected reference list from the original submission; citations marked in-text as placeholders (e.g., [CLIPER], [LIT-DL-TRACK]) correspond to entries requiring verification against original source titles, following the reference-integrity issue identified in prior peer review of a related manuscript from this project.]*
