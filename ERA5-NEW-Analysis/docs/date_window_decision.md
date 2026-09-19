# Date window decision

**Status: LOCKED — `full_era5` (1940–2024)**

Genesis-year filter for all training data. ERA5 reanalysis is homogeneous 1940–2024; this window maximizes trajectory + environmental examples for long-lead track prediction.

---

## Prerequisites completed

1. `results/ibtracs_quality_by_window.csv` — wind/intensity sparsity by era
2. `datasets/tracks_era5.csv` — ERA5 joined to tracks (+ `sst_missing`, nearest-ocean SST fill; see `docs/sst_missingness.md`)
3. `results/date_window_ablation.csv` — simple LGB ablation (9 DEV seeds, ERA5 features)
4. `results/date_window_fixed_test_control.csv` — fixed modern holdout control
5. `results/position_vs_wind_quality_by_era.csv` — position vs wind quality check (see `docs/position_vs_wind_quality.md`)

---

## Ablation (mean test median km, 9 DEV seeds, each window’s own test split)

| Window | 3h | 12h | 24h | 48h |
|--------|---:|----:|----:|----:|
| **1940–2024 (full_era5)** | **5.24** | **36.19** | **99.77** | **244.92** |
| 1979–2024 (satellite) | 6.13 | 44.04 | 112.38 | 271.40 |
| 1990–2024 (modern) | 6.14 | 44.11 | 108.12 | 261.58 |

---

## Fixed-test control (train per window → same 1990–2024 genesis holdout)

All models exclude the same ~29 modern test storms from training.

| Train window | 3h | 12h | 24h | 48h |
|--------------|---:|----:|----:|----:|
| **1940–2024** | **5.96** | **39.42** | **100.18** | **243.86** |
| 1979–2024 | 6.07 | 42.24 | 103.86 | 249.82 |
| 1990–2024 | 6.14 | 44.11 | 108.12 | 261.58 |

`full_era5` wins on the fixed modern test at **all horizons** — the wide-window advantage is not only a test-set composition effect.

---

## Horizon-dependent nuance (original ablation gap → fixed-test gap)

| Horizon | Original gap (1940 vs 1990) | Fixed-test gap (1940 vs 1990) | Interpretation |
|---------|----------------------------:|------------------------------:|----------------|
| **3h** | 0.90 km | **0.18 km** | Mostly test-composition artifact; marginal real benefit from extra history |
| **12h** | 7.9 km | **4.7 km** | Partially real; moderate training-data benefit |
| **24h** | 8.4 km | **8.0 km** | Real and persists on same modern holdout |
| **48h** | 16.7 km | **17.7 km** | Real and persists on same modern holdout |

**Takeaway:** Extra pre-1979 data helps most at **24h/48h** track prediction. Short-lead (3h) gains are small once test storms are held fixed.

---

## Position vs wind quality (why 1940–2024 is defensible)

See `docs/position_vs_wind_quality.md` for full table.

| Metric | Pre-satellite (1940–1978) | Modern (1990–2024) |
|--------|-------------------------:|-------------------:|
| IBTrACS WMO wind missing (point) | 100.0% | 48.8% |
| IBTrACS NEWDELHI wind missing (point) | 100.0% | 33.0% |
| IFLAG interpolated position (point) | 0.0% | 0.0% |
| Research-ready irregular dt (pre-QC) | 0.00% | 0.04% |
| Research-ready lat/lon mismatch (pre-QC) | 0.00% | 0.03% |
| QC-passing std(dLAT) deg | 0.172 | 0.218 |
| QC-passing std(dLON) deg | 0.310 | 0.332 |

**Interpretation:**

- The pre-1979 gap is concentrated in **wind/intensity observations**, not **position/timing quality**.
- Position proxies (IFLAG interpolation, irregular dt, lat/lon mismatch) are comparable or better pre-1979 after QC.
- Extra historical storms mainly add **trajectory + ERA5 environment** examples — exactly what long-lead track models need — even when IBTrACS wind reports are absent.
- ERA5 along-track wind spread differs by era because of **meteorology**, not reanalysis quality (ERA5 is homogeneous 1940–2024).

**Paper note:** Acknowledge sparse pre-1979 wind obs; argue that position + ERA5 features remain usable for track prediction.

---

## Known tradeoffs

- **SST missingness:** ~38% of rows flagged `sst_missing` (land/coast proximity); filled via nearest valid ocean cell. Pre-1979 row rate higher (~44%) but wind/msl/shear are 0% missing. See `docs/sst_missingness.md`.
- **Wind features:** If intensity-aware models are added later, pre-1979 wind sparsity must be handled explicitly (missing flags, era features, or modern-only intensity heads).

---

## Next step: SECE + ERA5 modeling

With this window locked, proceed to:

1. Build full modeling dataset — merge `tracks_era5.csv` with research-ready features + physics features from `Final_Cyclone_Pred_Results_P-3/`
2. Re-run **SECE with ERA5** on DEV seeds; compare to Phase 3 track-only baseline
3. Held-out evaluation only after architecture locked (`eval_protocol.py`: DEV seeds `[3,4,5,6,8,9,10,11,14]`, held-out `[0,1,2,7,13,99,123,2024,2026]`)
