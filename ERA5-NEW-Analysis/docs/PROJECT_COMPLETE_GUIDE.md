# SECE + ERA5 Bay of Bengal Track Prediction — Complete Project Guide

**Purpose:** One structured document from zero to finish. A reader with no prior context should understand *what was built*, *why each step exists*, *how evaluation works*, *what the final numbers mean*, and *where everything lives in the repository*.

**Last aligned with:** locked held-out confirm `results/sece_era5_environ_heldout/` and papers under `Paper Writing/` (Journal, Conference, WAF).

---

## Table of contents

1. [Problem and goal](#1-problem-and-goal)
2. [Project timeline (story in order)](#2-project-timeline-story-in-order)
3. [Data: from raw tracks to 63-D features](#3-data-from-raw-tracks-to-63-d-features)
4. [Models compared (the eight systems)](#4-models-compared-the-eight-systems)
5. [SECE v2 Phase 3 — architecture step by step](#5-sece-v2-phase-3--architecture-step-by-step)
6. [How training and splits work](#6-how-training-and-splits-work)
7. [Development phase vs held-out confirm](#7-development-phase-vs-held-out-confirm)
8. [Leakage: what went wrong and how it was fixed](#8-leakage-what-went-wrong-and-how-it-was-fixed)
9. [Controlled ablations (track-only → ERA5 → environ expert → spatial)](#9-controlled-ablations-track-only--era5--environ-expert--spatial)
10. [Final held-out results (numbers that go in the paper)](#10-final-held-out-results-numbers-that-go-in-the-paper)
11. [Statistics: CIs, Wilcoxon, Bonferroni](#11-statistics-cis-wilcoxon-bonferroni)
12. [Deep learning baselines (why they are excluded from the locked table)](#12-deep-learning-baselines-why-they-are-excluded-from-the-locked-table)
13. [Figures, tables, and case studies](#13-figures-tables-and-case-studies)
14. [Paper deliverables (three manuscripts)](#14-paper-deliverables-three-manuscripts)
15. [Repository map (folders and scripts)](#15-repository-map-folders-and-scripts)
16. [Limitations and honest scope](#16-limitations-and-honest-scope)
17. [Relationship to the original ICCACCESS paper](#17-relationship-to-the-original-iccaccess-paper)
18. [Quick reference cheat sheet](#18-quick-reference-cheat-sheet)

---

## 1. Problem and goal

**Scientific question:** How well can we predict **tropical cyclone track position** over the **North Indian Ocean / Bay of Bengal** at **3, 12, 24, and 48 hours** ahead using **machine learning on engineered tabular features**, optionally augmented with **ERA5 reanalysis** at the storm centre?

**Operational context:** Coastal Bangladesh and the Bay of Bengal rim are highly exposed. Full dynamical NWP is expensive; learned models on reanalysis-collocated features are a practical middle ground.

**What we deliver:**

- A **locked, leakage-aware benchmark** of **eight** tree/hybrid systems at four lead times.
- The proposed system: **SECE v2 Phase 3** with a dedicated **environ + ERA5 subset expert**.
- **Frozen exports** (predictions, bootstrap CIs, Wilcoxon tests) for external audit.
- **Three write-ups:** journal (`Journal_Paper.tex`), IEEE conference (`Conference_Paper.tex`), AMS WAF (`WAF/waf_manuscript.tex`).

**What we do *not* claim:**

- Operational IMD/RSMC skill parity (different verification protocol).
- Cross-basin transfer without retraining.
- Probabilistic track ensembles (deterministic point forecasts only).
- Held-out confirmation of CNN-GRU / BLSTM on the locked ERA5 protocol.

---

## 2. Project timeline (story in order)

| Phase | What happened | Outcome |
|-------|----------------|---------|
| **A. Original ICCACCESS paper** | SECE on IBTrACS-only features; claimed wins at 3/12/24 h | Submitted; dataset wording vs code mismatch; 3 h export bug later flagged by reviewer |
| **B. Audit (NEW WAY start)** | Count storms actually used; check exports | ~852 QC storms (not “312 BoB 1990–2022” as written); misaligned 3 h prediction files found |
| **C. SECE v2 Phases 1–3 (pre-fix)** | Iterative fusion/router improvements | Apparent “wins everywhere” on one split — **test-set leakage** (design tuned using test ranks) |
| **D. Leakage correction** | Separate **dev seeds** vs **held-out seeds**; freeze Phase 3 before final test | Honest mixed rank-1 counts; held-out run **once** |
| **E. ERA5 ingestion** | Download/join ERA5 point fields 1940–2024; SST missingness handling | 14 atmospheric features per origin |
| **F. Date window ablation** | Compare genesis windows 1940 / 1979 / 1990 + fixed modern holdout | **Locked: 1940–2024** (`docs/date_window_decision.md`) |
| **G. Development ablations** | Track-only → point ERA5 → environ expert → spatial patches | **Locked: point ERA5 + environ expert**; spatial **rejected** |
| **H. Held-out confirm** | `sece_era5_environ_heldout_eval.py` on 9 held-out seeds | Numbers in `docs/paper_writeup_data/01_*` |
| **I. Papers + figures** | LaTeX manuscripts, `generate_paper_figures.py`, WAF conversion | Submission-ready drafts; verify citations manually |

Plain-language narrative (same story, less jargon):  
`Paper Writing/Read MADE/Full_Project_Story_Plain_Language.md`

---

## 3. Data: from raw tracks to 63-D features

### 3.1 IBTrACS

- **IBTrACS** = international best-track archive (storm ID, time, lat/lon, intensity, basin metadata).
- **Basin:** North Indian Ocean storms relevant to **Bay of Bengal** exposure (genesis filter documented in methods).
- **Genesis window (locked):** **1940–2024** (maximize sample size; ERA5 homogeneous from 1940).

### 3.2 Quality control and forecast origins

- **3-hourly** track rows with valid **3 h displacement targets** (predict Δlat, Δlon).
- After QC: **852 storms**, **27,728** origins at **3 h** (`07_methods_storm_counts_paragraph.md`).
- **Multi-horizon** (12/24/48 h): need forward track; **583 storms**, **15,605** origins.

### 3.3 Feature groups (conceptual)

Roughly **49 kinematic / track / physics** engineered features (motion, lags, seasonality, land proximity, etc.) plus **14 ERA5 point** variables at storm centre:

| ERA5 / derived (environ) | Role |
|--------------------------|------|
| u/v 850, 500, 200 hPa | Vertical wind structure |
| MSL, SST | Pressure and ocean coupling |
| steer_u/v, shear_u/v, shear_mag | Steering and shear (derived) |
| sst_missing | Flag for nearest-ocean SST fill |

**Environ subset** also uses selected kinematic/context fields (e.g. DIST2LAND, STORM_SPEED, STORM_DIR) — see `02_architecture_hyperparameters.md`.

**Total input dimension:** **63-D** per forecast origin.

### 3.4 ERA5 pipeline (high level)

1. CDS download (see `docs/era5_cds_setup.md`).
2. Join to track times/locations → `datasets/tracks_era5.csv` / modeling CSV.
3. SST over land: **nearest-ocean fill** + **`sst_missing` flag** (`docs/sst_missingness.md`).

Main pipelines:

- `pipeline_era5.py` — locked **point ERA5** modeling path.
- `pipeline_trackonly_1940.py` — ablation without ERA5.
- `pipeline_era5_spatial.py` — **development-only** spatial patch experiment (not adopted).

---

## 4. Models compared (the eight systems)

All share the **same hyperparameter budget** unless noted (300 trees, depth 6, lr 0.01 for tree learners).

| # | Name | Idea |
|---|------|------|
| 1 | **SECE v2 Phase 3** | Subset experts + NNLS + champion fusion + val router (this paper) |
| 2 | **Stacking** | RF + XGB + LGB → Ridge stack |
| 3 | **PRC** | Persistence + LightGBM residual in **km**, map back to degrees |
| 4 | **Random Forest** | Single multi-output RF on full features |
| 5 | **LightGBM** | Single multi-output LGB |
| 6 | **XGBoost** | Single multi-output XGB |
| 7 | **CatBoost** | Single multi-output CatBoost |
| 8 | **CB+MotionNN** | CatBoost + MLP (64-32-16) on kinematic residuals |

**Target:** predict **Δlatitude** and **Δlongitude** at each horizon; error = **great-circle (Haversine) km**.

---

## 5. SECE v2 Phase 3 — architecture step by step

Think of SECE as a **committee with specialized sub-committees**, then a **final merge**, then a **validation-only chooser**.

### Stage 1 — Input

- One **63-D vector** per forecast origin (3 h row or multi-horizon row).

### Stage 2 — Four subset experts (horizon-specific feature views)

**Always four subsets**, not six, not mixed rows:

| Subset slot | At **3 h** | At **12 / 24 / 48 h** |
|-------------|------------|------------------------|
| 1 | **Position** | **Motion** |
| 2 | **Motion** | **Physics** |
| 3 | **Environ + ERA5** ★ | **Environ + ERA5** ★ |
| 4 | **Full 63-D** | **Full 63-D** |

★ **Locked design choice:** 14 ERA5 fields (+ environ kinematics) go to a **dedicated environ subset**, not scattered into motion/physics only.

### Stage 3 — Subset tree bank

- For **each** subset: train **CatBoost, XGBoost, LightGBM, Random Forest** (multi-output Δlat/Δlon).
- **4 subsets × 4 algorithms = 16 tree models per horizon.**
- **NNLS fusion within each subset** (non-negative least squares, **separate weights for lat and lon**) → compress to **one subset-consensus signal per horizon** (four NNLS fusions → four signals, then combined in fusion stage as one “subset expert channel” in the implementation — paper describes one fused subset signal entering the champion pool; see locked spec: **7 champions + 1 subset-NNLS signal**).

### Stage 4 — Seven champion baselines

- Train the seven standalone systems (Section 4) on the **same seed and split**.

### Stage 5 — Fusion candidates (validation only)

Build several **candidate final predictors** from champions ± subset signal, e.g.:

- **SECE NNLS:** 7 champions + subset-consensus (degree-space NNLS).
- **NNLS champions only.**
- **3 h extras:** RF+Stacking NNLS; Stacking + residual LGB.
- **24–48 h:** **km-NNLS** (persistence-anchored).
- **48 h extras:** LGB+PRC NNLS variants.

**Critical rule:** Candidates are compared on **validation median km only**. **Test is never used for selection.**

### Stage 6 — Phase 3 hard router

- For each **horizon** and each **random seed**, pick the candidate with **lowest validation median Haversine error**.
- **Freeze** that choice; write **test** predictions **once**.

### Stage 7 — Outputs

- **Δlat, Δlon** at 3, 12, 24, 48 h → convert to track error in km.

**Figure:** `docs/figures/sece_phase3_architecture.png` (+ editable `sece_phase3_architecture.drawio`).  
**Spec table:** `docs/paper_writeup_data/02_architecture_hyperparameters.csv`.

---

## 6. How training and splits work

- **Split unit:** entire **storms** (not random rows) → avoids leakage across time within one cyclone.
- **Ratio:** **70% train / 15% validation / 15% test** (`storm_split` in pipeline code).
- **Seed:** controls which storms fall in each fold; **different seeds = different splits**.

**Per seed, each system trains independently** on that seed’s train set; router uses val; metrics on test.

---

## 7. Development phase vs held-out confirm

| | **Development seeds** | **Held-out seeds** |
|---|------------------------|---------------------|
| **Purpose** | Architecture lock, ablations, spatial patch tests | **Final confirm only** |
| **IDs** | 3, 4, 5, 6, 8, 9, 10, 11, 14 | 0, 1, 2, 7, 13, 99, 123, 2024, 2026 |
| **How often used** | As many times as needed during design | **Once** after freeze |
| **Reported in ablation table** | Rank-1 / 9 and mean median km | “Held-out confirm (locked)” row |

**Never** tune Phase 3 router menus or subset design using held-out test ranks.

---

## 8. Leakage: what went wrong and how it was fixed

**Failure mode:** Earlier SECE iterations advanced Phase 1 → 2 → 3 while inspecting **test-set leaderboard** tables. That lets the **final exam** influence **study choices** → optimistic bias ([Cawley & Talbot, 2010](https://jmlr.org/papers/volume11/cawley10a.html) model-selection leakage).

**Symptoms:** “Wins at every horizon” on one split; **did not replicate** on fresh seeds after freeze.

**Fix:**

1. Lock **Phase 3 + environ + point ERA5** on dev seeds only.
2. Pre-register **held-out seed list**.
3. Run held-out pipeline **once**; export all predictions and stats.

**Paper stance:** Report leakage transparently as a **methods contribution**, not hidden.

---

## 9. Controlled ablations (track-only → ERA5 → environ expert → spatial)

Development-only comparisons (Table ablation in papers):

| Configuration | Main finding |
|---------------|--------------|
| Track-only (no ERA5) | Strong short-lead; weak 48 h |
| + point ERA5 | **48 h** mean improves (~250 → ~247 km class) |
| + **environ expert** | **24 h** improves where point-ERA5 was weakest; **locked** |
| + spatial 5° / 3° patches | Rank-1 sometimes up at 3 h; **24–48 h ≈ within noise** → **rejected** |

Spatial scripts: `sece_era5_spatial_dev_eval.py`, `pipeline_era5_spatial.py` (dev only).

---

## 10. Final held-out results (numbers that go in the paper)

**Source:** `docs/paper_writeup_data/01_heldout_comparison_ci_wilcoxon.csv`  
**Pooled:** 9 held-out seeds; **442 distinct test storm IDs**; all test points pooled for medians.

### SECE v2 Phase 3 — pooled median great-circle error (km)

| Horizon | SECE median km | 95% CI (approx.) |
|--------:|---------------:|------------------|
| 3 h | **4.895** | 4.714 – 5.079 |
| 12 h | **35.538** | 34.208 – 36.925 |
| 24 h | **101.186** | 97.301 – 104.808 |
| 48 h | **246.331** | 235.541 – 257.614 |

### Interpretation (ranking + significance)

- **3 h & 12 h:** SECE **significantly better than all seven baselines** (Wilcoxon; Bonferroni across 7 comparisons).
- **24 h & 48 h:** SECE **overlaps top tree stacks** (Stacking, XGB, LGB, PRC) — **not** significant separation vs leaders; still **never significantly worse than any baseline** at any horizon in the Bonferroni framework used.
- **CatBoost / CB+MotionNN:** Most often **weakest at short lead**.

**Per-seed rank-1 (SECE):** dev ablation vs held-out row in paper table (~6/9 at 3–12 h on held-out, weaker at 24–48 h) — always report **pooled medians + tests** as primary.

**IMD context (not direct comparison):** Mohapatra et al. 2013 official NIO verification ~**124 km @ 24 h**, ~**202 km @ 48 h** vs our **ML hindcast medians** — different protocol; papers state this explicitly.

---

## 11. Statistics: CIs, Wilcoxon, Bonferroni

- **Point metric:** sample-level median Haversine km over all pooled test predictions.
- **95% CI:** **10,000 storm resamples** (bootstrap-style pooling documented in export scripts).
- **Wilcoxon signed-rank:** paired **per-storm medians** SECE vs each baseline (442 storms).
- **Bonferroni:** 7 competitors × α=0.05 → raw **p < 0.0071** for family-wise claim.

Scripts: `heldout_task_a_significance.py`, exports under `results/sece_era5_environ_heldout/`.

---

## 12. Deep learning baselines (why they are excluded from the locked table)

- **CNN-GRU / BLSTM** were run in **exploratory development** (track-only, then informal ERA5 checks) under the **same fixed budget** (300 epochs/estimators, lr 0.01).
- They **trailed** tree/hybrid leaders at 3–24 h (often persistence-scale errors).
- Consistent with **Grinsztajn et al. 2022** (trees vs deep on tabular data at this sample size).
- **Not re-run** on nine held-out seeds for 48 h head + full ERA5 protocol → **do not invent DL rows in Table 1**.

Conference/journal **wording:** scoped out after exploratory dev; locked confirm = **eight tree systems**.

---

## 13. Figures, tables, and case studies

| Asset | Content |
|-------|---------|
| `sece_phase3_architecture.png` | Phase 3 pipeline diagram |
| `benchmark_3h_heldout.png` | Mean vs normalized accuracy (3 h); table has medians + CIs |
| `tsne_era5_63d.png` | 63-D feature space visualization (journal/WAF) |
| `case_study_laila_trajectories.png` | Cyclone Laila (2010), multi-lead endpoints |
| Feature importance | LightGBM **gain** on environ subset, **seed 0 only** — illustrative |
| Ablation table | Dev configs + held-out row |
| Held-out table | All eight systems × four horizons + CIs |

Generate/regenerate: `python NEW WAY/scripts/generate_paper_figures.py`  
WAF copies: `generate_waf_figures.py`

Case study picks: `docs/paper_writeup_data/05_case_study_storms.md`

---

## 14. Paper deliverables (three manuscripts)

| File | Venue style | Length target | Notes |
|------|-------------|---------------|-------|
| `Paper Writing/Journal_Paper.tex` | `article`, two-column | ~13 pp | Full methods, tables, discussion |
| `Paper Writing/Conference_Paper.tex` | IEEEtran conference | **≤ 6 pp** incl. refs | Compressed; same core claims |
| `Paper Writing/WAF/waf_manuscript.tex` | AMS `ametsocV6.1` | WAF word limit | Upload **entire `WAF/` folder** |

**Shared claims:** leakage correction, 1940–2024 genesis, environ expert, held-out medians, Wilcoxon, negative spatial result, DL scope-out, limitations (basin, era, collinearity on gain, deterministic outputs).

**Do not cite** unpublished self-horizon paper as DL justification.

**Numbers must match:** `docs/paper_writeup_data/` (see README there).

Overleaf instructions: `Paper Writing/README.md`

---

## 15. Repository map (folders and scripts)

```
NEW WAY/
├── pipeline_era5.py              # Main locked ERA5 point pipeline
├── pipeline_trackonly_1940.py    # Track-only ablations
├── pipeline_era5_spatial.py      # Spatial patch (dev, not locked)
├── datasets/                     # Modeling CSVs (tracks + ERA5)
├── results/
│   └── sece_era5_environ_heldout/   # Final confirm exports, predictions, flags
├── scripts/
│   ├── sece_era5_environ_heldout_eval.py   # Run held-out confirm
│   ├── sece_era5_environ_dev_eval.py       # Dev ablations
│   ├── build_paper_writeup_data.py         # Freeze tables for papers
│   ├── generate_paper_figures.py           # Figures for journal/conference
│   └── heldout_task_a_significance.py      # Significance exports
├── docs/
│   ├── paper_writeup_data/       # CSV/MD tables tied to paper numbers
│   ├── figures/                  # PNG/PDF for LaTeX
│   ├── PROJECT_COMPLETE_GUIDE.md # This file
│   └── *.md                      # ERA5 setup, date window, reports
└── Paper Writing/
    ├── Journal_Paper.tex
    ├── Conference_Paper.tex
    ├── WAF/waf_manuscript.tex
    └── Read MADE/Full_Project_Story_Plain_Language.md
```

**Resume / status files:** `HELDOUT_STATUS.txt`, `ENVIRON_DEV_STATUS.txt`, `run_*_resilient.ps1` for long runs.

---

## 16. Limitations and honest scope

- **Basin:** NIO / Bay of Bengal–focused; **no cross-basin** held-out transfer.
- **ERA5:** Point (and rejected patch) collocations — not full gridded deep encoders.
- **Era mixing:** 1940–2024; pre-1979 wind sparsity; fixed modern holdout supports window choice but cannot erase all era effects.
- **Deterministic:** single track point; no calibrated ensemble spread.
- **Attribution:** environ gain shares = **one seed**; shear/steering **collinear** — no SHAP-level mechanistic claims.
- **Operational:** comparison to **IMD** track verification is contextual only.

Future work listed in papers: probabilistic scores, gridded/attention ERA5, physics-constrained motion, multi-seed SHAP.

---

## 17. Relationship to the original ICCACCESS paper

| Topic | Original ICCACCESS track-only paper | This NEW WAY ERA5 project |
|-------|-------------------------------------|---------------------------|
| Data | IBTrACS features primarily | IBTrACS + **ERA5 point** |
| Scope | Revision path for conference submission | **Separate** journal/conference/WAF submission |
| Evaluation | Single-split risk | **Dev vs held-out seeds**, significance |
| Known bug | 3 h export mismatch (reviewer #2) | Found independently in audit; fix in **original** pipeline for revision |
| Main claim tone | Sweep all horizons | **Horizon-dependent:** strong 3–12 h; parity band 24–48 h |

Both can coexist: **revise ICCACCESS** on its own data; **submit ERA5 work** as new papers.

---

## 18. Quick reference cheat sheet

**Locked architecture:** SECE v2 Phase 3 + point ERA5 + **environ expert**

**Inputs:** 63-D (49 track/physics + 14 ERA5 point)

**Subsets:** 4 per horizon (Position/Motion vs Motion/Physics swap at 3 h vs 12–48 h)

**Trees:** 16 per horizon (4×4); NNLS per subset; fuse with 7 champions; val router

**Dev seeds:** 3,4,5,6,8,9,10,11,14 | **Held-out:** 0,1,2,7,13,99,123,2024,2026

**Storms (QC):** 852 (3 h) | 583 (multi-horizon) | **442** distinct pooled test storms

**SECE medians (km):** 4.895 | 35.538 | 101.186 | 246.331 @ 3/12/24/48 h

**Significance:** Wilcoxon + Bonferroni (7 comps); SECE wins all at 3–12 h; ties leaders at 24–48 h

**Not in held-out table:** CNN-GRU, BLSTM

**Primary evidence folder:** `docs/paper_writeup_data/`

---

*Maintainers: when held-out exports or architecture change, re-run `build_paper_writeup_data.py` and update this guide’s Section 10 and cheat sheet to match CSV timestamps.*
