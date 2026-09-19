# Paper writeup data (locked environ-subset held-out)

Generated `2026-09-19T08:59:23Z`. No architecture changes. Source: `results/sece_era5_environ_heldout/`.

| File | Contents |
|------|----------|
| `01_heldout_comparison_ci_wilcoxon.md` / `.csv` | Median km + 95% CI at 3/12/24/48h; Wilcoxon+Bonferroni vs SECE for all 7 evaluated competitors |
| `02_architecture_hyperparameters.md` / `.csv` | Locked Phase 3 environ SECE spec |
| `03_heldout_split_counts.md` / `.csv` | Train/val/test storms and samples per held-out seed |
| `04_environ_expert_feature_importance.md` / `.csv` | LightGBM gain on environ subset, seed 0, 24h and 48h |
| `05_case_study_storms.md`, `05_case_study_storm_picks.csv`, `05_case_study_trajectories.csv` | LAILA + 1996 typical; all 8 systems × 4 horizons |
| `06_compute_details.md` / `.csv` | Hardware + wall-clock from run logs |
| `07_methods_storm_counts_paragraph.md` | Paste-ready 852 / 583 / 442 definitions |

CNN-GRU / BLSTM: **not evaluated** on held-out. Do not add them to the paper table.
