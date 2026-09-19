"""
Task A on held-out environ Phase 3: Wilcoxon signed-rank + bootstrap 95% CI.

Pools test-set predictions across all HELDOUT seeds; per-storm median Haversine km.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT.parent / "Final_Cyclone_Pred_Results_P-3"
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(ROOT))

import pipeline_era5  # noqa: E402
import sece_v2_train  # noqa: E402
from eval_protocol import HELDOUT_SEEDS  # noqa: E402
from pipeline_common import (  # noqa: E402
    _prepare_eval_frame,
    apply_physics_features,
    apply_qc,
    build_mh_df,
    extract_model_names,
    per_storm_medians,
    sample_errors_for_model,
    storm_split,
)

sece_v2_train.load_pipeline_data = pipeline_era5.load_pipeline_data
sece_v2_train.SUBSET_KEYS = pipeline_era5.SUBSET_KEYS_ERA5
sece_v2_train.SECE_SUBSETS_P3_3H = ["position", "motion", "environ", "full"]
sece_v2_train.SECE_SUBSETS_P3_OTHER = ["motion", "physics", "environ", "full"]

HELDOUT_DIR = ROOT / "results" / "sece_era5_environ_heldout"
PREDS_DIR = HELDOUT_DIR / "task_a_test_predictions"
OUT_CSV = HELDOUT_DIR / "task_a_significance.csv"
OUT_MD = ROOT / "docs" / "heldout_task_a_significance.md"
PROG = HELDOUT_DIR / "task_a_export_progress.json"

REFERENCE_MODEL = "SECE v2 Phase3"
COMPETITORS = [
    "Stacking Ensemble",
    "Random Forest",
    "LightGBM",
    "XGBoost",
]
HORIZONS = ["3h", "12h", "24h", "48h"]
N_BOOT = 10_000
ALPHA = 0.05
N_COMPARISONS = len(COMPETITORS)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    d = x - y
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return np.nan
    ranks = rankdata(np.abs(d))
    w_plus = ranks[d > 0].sum()
    return float((2 * w_plus - n * (n + 1) / 2) / (n * (n + 1) / 2))


def bootstrap_median_ci(errors: np.ndarray, sids: np.ndarray, rng: np.random.RandomState) -> tuple[float, float]:
    unique = np.unique(sids)
    if len(unique) == 0 or len(errors) == 0:
        return np.nan, np.nan
    by_storm = [errors[sids == s] for s in unique]
    n = len(by_storm)
    boot_medians = np.empty(N_BOOT, dtype=float)
    for b in range(N_BOOT):
        idx = rng.randint(0, n, size=n)
        pooled = np.concatenate([by_storm[i] for i in idx])
        boot_medians[b] = np.median(pooled)
    lo, hi = np.percentile(boot_medians, [2.5, 97.5])
    return float(lo), float(hi)


def seed_preds_complete(seed: int) -> bool:
    return all((PREDS_DIR / f"seed{seed}_predictions_{h}.csv").exists() for h in HORIZONS)


def export_predictions_for_seed(seed: int) -> None:
    if seed_preds_complete(seed):
        print(f"Seed {seed}: predictions already exported", flush=True)
        return
    print(f"Seed {seed}: training + exporting test predictions …", flush=True)
    sece_v2_train.run_sece_v2(
        phase=3,
        quick=False,
        seed=seed,
        quiet=True,
        export_all_test_preds_dir=PREDS_DIR,
    )


def load_pooled_eval_frames() -> dict[str, pd.DataFrame]:
    """One QC/physics pass; per-seed test masks only (avoids 9x full pipeline load)."""
    merge_keys = ["SID", "ISO_TIME", "LAT", "LON"]
    frames: dict[str, list[pd.DataFrame]] = {h: [] for h in HORIZONS}

    df_raw = pd.read_csv(pipeline_era5.DATA_PATH)
    df = apply_physics_features(apply_qc(df_raw))
    mh_df = build_mh_df(df)

    for seed in HELDOUT_SEEDS:
        _, _, test_storms = storm_split(mh_df, seed=seed)
        df_ts3 = df[df["SID"].isin(test_storms)]
        mh_test = mh_df[mh_df["SID"].isin(test_storms)]
        for h in HORIZONS:
            pred_path = PREDS_DIR / f"seed{seed}_predictions_{h}.csv"
            pred = pd.read_csv(pred_path)
            base = df_ts3 if h == "3h" else mh_test
            frame = _prepare_eval_frame(base, pred, merge_keys)
            frame["split_seed"] = seed
            frames[h].append(frame)

    return {h: pd.concat(parts, ignore_index=True) for h, parts in frames.items()}


def run_task_a(export_only: bool = False, stats_only: bool = False) -> pd.DataFrame:
    HELDOUT_DIR.mkdir(parents=True, exist_ok=True)
    if not stats_only:
        done_seeds = []
        for seed in HELDOUT_SEEDS:
            export_predictions_for_seed(seed)
            done_seeds.append(seed)
            PROG.write_text(
                json.dumps({"updated_at": _utc(), "seeds_exported": done_seeds}, indent=2),
                encoding="utf-8",
            )
        if export_only:
            print("Export complete.", flush=True)
            return pd.DataFrame()
    else:
        missing = [s for s in HELDOUT_SEEDS if not seed_preds_complete(s)]
        if missing:
            raise SystemExit(f"--stats-only but missing prediction exports for seeds: {missing}")

    print("Loading pooled held-out test frames …", flush=True)
    frames = load_pooled_eval_frames()
    print("Running bootstrap CIs and Wilcoxon tests …", flush=True)
    rng = np.random.RandomState(42)
    rows: list[dict] = []
    storm_medians_cache: dict[tuple[str, str], dict[str, float]] = {}

    models = [REFERENCE_MODEL] + COMPETITORS
    for horizon in HORIZONS:
        frame = frames[horizon]
        for model in models:
            if model not in extract_model_names(frame.columns):
                raise RuntimeError(f"Missing predictions for {model} at {horizon}")
            errors, sids = sample_errors_for_model(frame, horizon, model)
            storm_med = per_storm_medians(errors, sids)
            storm_medians_cache[(horizon, model)] = storm_med
            ci_lo, ci_hi = bootstrap_median_ci(errors, sids, rng)
            rows.append(
                {
                    "model": model,
                    "horizon": horizon,
                    "median_km": round(float(np.median(errors)), 3) if len(errors) else np.nan,
                    "ci95_lo": round(ci_lo, 3),
                    "ci95_hi": round(ci_hi, 3),
                    "p_value_vs_sece": np.nan,
                    "p_value_bonferroni_vs_sece": np.nan,
                    "rank_biserial_vs_sece": np.nan,
                    "significant_raw": "N/A" if model == REFERENCE_MODEL else "",
                    "significant_bonferroni": "N/A" if model == REFERENCE_MODEL else "",
                    "n_storms": len(storm_med),
                    "n_samples": int(len(errors)),
                    "n_storms_paired": np.nan,
                }
            )

        sece_storms = storm_medians_cache.get((horizon, REFERENCE_MODEL), {})
        for model in COMPETITORS:
            other_storms = storm_medians_cache.get((horizon, model), {})
            common = sorted(set(sece_storms) & set(other_storms))
            if len(common) < 5:
                continue
            sece_vals = np.array([sece_storms[s] for s in common])
            other_vals = np.array([other_storms[s] for s in common])
            try:
                stat, p_val = wilcoxon(sece_vals, other_vals, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                p_val, stat = np.nan, np.nan
            r_rb = rank_biserial(sece_vals, other_vals)
            p_bonf = min(float(p_val * N_COMPARISONS), 1.0) if pd.notna(p_val) else np.nan
            sig_raw = "Y" if pd.notna(p_val) and p_val < ALPHA else "N"
            sig_bonf = "Y" if pd.notna(p_bonf) and p_bonf < ALPHA else "N"
            for row in rows:
                if row["horizon"] == horizon and row["model"] == model:
                    row["p_value_vs_sece"] = round(float(p_val), 6) if pd.notna(p_val) else np.nan
                    row["p_value_bonferroni_vs_sece"] = round(p_bonf, 6) if pd.notna(p_bonf) else np.nan
                    row["rank_biserial_vs_sece"] = round(r_rb, 4) if pd.notna(r_rb) else np.nan
                    row["significant_raw"] = sig_raw
                    row["significant_bonferroni"] = sig_bonf
                    row["n_storms_paired"] = len(common)
                    row["wilcoxon_stat"] = float(stat) if pd.notna(stat) else np.nan

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    write_report(df)
    print(df.to_string(index=False), flush=True)
    print(f"\nWrote {OUT_CSV}", flush=True)
    print(f"Wrote {OUT_MD}", flush=True)
    return df


def write_report(df: pd.DataFrame) -> None:
    lines = [
        "# Held-out Task A — significance vs SECE (environ Phase 3, locked)",
        "",
        "Pooled **held-out test** predictions across seeds "
        f"`{HELDOUT_SEEDS}`. Per-storm median Haversine error (km); "
        "bootstrap 95% CI resamples storms; Wilcoxon signed-rank vs SECE on paired storm medians.",
        "",
        f"Bonferroni: α={ALPHA} × {N_COMPARISONS} competitors → threshold {ALPHA / N_COMPARISONS:.4f} on raw p.",
        "",
        f"Predictions cache: `{PREDS_DIR.relative_to(ROOT)}`",
        f"CSV: `{OUT_CSV.relative_to(ROOT)}`",
        "",
        "## All models — median km and bootstrap 95% CI",
        "",
        "| Horizon | Model | Median km | CI 95% lo | CI 95% hi | n storms | n samples |",
        "|---------|-------|----------:|----------:|----------:|---------:|----------:|",
    ]
    for horizon in HORIZONS:
        sub = df[df.horizon == horizon].sort_values("median_km")
        for _, r in sub.iterrows():
            lines.append(
                f"| {horizon} | {r['model']} | {r['median_km']} | {r['ci95_lo']} | {r['ci95_hi']} | "
                f"{int(r['n_storms'])} | {int(r['n_samples'])} |"
            )

    lines.extend([
        "",
        "## Wilcoxon vs SECE (paired per-storm medians)",
        "",
        "| Horizon | Competitor | p (raw) | p (Bonferroni) | rank-biserial | sig raw | sig Bonf | n paired storms |",
        "|---------|------------|--------:|---------------:|--------------:|:-------:|:--------:|----------------:|",
    ])
    for horizon in HORIZONS:
        for model in COMPETITORS:
            r = df[(df.horizon == horizon) & (df.model == model)]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(
                f"| {horizon} | {model} | {r['p_value_vs_sece']} | {r['p_value_bonferroni_vs_sece']} | "
                f"{r['rank_biserial_vs_sece']} | {r['significant_raw']} | {r['significant_bonferroni']} | "
                f"{int(r['n_storms_paired']) if pd.notna(r['n_storms_paired']) else ''} |"
            )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--export-only", action="store_true", help="only export per-seed test predictions")
    parser.add_argument("--stats-only", action="store_true", help="skip export; use cached task_a_test_predictions/")
    args = parser.parse_args()
    run_task_a(export_only=args.export_only, stats_only=args.stats_only)
