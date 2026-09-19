"""Build NEW WAY/docs/paper_writeup_data/ from locked environ held-out results.

No architecture changes. CNN-GRU/BLSTM were not in the held-out run.
"""
from __future__ import annotations

import json
import platform
import subprocess
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
from eval_protocol import DEV_SEEDS, HELDOUT_SEEDS  # noqa: E402
from pipeline_common import (  # noqa: E402
    _prepare_eval_frame,
    apply_physics_features,
    apply_qc,
    build_mh_df,
    extract_model_names,
    per_storm_medians,
    sample_errors_for_model,
    storm_split,
    track_errors_km,
)

sece_v2_train.load_pipeline_data = pipeline_era5.load_pipeline_data
sece_v2_train.SUBSET_KEYS = pipeline_era5.SUBSET_KEYS_ERA5
sece_v2_train.SECE_SUBSETS_P3_3H = ["position", "motion", "environ", "full"]
sece_v2_train.SECE_SUBSETS_P3_OTHER = ["motion", "physics", "environ", "full"]

OUT = ROOT / "docs" / "paper_writeup_data"
PREDS_DIR = ROOT / "results" / "sece_era5_environ_heldout" / "task_a_test_predictions"
REFERENCE = "SECE v2 Phase3"
COMPETITORS = [
    "Stacking Ensemble",
    "Persistence Residual Cascade",
    "Random Forest",
    "LightGBM",
    "XGBoost",
    "CatBoost",
    "CB+MotionNN",
]
HORIZONS = ["3h", "12h", "24h", "48h"]
N_BOOT = 10_000
ALPHA = 0.05
N_COMPARISONS = len(COMPETITORS)
CASE_SEED = 0
ERA5_FEATS = list(pipeline_era5.ERA5_FEATS)
CASE_LAILA_SID = "2010137N10090"
CASE_TYPICAL_SID = "1996163N08088"
ALL_HELDOUT_MODELS = [REFERENCE] + COMPETITORS
# Held-out pooled median km rank-1 competitor per horizon (Table 2 / 01_*), for figure annotation.
BEST_COMPETITOR_BY_HORIZON = {
    "3h": "Random Forest",
    "12h": "Stacking Ensemble",
    "24h": "Stacking Ensemble",
    "48h": "XGBoost",
}


def utc() -> str:
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


def load_pooled_frames() -> dict[str, pd.DataFrame]:
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
            pred = pd.read_csv(PREDS_DIR / f"seed{seed}_predictions_{h}.csv")
            base = df_ts3 if h == "3h" else mh_test
            frame = _prepare_eval_frame(base, pred, merge_keys)
            frame["split_seed"] = seed
            frames[h].append(frame)
    return {h: pd.concat(parts, ignore_index=True) for h, parts in frames.items()}


def item1_comparison(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    print("Item 1: pooled CIs + Wilcoxon vs all held-out systems …", flush=True)
    rng = np.random.RandomState(42)
    rows: list[dict] = []
    storm_medians_cache: dict[tuple[str, str], dict[str, float]] = {}
    models = [REFERENCE] + COMPETITORS
    for horizon in HORIZONS:
        frame = frames[horizon]
        names = extract_model_names(frame.columns)
        for model in models:
            if model not in names:
                raise RuntimeError(f"Missing {model} at {horizon}")
            errors, sids = sample_errors_for_model(frame, horizon, model)
            storm_med = per_storm_medians(errors, sids)
            storm_medians_cache[(horizon, model)] = storm_med
            ci_lo, ci_hi = bootstrap_median_ci(errors, sids, rng)
            rows.append({
                "model": model,
                "horizon": horizon,
                "median_km": round(float(np.median(errors)), 3),
                "ci95_lo": round(ci_lo, 3),
                "ci95_hi": round(ci_hi, 3),
                "p_value_vs_sece": np.nan,
                "p_value_bonferroni_vs_sece": np.nan,
                "rank_biserial_vs_sece": np.nan,
                "significant_raw": "N/A" if model == REFERENCE else "",
                "significant_bonferroni": "N/A" if model == REFERENCE else "",
                "n_storms": len(storm_med),
                "n_samples": int(len(errors)),
                "n_storms_paired": np.nan,
            })
        sece_storms = storm_medians_cache[(horizon, REFERENCE)]
        for model in COMPETITORS:
            other = storm_medians_cache[(horizon, model)]
            common = sorted(set(sece_storms) & set(other))
            sece_vals = np.array([sece_storms[s] for s in common])
            other_vals = np.array([other[s] for s in common])
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
    df.to_csv(OUT / "01_heldout_comparison_ci_wilcoxon.csv", index=False)

    lines = [
        "# 1. Held-out comparison (locked environ SECE, all evaluated systems)",
        "",
        f"Generated `{utc()}`. Pooled test predictions across held-out seeds `{HELDOUT_SEEDS}`.",
        "Metric: sample-level median Haversine km; 95% CI from 10,000 storm-resamples.",
        "Wilcoxon signed-rank on paired per-storm medians vs SECE. Negative rank-biserial: SECE lower error.",
        f"Bonferroni: α={ALPHA} × {N_COMPARISONS} competitors → raw-p threshold {ALPHA / N_COMPARISONS:.4f}.",
        "",
        "**Not in this table:** CNN-GRU / BLSTM were not trained in the locked held-out run. Do not invent DL numbers.",
        "",
        "## Median km + 95% CI",
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
        "## Wilcoxon vs SECE",
        "",
        "| Horizon | Competitor | p (raw) | p (Bonferroni) | rank-biserial | sig raw | sig Bonf | n paired |",
        "|---------|------------|--------:|---------------:|--------------:|:-------:|:--------:|---------:|",
    ])
    for horizon in HORIZONS:
        for model in COMPETITORS:
            r = df[(df.horizon == horizon) & (df.model == model)].iloc[0]
            lines.append(
                f"| {horizon} | {model} | {r['p_value_vs_sece']} | {r['p_value_bonferroni_vs_sece']} | "
                f"{r['rank_biserial_vs_sece']} | {r['significant_raw']} | {r['significant_bonferroni']} | "
                f"{int(r['n_storms_paired'])} |"
            )
    (OUT / "01_heldout_comparison_ci_wilcoxon.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return df, storm_medians_cache


def item2_architecture() -> None:
    print("Item 2: architecture specification …", flush=True)
    p = sece_v2_train.build_params(quick=False, seed=42)
    rows = [
        {"component": "Locked architecture", "setting": "SECE v2 Phase 3 + point ERA5 + environ expert"},
        {"component": "Genesis window", "setting": "1940–2024 IBTrACS NI + ERA5 point features"},
        {"component": "Held-out seeds", "setting": str(HELDOUT_SEEDS)},
        {"component": "Dev seeds (architecture lock only)", "setting": str(DEV_SEEDS)},
        {"component": "Split", "setting": "storm-wise 70:15:15, seed-specific"},
        {"component": "3h SECE subsets", "setting": "position, motion, environ, full"},
        {"component": "12h/24h/48h SECE subsets", "setting": "motion, physics, environ, full"},
        {"component": "Subset tree algorithms", "setting": "CatBoost, XGBoost, LightGBM, Random Forest × each subset"},
        {"component": "3h subset experts", "setting": "4 subsets × 4 algorithms = 16 trees, then NNLS-fused to 1 subset signal"},
        {"component": "12h+ subset experts", "setting": "4 subsets × 4 algorithms = 16 trees, then NNLS-fused to 1 subset signal"},
        {"component": "Champion / standalone bases in fusion pool", "setting": "Stacking, PRC, RF, LightGBM, XGBoost, CatBoost, CB+MotionNN"},
        {"component": "NNLS full pool", "setting": "7 champions + 1 subset-NNLS signal (degree-space NNLS, separate lat/lon weights)"},
        {"component": "Router (Phase 3)", "setting": "hard val-best: pick candidate with lowest validation median km"},
        {"component": "3h extra fusion candidates", "setting": "SECE NNLS full, NNLS champions, RF+STK NNLS, STK+residual LGB"},
        {"component": "24h extra fusion candidates", "setting": "SECE NNLS full, NNLS champions, km-NNLS (champions, persistence-anchored)"},
        {"component": "48h extra fusion candidates", "setting": "as 24h plus LGB+PRC NNLS and LGB+PRC km-NNLS"},
        {"component": "GLOBAL_LR", "setting": str(sece_v2_train.GLOBAL_LR)},
        {"component": "Tree n_estimators (RF/XGB/LGB/CB)", "setting": "300 (quick=False)"},
        {"component": "Tree max_depth / CatBoost depth", "setting": "6"},
        {"component": "LightGBM num_leaves", "setting": "63"},
        {"component": "XGB subsample / colsample_bytree", "setting": "0.8 / 0.8"},
        {"component": "Stacking STK_N (RF+XGB+LGB Ridge stack)", "setting": str(p["STK_N"])},
        {"component": "MotionNN", "setting": "MLP 64-32-16, ReLU, lr=0.01, max_iter=100, early stopping"},
        {"component": "PRC", "setting": "persistence displacement + LightGBM residual in km, mapped back to degrees"},
        {"component": "km-NNLS", "setting": "NNLS on (pred−persist)×(111,111) km, then add persist"},
        {"component": "ERA5 environ keys", "setting": ", ".join(ERA5_FEATS)},
        {"component": "Environ subset also includes", "setting": "DIST2LAND, LANDFALL, NEWDELHI, STORM_SPEED, STORM_DIR + ERA5 keys"},
        {"component": "Not evaluated on held-out", "setting": "CNN-GRU, BLSTM, other DL"},
    ]
    pd.DataFrame(rows).to_csv(OUT / "02_architecture_hyperparameters.csv", index=False)
    md = ["# 2. Locked architecture specification", ""]
    md.append("| Component | Setting |")
    md.append("|-----------|---------|")
    for r in rows:
        md.append(f"| {r['component']} | {r['setting']} |")
    (OUT / "02_architecture_hyperparameters.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def item3_splits() -> pd.DataFrame:
    print("Item 3: per-seed train/val/test counts …", flush=True)
    df_raw = pd.read_csv(pipeline_era5.DATA_PATH)
    df = apply_physics_features(apply_qc(df_raw))
    mh = build_mh_df(df)
    rows = []
    for seed in HELDOUT_SEEDS:
        tr, va, te = storm_split(mh, seed=seed)
        mh_tr, mh_va, mh_te = mh[mh.SID.isin(tr)], mh[mh.SID.isin(va)], mh[mh.SID.isin(te)]
        dtr, dva, dte = df[df.SID.isin(tr)], df[df.SID.isin(va)], df[df.SID.isin(te)]
        rows.append({
            "seed": seed,
            "n_train_storms": len(tr),
            "n_val_storms": len(va),
            "n_test_storms": len(te),
            "n_train_mh_samples": len(mh_tr),
            "n_val_mh_samples": len(mh_va),
            "n_test_mh_samples": len(mh_te),
            "n_train_3h_samples": len(dtr),
            "n_val_3h_samples": len(dva),
            "n_test_3h_samples": len(dte),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "03_heldout_split_counts.csv", index=False)
    tot = {
        "qc_rows": len(df),
        "qc_storms": int(df.SID.nunique()),
        "mh_rows": len(mh),
        "mh_storms": int(mh.SID.nunique()),
    }
    lines = [
        "# 3. Held-out dataset split counts",
        "",
        f"Source: `{pipeline_era5.DATA_PATH.name}` after QC + physics features.",
        f"Full QC table: {tot['qc_storms']} storms / {tot['qc_rows']} 3h rows; "
        f"multi-horizon pool: {tot['mh_storms']} storms / {tot['mh_rows']} samples.",
        "Split: storm-wise 70:15:15 (`pipeline_common.storm_split`).",
        "",
        "| Seed | Train storms | Val storms | Test storms | Train MH | Val MH | Test MH | Train 3h | Val 3h | Test 3h |",
        "|-----:|-------------:|-----------:|------------:|---------:|-------:|--------:|---------:|-------:|--------:|",
    ]
    for _, r in out.iterrows():
        lines.append(
            f"| {int(r.seed)} | {int(r.n_train_storms)} | {int(r.n_val_storms)} | {int(r.n_test_storms)} | "
            f"{int(r.n_train_mh_samples)} | {int(r.n_val_mh_samples)} | {int(r.n_test_mh_samples)} | "
            f"{int(r.n_train_3h_samples)} | {int(r.n_val_3h_samples)} | {int(r.n_test_3h_samples)} |"
        )
    (OUT / "03_heldout_split_counts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _lgb_gain(model, names: list[str]) -> pd.Series:
    # MultiOutputRegressor wraps two LGBMs
    imps = []
    ests = getattr(model, "estimators_", [model])
    for est in ests:
        inner = est
        if hasattr(inner, "named_steps"):
            inner = list(inner.named_steps.values())[-1]
        gain = inner.booster_.feature_importance(importance_type="gain")
        imps.append(pd.Series(gain, index=names))
    return pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)


def item4_importance() -> pd.DataFrame:
    print("Item 4: environ-subset LightGBM gain at 24h/48h (seed 0) …", flush=True)
    import lightgbm as lgb
    from sklearn.multioutput import MultiOutputRegressor

    data = pipeline_era5.load_pipeline_data(seed=CASE_SEED)
    feat_cols = data["feature_cols"]
    env_idx = sece_v2_train.subset_columns(feat_cols, "environ")
    env_names = [feat_cols[i] for i in env_idx]
    rows = []
    for horizon in ("24h", "48h"):
        fr = sece_v2_train.get_horizon_frames(data, horizon, phase=3)
        Xtr = fr["X_tr"][:, env_idx]
        Xvl = fr["X_vl"][:, env_idx]
        ytr, yvl = fr["y_tr"], fr["y_vl"]
        params = dict(sece_v2_train.build_params(False, seed=CASE_SEED)["LGB_P"])
        model = MultiOutputRegressor(lgb.LGBMRegressor(**params))
        model.fit(Xtr, ytr)
        gain = _lgb_gain(model, env_names)
        total = float(gain.sum()) if float(gain.sum()) else 1.0
        for rank, (name, g) in enumerate(gain.items(), start=1):
            rows.append({
                "seed": CASE_SEED,
                "horizon": horizon,
                "feature": name,
                "is_era5": name in ERA5_FEATS or any(k in name for k in ("steer", "shear", "sst", "msl", "u850", "v850", "u500", "v500", "u200", "v200")),
                "gain": round(float(g), 4),
                "gain_share": round(float(g) / total, 6),
                "rank": rank,
            })
        # sanity val median
        pred = model.predict(Xvl)
        med = sece_v2_train.median_km(fr["vl_frame"]["LAT"].values, fr["vl_frame"]["LON"].values, yvl, pred)
        print(f"  {horizon} environ LGB val median {med:.2f} km", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "04_environ_expert_feature_importance.csv", index=False)
    era = out[out.is_era5]
    lines = [
        "# 4. Environ-subset expert feature importance (LightGBM gain)",
        "",
        f"Diagnostic retrain on held-out **seed {CASE_SEED}** only. Same LGB hyperparameters as locked SECE (n=300, depth=6, lr=0.01).",
        "Importance = mean LightGBM **gain** across the two MultiOutput heads (Δlat, Δlon).",
        "This is the environ-subset expert, not the full SECE stack. Not SHAP (tree gain is the fastest locked-spec diagnostic).",
        "",
    ]
    for horizon in ("24h", "48h"):
        sub = era[era.horizon == horizon].sort_values("rank")
        lines.append(f"## {horizon} — ERA5 features in the environ expert")
        lines.append("")
        lines.append("| Rank (within ERA5+environ) | Feature | Gain | Share of environ-gain |")
        lines.append("|---:|---------|-----:|----------------------:|")
        for _, r in sub.iterrows():
            lines.append(f"| {int(r['rank'])} | `{r['feature']}` | {r['gain']} | {r['gain_share']:.4f} |")
        lines.append("")
        top = out[out.horizon == horizon].nsmallest(8, "rank")
        lines.append("Top 8 environ-subset features (track + ERA5): " + ", ".join(f"`{x}`" for x in top.feature))
        lines.append("")
    (OUT / "04_environ_expert_feature_importance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _future_ll(frame: pd.DataFrame, horizon: str) -> tuple[np.ndarray, np.ndarray]:
    if horizon == "3h" and "dLAT" in frame.columns and "dLAT_3h" not in frame.columns:
        return frame["LAT"].values + frame["dLAT"].values, frame["LON"].values + frame["dLON"].values
    col = "dLAT_3h" if horizon == "3h" else f"dLAT_{horizon}"
    return frame["LAT"].values + frame[col].values, frame["LON"].values + frame[col.replace("dLAT", "dLON")].values


def model_slug(model: str) -> str:
    return (
        model.lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace(".", "")
        .replace("v2", "v2")
    )


def item5_case_studies() -> None:
    print("Item 5: case-study storms (all held-out systems) …", flush=True)
    helped_sid = CASE_LAILA_SID
    typical_sid = CASE_TYPICAL_SID

    names = pd.read_csv(pipeline_era5.DATA_PATH, usecols=["SID", "NAME", "SEASON"]).drop_duplicates("SID")
    name_map = dict(zip(names.SID.astype(str), names.NAME.astype(str) + " (" + names.SEASON.astype(str) + ")"))

    help_path = OUT / "05_seed0_lgb_era5_vs_trackonly_per_storm.csv"
    delta_24 = delta_48 = np.nan
    if help_path.exists():
        help_df = pd.read_csv(help_path)
        help_df["SID"] = help_df["SID"].astype(str)
        sub = help_df[help_df["SID"] == helped_sid]
        if not sub.empty:
            delta_24 = float(sub.loc[sub["horizon"] == "24h", "delta_km"].iloc[0]) if (sub["horizon"] == "24h").any() else np.nan
            delta_48 = float(sub.loc[sub["horizon"] == "48h", "delta_km"].iloc[0]) if (sub["horizon"] == "48h").any() else np.nan

    sece_24_typical = np.nan
    pred24 = pd.read_csv(PREDS_DIR / f"seed{CASE_SEED}_predictions_24h.csv")
    pred24["SID"] = pred24["SID"].astype(str)
    df_raw = pd.read_csv(pipeline_era5.DATA_PATH)
    df = apply_physics_features(apply_qc(df_raw))
    mh = build_mh_df(df)
    _, _, test_storms = storm_split(mh, seed=CASE_SEED)
    if typical_sid not in {str(s) for s in test_storms}:
        raise RuntimeError(f"{typical_sid} not in seed {CASE_SEED} test storms")
    if helped_sid not in {str(s) for s in test_storms}:
        raise RuntimeError(f"{helped_sid} not in seed {CASE_SEED} test storms")

    base24 = mh[mh["SID"].astype(str) == typical_sid].copy()
    pred24_sid = pred24[pred24["SID"] == typical_sid].copy()
    base24["ISO_TIME"] = pd.to_datetime(base24["ISO_TIME"])
    pred24_sid["ISO_TIME"] = pd.to_datetime(pred24_sid["ISO_TIME"])
    m24 = pd.merge(
        base24,
        pred24_sid,
        on=["SID", "ISO_TIME", "LAT", "LON"],
        how="inner",
    )
    if m24.empty:
        m24 = pd.merge(base24, pred24_sid, on=["SID", "ISO_TIME"], how="inner")
    if not m24.empty:
        y = m24[[f"dLAT_24h", f"dLON_24h"]].values
        lat = m24["LAT"].values
        lon = m24["LON"].values
        pred = m24[[f"pred_dLAT_{REFERENCE}", f"pred_dLON_{REFERENCE}"]].values
        sece_24_typical = float(np.median(track_errors_km(lat, lon, y, pred)))

    picks = pd.DataFrame([
        {
            "role": "era5_helped_24h_48h",
            "SID": helped_sid,
            "label": name_map.get(helped_sid, helped_sid),
            "delta_24h": delta_24,
            "delta_48h": delta_48,
            "sece_24h_median_km": np.nan,
            "note": "LAILA (2010); LGB track-only minus ERA5 error diagnostic on seed 0. Trajectories: all 8 held-out systems.",
        },
        {
            "role": "typical_sece_24h",
            "SID": typical_sid,
            "label": name_map.get(typical_sid, typical_sid),
            "delta_24h": np.nan,
            "delta_48h": np.nan,
            "sece_24h_median_km": sece_24_typical,
            "note": "Post-1990 BoB typical case: 1996163N08088 (1996), ~107 km SECE 24h vs seed-0 median ~106 km.",
        },
    ])
    picks.to_csv(OUT / "05_case_study_storm_picks.csv", index=False)

    traj_rows = []
    for sid, role in ((helped_sid, "era5_helped_24h_48h"), (typical_sid, "typical_sece_24h")):
        for horizon in HORIZONS:
            pred = pd.read_csv(PREDS_DIR / f"seed{CASE_SEED}_predictions_{horizon}.csv")
            pred["SID"] = pred["SID"].astype(str)
            base = df if horizon == "3h" else mh
            base = base[base["SID"].astype(str) == str(sid)].copy()
            pred = pred[pred["SID"].astype(str) == str(sid)].copy()
            base["ISO_TIME"] = pd.to_datetime(base["ISO_TIME"])
            pred["ISO_TIME"] = pd.to_datetime(pred["ISO_TIME"])
            merged = pd.merge(base, pred, on=["SID", "ISO_TIME", "LAT", "LON"], how="inner", suffixes=("", "_dup"))
            if merged.empty:
                merged = pd.merge(base, pred, on=["SID", "ISO_TIME"], how="inner", suffixes=("", "_dup"))
            if merged.empty:
                raise RuntimeError(f"No merge for case study {sid} {horizon}")

            gt_lat, gt_lon = _future_ll(merged, horizon)
            best_name = BEST_COMPETITOR_BY_HORIZON[horizon]
            chunk = {
                "role": role,
                "SID": merged["SID"].astype(str),
                "NAME": name_map.get(str(sid), str(sid)),
                "horizon": horizon,
                "ISO_TIME": merged["ISO_TIME"],
                "origin_lat": merged["LAT"].values,
                "origin_lon": merged["LON"].values,
                "true_lat": gt_lat,
                "true_lon": gt_lon,
                "best_competitor_at_horizon": best_name,
            }
            for model in ALL_HELDOUT_MODELS:
                slug = model_slug(model)
                chunk[f"{slug}_lat"] = (merged["LAT"] + merged[f"pred_dLAT_{model}"]).values
                chunk[f"{slug}_lon"] = (merged["LON"] + merged[f"pred_dLON_{model}"]).values
            traj_rows.append(pd.DataFrame(chunk))

    traj = pd.concat(traj_rows, ignore_index=True)
    traj.to_csv(OUT / "05_case_study_trajectories.csv", index=False)

    md = [
        "# 5. Case-study storms (held-out seed 0 trajectories)",
        "",
        f"- **ERA5-helped:** `{helped_sid}` {name_map.get(helped_sid, '')} (LAILA 2010).",
        f"- **Typical (post-1990):** `{typical_sid}` {name_map.get(typical_sid, '')} — replaces 1976 case.",
        "- **Trajectories:** ground truth + **all 8 held-out systems** at each origin time × horizon.",
        "- **`best_competitor_at_horizon`:** pooled held-out median winner for figure overlay (3h RF, 12h Stacking, 24h Stacking, 48h XGBoost).",
        "- Column prefix per model: `sece_v2_phase3`, `stacking_ensemble`, `persistence_residual_cascade`, `random_forest`, `lightgbm`, `xgboost`, `catboost`, `cbplusmotionnn`.",
        "- Files: `05_case_study_storm_picks.csv`, `05_case_study_trajectories.csv`.",
        "",
    ]
    (OUT / "05_case_study_storms.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"  wrote {len(traj)} trajectory rows", flush=True)


def item7_methods_storm_counts() -> None:
    print("Item 7: methods storm-count paragraph …", flush=True)
    text = """# 7. Methods — storm and sample counts (paste-ready)

We use IBTrACS North Indian Ocean cyclones with genesis in 1940–2024. After quality control on 3-hourly track rows (consistent timesteps and valid 3 h displacement targets), the modeling table contains **852 storms** and **27,728** forecast origins at 3 h. Multi-horizon training and evaluation (12 h, 24 h, and 48 h) require a complete forward track within each storm; **583 storms** yield **15,605** multi-horizon origins. Storm-wise splits (70% train / 15% validation / 15% test) are redrawn for each of nine held-out random seeds; each seed assigns about **408 / 87 / 88** storms to train, validation, and test. For significance testing we pool held-out **test-set** predictions across all nine seeds; **442** is the number of **distinct storm IDs** appearing in that pooled test sample (the same storm may fall in test under more than one seed). Reported median kilometre errors and Wilcoxon tests use all pooled test points, with paired per-storm medians defined on those 442 storms.
"""
    (OUT / "07_methods_storm_counts_paragraph.md").write_text(text, encoding="utf-8")


def _ps(cmd: str) -> str:
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=30)
        return (r.stdout or r.stderr or "").strip()
    except Exception as exc:
        return str(exc)


def item6_compute() -> None:
    print("Item 6: compute notes …", flush=True)
    cpu = _ps("(Get-CimInstance Win32_Processor).Name")
    gpu = _ps("(Get-CimInstance Win32_VideoController).Name")
    ram = _ps("[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)")
    rows = [
        {"item": "OS", "value": f"{platform.system()} {platform.release()} {platform.version()}"},
        {"item": "Python", "value": sys.version.split()[0]},
        {"item": "CPU", "value": cpu.replace("\n", " / ")},
        {"item": "GPU", "value": gpu.replace("\n", " / ") or "not reported (held-out trees/NNLS are CPU)"},
        {"item": "RAM_GB", "value": ram},
        {"item": "ERA5 CDS download", "value": "elapsed not stored; `era5_download_progress.json` status=complete 2026-09-10"},
        {"item": "Point ERA5 join / modeling table build", "value": "19.9 s (`build_modeling_dataset_progress.json`)"},
        {"item": "Track-only Phase 3 DEV (9 seeds)", "value": "4.89 h"},
        {"item": "Point-ERA5 Phase 3 DEV (9 seeds)", "value": "8.38 h"},
        {"item": "Environ-subset Phase 3 DEV (9 seeds, lock)", "value": "10.91 h"},
        {"item": "Spatial ERA5 5/3deg join", "value": "3.43 h (not used in locked held-out)"},
        {"item": "Held-out environ Phase 3 (9 seeds, locked)", "value": "9.4 h"},
        {"item": "Task A prediction export", "value": "cached under task_a_test_predictions/; first export required retraining per seed"},
        {"item": "This writeup pack", "value": f"generated {utc()}; stats-only + one-seed environ LGB + case-study LGB"},
    ]
    pd.DataFrame(rows).to_csv(OUT / "06_compute_details.csv", index=False)
    lines = ["# 6. Compute details", "", "| Item | Value |", "|------|-------|"]
    for r in rows:
        lines.append(f"| {r['item']} | {r['value']} |")
    lines.append("")
    lines.append("Held-out SECE is tree + NNLS CPU work. GPU is unused for the locked eval.")
    (OUT / "06_compute_details.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme() -> None:
    text = f"""# Paper writeup data (locked environ-subset held-out)

Generated `{utc()}`. No architecture changes. Source: `results/sece_era5_environ_heldout/`.

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
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--closing-only",
        action="store_true",
        help="regenerate case studies (05) + methods paragraph (07) + README only",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {OUT}", flush=True)

    if args.closing_only:
        item5_case_studies()
        item7_methods_storm_counts()
        write_readme()
        print("Done (closing-only).", flush=True)
        return

    skip_stats = (OUT / "01_heldout_comparison_ci_wilcoxon.csv").exists()
    if skip_stats:
        print("Item 1 already on disk — skip bootstrap.", flush=True)
        item5_case_studies()
        item7_methods_storm_counts()
    else:
        frames = load_pooled_frames()
        item1_comparison(frames)
        item2_architecture()
        item3_splits()
        item4_importance()
        item5_case_studies()
        item7_methods_storm_counts()
    item6_compute()
    write_readme()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
