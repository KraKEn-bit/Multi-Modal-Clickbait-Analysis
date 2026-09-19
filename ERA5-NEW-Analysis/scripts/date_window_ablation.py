"""
Date-window ablation: simple LightGBM + ERA5 features, DEV seeds only.

Compares genesis-year windows 1940-2024, 1979-2024, 1990-2024 on val/test median track error.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT.parent / "Final_Cyclone_Pred_Results_P-3"
sys.path.insert(0, str(OLD))

from eval_protocol import DEV_SEEDS  # noqa: E402
from pipeline_common import (  # noqa: E402
    H_LABELS,
    apply_physics_features,
    apply_qc,
    build_mh_df,
    get_feature_cols,
    haversine_km,
    storm_split,
)

READY = OLD / "datasets" / "bangladesh_nextstep_dataset_research_ready.csv"
ERA5 = ROOT / "datasets" / "tracks_era5.csv"
OUT = ROOT / "results" / "date_window_ablation.csv"
PROG = ROOT / "results" / "date_window_ablation_progress.json"

ERA5_FEATS = [
    "u850", "v850", "u500", "v500", "u200", "v200",
    "msl", "sst", "steer_u", "steer_v", "shear_u", "shear_v", "shear_mag",
    "sst_missing",
]

WINDOWS = [
    ("full_era5", 1940, 2024),
    ("satellite", 1979, 2024),
    ("modern", 1990, 2024),
]

LGB_P = dict(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    verbose=-1,
    n_jobs=-1,
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def genesis_year_filter(df: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    df = df.copy()
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"])
    gen = df.groupby("SID")["ISO_TIME"].min().dt.year
    keep = set(gen[(gen >= y0) & (gen <= y1)].index.astype(str))
    return df[df["SID"].astype(str).isin(keep)].reset_index(drop=True)


def load_window_dataset(y0: int, y1: int) -> pd.DataFrame:
    ready = pd.read_csv(READY)
    era5 = pd.read_csv(ERA5)
    ready["ISO_TIME"] = pd.to_datetime(ready["ISO_TIME"])
    era5["ISO_TIME"] = pd.to_datetime(era5["ISO_TIME"])
    ready = genesis_year_filter(ready, y0, y1)
    sids = set(ready["SID"].astype(str))
    era5 = era5[era5["SID"].astype(str).isin(sids)]
    df = ready.merge(era5, on=["SID", "ISO_TIME"], how="inner")
    df = apply_qc(df)
    return apply_physics_features(df)


def eval_horizon(mh_df: pd.DataFrame, split: str, storms: set, feat_cols: list[str],
                 imputer: SimpleImputer, model, horizon: str) -> float:
    sub = mh_df[mh_df["SID"].isin(storms)].copy()
    if sub.empty:
        return float("nan")
    X = imputer.transform(sub[feat_cols].values)
    y = sub[[f"dLAT_{horizon}", f"dLON_{horizon}"]].values
    pred = model.predict(X)
    err = haversine_km(
        sub["LAT"].values + y[:, 0],
        sub["LON"].values + y[:, 1],
        sub["LAT"].values + pred[:, 0],
        sub["LON"].values + pred[:, 1],
    )
    return float(np.median(err)) if err.size else float("nan")


def run_one(window: str, y0: int, y1: int, seed: int) -> list[dict]:
    df = load_window_dataset(y0, y1)
    mh = build_mh_df(df)
    if mh.empty:
        return []
    train, val, test = storm_split(mh, seed=seed)
    base_feats = get_feature_cols(mh)
    feat_cols = base_feats + [c for c in ERA5_FEATS if c in mh.columns]
    mh_tr = mh[mh["SID"].isin(train)]
    imputer = SimpleImputer(strategy="median")
    imputer.fit(mh_tr[feat_cols].values)

    rows = []
    for horizon in H_LABELS:
        model = MultiOutputRegressor(lgb.LGBMRegressor(**LGB_P))
        y_tr = mh_tr[[f"dLAT_{horizon}", f"dLON_{horizon}"]].values
        model.fit(imputer.transform(mh_tr[feat_cols].values), y_tr)
        med_val = eval_horizon(mh, "val", val, feat_cols, imputer, model, horizon)
        med_ts = eval_horizon(mh, "test", test, feat_cols, imputer, model, horizon)
        rows.append(
            {
                "window": window,
                "year_start": y0,
                "year_end": y1,
                "seed": seed,
                "horizon": horizon,
                "n_storms": mh["SID"].nunique(),
                "n_samples": len(mh),
                "median_km_val": med_val,
                "median_km_test": med_ts,
            }
        )
    return rows


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    if OUT.exists():
        prev = pd.read_csv(OUT)
        done = set(zip(prev["window"], prev["seed"]))
        all_rows = prev.to_dict("records")
    else:
        done = set()

    total = len(WINDOWS) * len(DEV_SEEDS)
    n_done = len(done)

    for window, y0, y1 in WINDOWS:
        for seed in DEV_SEEDS:
            if (window, seed) in done:
                print(f"skip {window} seed={seed}", flush=True)
                continue
            print(f"run {window} seed={seed} ({n_done+1}/{total})", flush=True)
            PROG.write_text(
                json.dumps({"status": "running", "window": window, "seed": seed, "updated_at": _utc()}, indent=2),
                encoding="utf-8",
            )
            try:
                rows = run_one(window, y0, y1, seed)
                all_rows.extend(rows)
                pd.DataFrame(all_rows).drop_duplicates(
                    subset=["window", "seed", "horizon"], keep="last"
                ).to_csv(OUT, index=False)
                n_done += 1
            except Exception as exc:
                print(f"ERROR {window} seed={seed}: {exc}", flush=True)
                raise

    summary = (
        pd.DataFrame(all_rows)
        .groupby(["window", "horizon"])[["median_km_val", "median_km_test"]]
        .mean()
        .reset_index()
    )
    print("\n=== Mean median km (val / test) ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    PROG.write_text(
        json.dumps({"status": "complete", "updated_at": _utc(), "n_rows": len(all_rows)}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
