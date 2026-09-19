"""
Fixed-test control: train on each date window, evaluate on the SAME modern holdout.

For each DEV seed, test storms = 15% storm-wise split of genesis-1990-2024 cohort only.
All three train windows exclude those test storms from training.
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
OUT = ROOT / "results" / "date_window_fixed_test_control.csv"
PROG = ROOT / "results" / "date_window_fixed_test_control_progress.json"

MODERN_Y0, MODERN_Y1 = 1990, 2024

ERA5_FEATS = [
    "u850", "v850", "u500", "v500", "u200", "v200",
    "msl", "sst", "steer_u", "steer_v", "shear_u", "shear_v", "shear_mag",
    "sst_missing",
]

TRAIN_WINDOWS = [
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


def eval_horizon(mh_test: pd.DataFrame, feat_cols: list[str],
                 imputer: SimpleImputer, model, horizon: str) -> float:
    if mh_test.empty:
        return float("nan")
    X = imputer.transform(mh_test[feat_cols].values)
    y = mh_test[[f"dLAT_{horizon}", f"dLON_{horizon}"]].values
    pred = model.predict(X)
    err = haversine_km(
        mh_test["LAT"].values + y[:, 0],
        mh_test["LON"].values + y[:, 1],
        mh_test["LAT"].values + pred[:, 0],
        mh_test["LON"].values + pred[:, 1],
    )
    return float(np.median(err)) if err.size else float("nan")


def modern_split(seed: int) -> tuple[set, set, set, pd.DataFrame]:
    """Storm-wise split on genesis 1990-2024 cohort; return test mh rows."""
    df = load_window_dataset(MODERN_Y0, MODERN_Y1)
    mh = build_mh_df(df)
    train, val, test = storm_split(mh, seed=seed)
    mh_test = mh[mh["SID"].isin(test)].copy()
    return train, val, test, mh_test


def train_storms_for_window(window: str, y0: int, y1: int, seed: int,
                            test_storms: set, modern_train: set) -> set:
    if window == "modern":
        return set(modern_train)
    df = load_window_dataset(y0, y1)
    mh = build_mh_df(df)
    all_storms = set(mh["SID"].astype(str))
    return {str(s) for s in all_storms - {str(s) for s in test_storms}}


def run_seed(seed: int) -> list[dict]:
    modern_train, _, test_storms, mh_test = modern_split(seed)
    test_storms_s = {str(s) for s in test_storms}
    print(
        f"  seed {seed}: fixed test storms={len(test_storms_s)}  "
        f"samples={len(mh_test)}",
        flush=True,
    )

    base_feats = get_feature_cols(mh_test)
    feat_cols = base_feats + [c for c in ERA5_FEATS if c in mh_test.columns]
    rows = []

    for window, y0, y1 in TRAIN_WINDOWS:
        train_set = train_storms_for_window(window, y0, y1, seed, test_storms_s, modern_train)
        df = load_window_dataset(y0, y1)
        mh = build_mh_df(df)
        mh_tr = mh[mh["SID"].astype(str).isin({str(s) for s in train_set})]
        if mh_tr.empty:
            continue

        imputer = SimpleImputer(strategy="median")
        imputer.fit(mh_tr[feat_cols].values)

        for horizon in H_LABELS:
            model = MultiOutputRegressor(lgb.LGBMRegressor(**LGB_P))
            y_tr = mh_tr[[f"dLAT_{horizon}", f"dLON_{horizon}"]].values
            model.fit(imputer.transform(mh_tr[feat_cols].values), y_tr)
            med = eval_horizon(mh_test, feat_cols, imputer, model, horizon)
            rows.append(
                {
                    "train_window": window,
                    "train_year_start": y0,
                    "train_year_end": y1,
                    "seed": seed,
                    "horizon": horizon,
                    "test_cohort": f"{MODERN_Y0}-{MODERN_Y1}",
                    "n_train_storms": mh_tr["SID"].nunique(),
                    "n_train_samples": len(mh_tr),
                    "n_test_storms": len(test_storms_s),
                    "n_test_samples": len(mh_test),
                    "median_km_test": med,
                }
            )
    return rows


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for i, seed in enumerate(DEV_SEEDS, start=1):
        print(f"run seed {seed} ({i}/{len(DEV_SEEDS)})", flush=True)
        PROG.write_text(
            json.dumps({"status": "running", "seed": seed, "updated_at": _utc()}, indent=2),
            encoding="utf-8",
        )
        all_rows.extend(run_seed(seed))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT, index=False)

    summary = (
        df.groupby(["train_window", "horizon"])["median_km_test"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    print("\n=== Fixed modern test (1990-2024 holdout) — mean median km ===", flush=True)
    print(summary.to_string(index=False), flush=True)

    PROG.write_text(
        json.dumps({"status": "complete", "updated_at": _utc(), "n_rows": len(df)}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
