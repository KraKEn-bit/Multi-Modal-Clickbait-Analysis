"""Build modeling CSV: research-ready + point ERA5 + spatial sidecar + physics."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT.parent / "Final_Cyclone_Pred_Results_P-3"
sys.path.insert(0, str(OLD))

from pipeline_common import apply_physics_features, apply_qc  # noqa: E402

READY = OLD / "datasets" / "bangladesh_nextstep_dataset_research_ready.csv"
POINT_ERA5 = ROOT / "datasets" / "tracks_era5.csv"
YEAR_START, YEAR_END = 1940, 2024


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def genesis_year_filter(df: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    df = df.copy()
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"])
    gen = df.groupby("SID")["ISO_TIME"].min().dt.year
    keep = set(gen[(gen >= y0) & (gen <= y1)].index.astype(str))
    return df[df["SID"].astype(str).isin(keep)].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", choices=["5deg", "3deg"], required=True)
    args = parser.parse_args()

    spatial_csv = ROOT / "datasets" / f"tracks_era5_spatial_{args.patch}.csv"
    out_csv = ROOT / "datasets" / f"bangladesh_nextstep_dataset_era5_spatial_{args.patch}.csv"
    prog = ROOT / "results" / f"build_modeling_dataset_spatial_{args.patch}_progress.json"

    t0 = datetime.now(timezone.utc)
    ready = pd.read_csv(READY)
    point = pd.read_csv(POINT_ERA5)
    spatial = pd.read_csv(spatial_csv)
    for df in (ready, point, spatial):
        df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"])

    ready = genesis_year_filter(ready, YEAR_START, YEAR_END)
    sids = set(ready["SID"].astype(str))
    point = point[point["SID"].astype(str).isin(sids)]
    spatial = spatial[spatial["SID"].astype(str).isin(sids)]

    merged = ready.merge(point, on=["SID", "ISO_TIME"], how="inner", suffixes=("", "_pt"))
    merged = merged.merge(spatial, on=["SID", "ISO_TIME"], how="inner")
    n_before = len(merged)
    df = apply_physics_features(apply_qc(merged))
    df.to_csv(out_csv, index=False)

    elapsed_s = (datetime.now(timezone.utc) - t0).total_seconds()
    prog.parent.mkdir(parents=True, exist_ok=True)
    prog.write_text(
        json.dumps(
            {
                "status": "complete",
                "updated_at": _utc(),
                "patch": args.patch,
                "spatial_csv": str(spatial_csv),
                "out_csv": str(out_csv),
                "n_merged": n_before,
                "n_qc": len(df),
                "elapsed_seconds": round(elapsed_s, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"patch={args.patch} merged={n_before} qc={len(df)} -> {out_csv}")


if __name__ == "__main__":
    main()
