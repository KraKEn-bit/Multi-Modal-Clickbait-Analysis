"""
Build full modeling dataset: research-ready tracks + ERA5 + physics features.

Window: genesis-year 1940-2024 (locked in docs/date_window_decision.md).
"""

from __future__ import annotations

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
ERA5 = ROOT / "datasets" / "tracks_era5.csv"
OUT = ROOT / "datasets" / "bangladesh_nextstep_dataset_era5.csv"
PROG = ROOT / "results" / "build_modeling_dataset_progress.json"

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
    t0 = datetime.now(timezone.utc)
    PROG.parent.mkdir(parents=True, exist_ok=True)
    PROG.write_text(
        json.dumps({"status": "running", "updated_at": _utc(), "step": "load"}, indent=2),
        encoding="utf-8",
    )

    ready = pd.read_csv(READY)
    era5 = pd.read_csv(ERA5)
    ready["ISO_TIME"] = pd.to_datetime(ready["ISO_TIME"])
    era5["ISO_TIME"] = pd.to_datetime(era5["ISO_TIME"])

    ready = genesis_year_filter(ready, YEAR_START, YEAR_END)
    sids = set(ready["SID"].astype(str))
    era5 = era5[era5["SID"].astype(str).isin(sids)]

    merged = ready.merge(era5, on=["SID", "ISO_TIME"], how="inner", suffixes=("", "_era5"))
    n_before = len(merged)

    PROG.write_text(
        json.dumps(
            {"status": "running", "updated_at": _utc(), "step": "qc_physics", "n_merged": n_before},
            indent=2,
        ),
        encoding="utf-8",
    )

    df = apply_qc(merged)
    df = apply_physics_features(df)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    elapsed_s = (datetime.now(timezone.utc) - t0).total_seconds()
    summary = {
        "status": "complete",
        "updated_at": _utc(),
        "window": f"{YEAR_START}-{YEAR_END}",
        "n_ready_rows": int(len(ready)),
        "n_merged_rows": int(n_before),
        "n_qc_rows": int(len(df)),
        "n_storms": int(df["SID"].nunique()),
        "era5_cols": [c for c in era5.columns if c not in ("SID", "ISO_TIME")],
        "out_path": str(OUT),
        "elapsed_seconds": round(elapsed_s, 1),
    }
    PROG.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Merged {n_before} rows -> QC {len(df)} rows, {df['SID'].nunique()} storms")
    print(f"Elapsed {elapsed_s:.1f}s")
    print(f"Wrote {OUT}")
    print(f"Progress {PROG}")


if __name__ == "__main__":
    main()
