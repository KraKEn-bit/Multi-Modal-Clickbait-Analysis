"""
SECE + ERA5 + spatial patch Phase 3 dev eval (point-ERA5 subsets, NOT environ).

Usage: python sece_era5_spatial_dev_eval.py --patch 5deg [--resume]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT.parent / "Final_Cyclone_Pred_Results_P-3"
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(ROOT))

import pipeline_era5 as pe  # noqa: E402
import pipeline_era5_spatial as ps  # noqa: E402
import sece_v2_train  # noqa: E402
from eval_protocol import DEV_SEEDS, HELDOUT_SEEDS  # noqa: E402

SECE_LABEL = "SECE v2 Phase3"
HORIZONS = ["3h", "12h", "24h", "48h"]
MODELS_PER_SEED = 8
HORIZONS_PER_SEED = 4

SPATIAL_SUBSET_KEYS = {
    **pe.SUBSET_KEYS_ERA5,
    "environ": pe.SUBSET_KEYS_ERA5["environ"]
    + ["shear_grad", "wspd850_std", "msl_trough", "msl_patch", "shear_asym"],
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _out_dir(patch: str) -> Path:
    return ROOT / "results" / f"sece_era5_spatial_{patch}"


def _paths(patch: str, quick: bool) -> dict[str, Path]:
    out = _out_dir(patch)
    label = "phase3" + ("_quick" if quick else "")
    return {
        "raw": out / f"dev_multiseed_{label}_raw.csv",
        "stability": out / f"dev_multiseed_{label}_stability.csv",
        "progress": out / f"dev_multiseed_{label}_progress.json",
    }


def load_raw(raw_path: Path) -> pd.DataFrame:
    if raw_path.exists():
        return pd.read_csv(raw_path)
    return pd.DataFrame(columns=["seed", "horizon", "model", "median_km", "rank", "is_best", "router_pick"])


def completed_seeds(raw_df: pd.DataFrame) -> set[int]:
    if raw_df.empty:
        return set()
    need = MODELS_PER_SEED * HORIZONS_PER_SEED
    done = set()
    for seed, grp in raw_df.groupby("seed"):
        if grp["horizon"].nunique() == HORIZONS_PER_SEED and len(grp) >= need:
            done.add(int(seed))
    return done


def append_seed_rows(raw_path: Path, rows: list[dict]) -> None:
    raw_df = load_raw(raw_path)
    new_df = pd.DataFrame(rows)
    seed = int(new_df["seed"].iloc[0])
    raw_df = raw_df[raw_df["seed"] != seed] if not raw_df.empty else raw_df
    pd.concat([raw_df, new_df], ignore_index=True).to_csv(raw_path, index=False)


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    flags = []
    for horizon in HORIZONS:
        sub = raw_df[(raw_df["horizon"] == horizon) & (raw_df["model"] == SECE_LABEL)]
        best = (
            raw_df[raw_df["horizon"] == horizon]
            .sort_values("rank")
            .groupby("seed")["model"]
            .first()
        )
        flags.append({
            "horizon": horizon,
            "sece_rank1_count": int(sub["is_best"].sum()),
            "n_seeds": sub["seed"].nunique(),
            "unique_winners": ", ".join(f"{k}({v})" for k, v in best.value_counts().items()),
        })
    return pd.DataFrame(flags)


def run_dev_eval(patch: str, quick: bool = False, seeds: list[int] | None = None, resume: bool = False):
    out_dir = _out_dir(patch)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = _paths(patch, quick)
    seeds = seeds or DEV_SEEDS

    def _loader(seed=pe.SPLIT_SEED, train_storms=None, val_storms=None, test_storms=None):
        return ps.load_pipeline_data(
            patch=patch, seed=seed,
            train_storms=train_storms, val_storms=val_storms, test_storms=test_storms,
        )

    sece_v2_train.load_pipeline_data = _loader
    sece_v2_train.SUBSET_KEYS = SPATIAL_SUBSET_KEYS
    # point-ERA5 Phase 3 subsets (NOT environ expert architecture)
    sece_v2_train.SECE_SUBSETS_P3_3H = ["position", "motion", "full"]
    sece_v2_train.SECE_SUBSETS_P3_OTHER = ["motion", "physics", "full"]

    for s in seeds:
        if s in HELDOUT_SEEDS:
            raise ValueError(f"Seed {s} is held-out")

    t0 = time.time()
    done = completed_seeds(load_raw(paths["raw"])) if resume else set()
    cfg = ps.config(patch)

    print(f"SECE+ERA5+spatial patch={patch} Phase3 (point subsets)", flush=True)
    print(f"Data: {cfg['data_path']}", flush=True)
    print(f"Features: {len(cfg['all_feature_cols'])}", flush=True)

    pending = [s for s in seeds if s not in done]
    meta = {"patch": patch, "architecture": "phase3_point_subsets_spatial"}
    paths["progress"].write_text(
        json.dumps({
            "updated_at": _utc_now(), "status": "running" if pending else "complete",
            "meta": meta, "seeds_done": sorted(done), "seeds_remaining": pending,
            "elapsed_hours": round((time.time() - t0) / 3600, 2),
        }, indent=2),
        encoding="utf-8",
    )

    for i, seed in enumerate(pending, start=1):
        print(f"\n=== Seed {seed} ({i}/{len(pending)}) ===", flush=True)
        rank_df = sece_v2_train.run_sece_v2(phase=3, quick=quick, seed=seed, quiet=True)
        rows = [
            {
                "seed": seed, "horizon": row["horizon"], "model": row["model"],
                "median_km": row["median_km"], "rank": row["rank"],
                "is_best": row["is_best"], "router_pick": row.get("router_pick", ""),
            }
            for _, row in rank_df.iterrows()
        ]
        append_seed_rows(paths["raw"], rows)
        done.add(seed)
        print(f"  Elapsed {(time.time()-t0)/3600:.2f} h", flush=True)

    raw_df = load_raw(paths["raw"])
    flags_df = summarize(raw_df)
    flags_df.to_csv(paths["stability"], index=False)
    if not quick:
        (out_dir / "dev_multiseed_phase3_complete.flag").write_text(f"complete {_utc_now()}\n")
    print(flags_df.to_string(index=False), flush=True)
    return flags_df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--patch", choices=["5deg", "3deg"], required=True)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    args = p.parse_args()
    try:
        run_dev_eval(args.patch, quick=args.quick, seeds=args.seeds, resume=args.resume)
    except Exception:
        sys.exit(2)
