"""
SECE + ERA5 dev multi-seed evaluation (1940-2024 window).

Patches sece_v2_train to load bangladesh_nextstep_dataset_era5.csv and use ERA5-aware
subset keys. Checkpoints after every seed for resume.
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

import pipeline_era5  # noqa: E402
import sece_v2_train  # noqa: E402
from eval_protocol import DEV_SEEDS, HELDOUT_SEEDS  # noqa: E402

# Patch loader + environ subset before run_sece_v2
sece_v2_train.load_pipeline_data = pipeline_era5.load_pipeline_data
sece_v2_train.SUBSET_KEYS = pipeline_era5.SUBSET_KEYS_ERA5

OUT_DIR = ROOT / "results" / "sece_era5"
# Labels match sece_v2_train.run_sece_v2 output (track-only baseline uses same names)
SECE_LABELS = {3: "SECE v2 Phase3", 4: "SECE v2 Phase4"}
HORIZONS = ["3h", "12h", "24h", "48h"]
MODELS_PER_SEED = 8
HORIZONS_PER_SEED = 4


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _paths(phase: int, quick: bool) -> dict[str, Path]:
    label = f"phase{phase}" + ("_quick" if quick else "")
    return {
        "raw": OUT_DIR / f"dev_multiseed_{label}_raw.csv",
        "stability": OUT_DIR / f"dev_multiseed_{label}_stability.csv",
        "progress": OUT_DIR / f"dev_multiseed_{label}_progress.json",
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


def write_progress(progress_path: Path, payload: dict) -> None:
    progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize(raw_df: pd.DataFrame, sece_label: str) -> pd.DataFrame:
    flags = []
    for horizon in HORIZONS:
        sub = raw_df[(raw_df["horizon"] == horizon) & (raw_df["model"] == sece_label)]
        best = (
            raw_df[raw_df["horizon"] == horizon]
            .sort_values("rank")
            .groupby("seed")["model"]
            .first()
        )
        winners = best.value_counts()
        flags.append({
            "horizon": horizon,
            "sece_rank1_count": int(sub["is_best"].sum()),
            "n_seeds": sub["seed"].nunique(),
            "unique_winners": ", ".join(f"{k}({v})" for k, v in winners.items()),
        })
    return pd.DataFrame(flags)


def run_dev_eval(
    phase: int = 3,
    quick: bool = False,
    seeds: list[int] | None = None,
    resume: bool = False,
) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = seeds or DEV_SEEDS
    sece_label = SECE_LABELS.get(phase, f"SECE+ERA5 Phase{phase}")
    paths = _paths(phase, quick)

    for s in seeds:
        if s in HELDOUT_SEEDS:
            raise ValueError(f"Seed {s} is held-out")

    t0 = time.time()
    done = completed_seeds(load_raw(paths["raw"])) if resume else set()

    print(f"SECE+ERA5 dev eval phase={phase} quick={quick}", flush=True)
    print(f"Data: {pipeline_era5.DATA_PATH}", flush=True)
    print(f"Features: {len(pipeline_era5.ALL_FEATURE_COLS)} (incl. {len(pipeline_era5.ERA5_FEATS)} ERA5)", flush=True)
    if resume:
        print(f"Resume: {len(done)} seeds done: {sorted(done)}", flush=True)

    pending = [s for s in seeds if s not in done]
    write_progress(
        paths["progress"],
        {
            "updated_at": _utc_now(),
            "status": "running" if pending else "complete",
            "phase": phase,
            "quick": quick,
            "seeds_target": seeds,
            "seeds_done": sorted(done),
            "seeds_remaining": pending,
            "elapsed_hours": round((time.time() - t0) / 3600, 2),
        },
    )

    for i, seed in enumerate(pending, start=1):
        print(f"\n=== Seed {seed} ({i}/{len(pending)} pending, {len(done)}/{len(seeds)} done) ===", flush=True)
        write_progress(
            paths["progress"],
            {
                "updated_at": _utc_now(),
                "status": "running",
                "phase": phase,
                "quick": quick,
                "seeds_target": seeds,
                "seeds_done": sorted(done),
                "seeds_remaining": [s for s in pending if s not in done],
                "current_seed": seed,
                "elapsed_hours": round((time.time() - t0) / 3600, 2),
            },
        )
        try:
            rank_df = sece_v2_train.run_sece_v2(phase=phase, quick=quick, seed=seed, quiet=True)
            rows = [
                {
                    "seed": seed,
                    "horizon": row["horizon"],
                    "model": row["model"],
                    "median_km": row["median_km"],
                    "rank": row["rank"],
                    "is_best": row["is_best"],
                    "router_pick": row.get("router_pick", ""),
                }
                for _, row in rank_df.iterrows()
            ]
            append_seed_rows(paths["raw"], rows)
            done.add(seed)
            elapsed_h = (time.time() - t0) / 3600
            eta_h = elapsed_h / len(done) * (len(seeds) - len(done)) if done else None
            print(f"  Checkpoint saved. Elapsed {elapsed_h:.2f} h", flush=True)
            if eta_h is not None:
                print(f"  ETA remaining ~{eta_h:.1f} h", flush=True)
        except Exception as exc:
            write_progress(
                paths["progress"],
                {
                    "updated_at": _utc_now(),
                    "status": "error",
                    "current_seed": seed,
                    "last_error": str(exc),
                    "seeds_done": sorted(done),
                    "elapsed_hours": round((time.time() - t0) / 3600, 2),
                },
            )
            raise

    raw_df = load_raw(paths["raw"])
    flags_df = summarize(raw_df, sece_label)
    flags_df.to_csv(paths["stability"], index=False)
    write_progress(
        paths["progress"],
        {
            "updated_at": _utc_now(),
            "status": "complete",
            "phase": phase,
            "seeds_done": sorted(done),
            "seeds_remaining": [],
            "elapsed_hours": round((time.time() - t0) / 3600, 2),
        },
    )
    if not quick and phase == 3:
        done_flag = OUT_DIR / "dev_multiseed_phase3_complete.flag"
        done_flag.write_text(f"complete {_utc_now()}\n", encoding="utf-8")
    print(f"\nWrote {paths['raw']}", flush=True)
    print(flags_df.to_string(index=False), flush=True)
    print(f"Total elapsed: {(time.time() - t0) / 3600:.2f} h", flush=True)
    return flags_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=3, choices=[3, 4])
    parser.add_argument("--quick", action="store_true", help="smoke test (~15-30 min total)")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        run_dev_eval(phase=args.phase, quick=args.quick, seeds=args.seeds, resume=args.resume)
    except Exception:
        sys.exit(2)
