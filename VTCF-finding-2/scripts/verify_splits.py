"""Verify train/val/test split disjointness and split-stability properties."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.custom_dataset import (
    _build_split_indices,
    _load_persisted_split_csv,
    build_split_assignments,
    prepare_training_dataframe,
)
from scripts._paths import DEFAULT_CONFIG_PATH, load_config, resolve_path
from scripts.extend_dataset import map_detection_label
from scripts.train_finding2 import MIN_TRANSCRIPT_WORDS


def load_training_frame(csv_path: Path, transcripts_dir: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    return prepare_training_dataframe(
        raw,
        transcripts_dir=transcripts_dir,
        require_gemini=True,
        min_transcript_words=MIN_TRANSCRIPT_WORDS,
    )


def split_video_ids(
    dataframe: pd.DataFrame,
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, set[str]]:
    labels = dataframe["human_label"].map(map_detection_label).tolist()
    split_indices = _build_split_indices(labels, val_ratio, test_ratio, seed)
    return {
        name: set(dataframe.iloc[idxs]["video_id"].astype(str))
        for name, idxs in split_indices.items()
    }


def main() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    csv_path = resolve_path(config["data"]["finding2_verified_csv"])
    transcripts_dir = resolve_path(config["data"]["transcripts_dir"])
    train_cfg = config.get("training", {})
    seed = int(train_cfg.get("seed", 42))
    val_ratio = float(train_cfg.get("val_ratio", 0.1))
    test_ratio = float(train_cfg.get("test_ratio", 0.1))

    raw = pd.read_csv(csv_path)
    print("=== finding2_verified.csv ===")
    print(f"Total rows:                     {len(raw)}")
    print(f"Unique video_ids in CSV:        {raw['video_id'].astype(str).nunique()}")
    print(f"Duplicate video_id rows:        {int(raw['video_id'].astype(str).duplicated().sum())}")

    df = load_training_frame(csv_path, transcripts_dir)
    print(f"\nTraining-ready rows (current):  {len(df)}")
    print(f"  filters: summary_source=gemini, transcript_words>{MIN_TRANSCRIPT_WORDS}")

    splits = split_video_ids(df, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    for name in ("train", "val", "test"):
        print(f"{name:5s} unique video_ids:          {len(splits[name])}")

    print("\n=== Pairwise video_id intersections (expect 0) ===")
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    all_disjoint = True
    for a, b in pairs:
        overlap = splits[a] & splits[b]
        print(f"{a} & {b}: {len(overlap)}")
        if overlap:
            all_disjoint = False
            print(f"  examples: {sorted(overlap)[:5]}")

    union = splits["train"] | splits["val"] | splits["test"]
    print(f"\nUnion unique video_ids:         {len(union)}")
    print(f"Sum of split sizes:             {sum(len(s) for s in splits.values())}")
    print(f"Partition covers all rows:      {len(union) == len(df)}")

    print("\n=== Split persistence audit ===")
    split_artifact = resolve_path(config.get("data", {}).get("finding2_split_csv", "data/finding2_split.csv"))
    print(f"Persisted split file exists:    {split_artifact.exists()} ({split_artifact})")
    if split_artifact.exists():
        persisted = _load_persisted_split_csv(split_artifact)
        persisted_sets = {
            name: set(persisted.loc[persisted["split"] == name, "video_id"].astype(str))
            for name in ("train", "val", "test")
        }
        print(f"Persisted rows:                 {len(persisted)}")
        for name in ("train", "val", "test"):
            print(f"  {name:5s} in file:               {len(persisted_sets[name])}")
        matches = all(persisted_sets[name] == splits[name] for name in ("train", "val", "test"))
        print(f"Matches sklearn recomputation:  {matches}")
        if not matches:
            for name in ("train", "val", "test"):
                diff = len(persisted_sets[name] ^ splits[name])
                if diff:
                    print(f"  mismatch {name}: {diff} video_id(s)")
    else:
        print(
            "Split source at runtime:        sklearn recomputation on each "
            "SemanticVTCFDataset init."
        )

    # Stability: if dataset grew mid-pipeline, would test membership change?
    print("\n=== Split stability if dataset size changed (same seed=42) ===")
    current_test = splits["test"]
    checkpoints = [800, 1000, 1200, 1400, len(df)]
    for n in checkpoints:
        if n > len(df):
            continue
        partial = df.iloc[:n]
        partial_splits = split_video_ids(
            partial, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
        partial_test = partial_splits["test"]
        overlap_with_current = len(partial_test & current_test)
        jaccard = (
            overlap_with_current / len(partial_test | current_test)
            if partial_test or current_test
            else 1.0
        )
        print(
            f"n={n:4d} -> test={len(partial_test):3d} | "
            f"overlap with current test: {overlap_with_current:3d} | "
            f"Jaccard vs current test: {jaccard:.3f}"
        )

    print("\n=== Verdict ===")
    if all_disjoint and len(union) == len(df):
        print("CURRENT RUN: train/val/test video_id sets are pairwise disjoint and partition the training frame.")
    else:
        print("CURRENT RUN: DISJOINTNESS OR PARTITION CHECK FAILED.")


if __name__ == "__main__":
    main()
