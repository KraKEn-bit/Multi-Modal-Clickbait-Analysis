"""Smoke test Phase 2 dataset loading over finding2_verified.csv."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.custom_dataset import SemanticVTCFDataset, get_dataloaders, prepare_training_dataframe
from scripts._paths import DEFAULT_CONFIG_PATH, load_config, resolve_path
from scripts.extend_dataset import map_detection_label

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = load_config(DEFAULT_CONFIG_PATH)
    csv_path = resolve_path(config["data"]["finding2_verified_csv"])
    transcripts_dir = resolve_path(config["data"]["transcripts_dir"])

    raw = __import__("pandas").read_csv(csv_path)
    logger.info("Raw verified CSV: %s rows", len(raw))

    train_df = prepare_training_dataframe(raw, transcripts_dir=transcripts_dir)
    labels = train_df["human_label"].map(map_detection_label)
    clickbait = int((labels == 1).sum())
    non_clickbait = int((labels == 0).sum())
    ocr_usable = int(train_df.get("hook_ocr_usable", False).astype(bool).sum())

    print("\n=== Phase 2 Dataset Smoke Test ===")
    print(f"verified_csv:     {csv_path}")
    print(f"transcripts_dir:  {transcripts_dir}")
    print(f"raw rows:         {len(raw)}")
    print(f"trainable rows:   {len(train_df)}")
    print(f"clickbait:        {clickbait}")
    print(f"non_clickbait:    {non_clickbait}")
    print(f"hook_ocr_usable:  {ocr_usable}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_encoder"])
    train_loader, val_loader, test_loader = get_dataloaders(
        config_path=DEFAULT_CONFIG_PATH,
        tokenizer=tokenizer,
        batch_size=4,
        num_workers=0,
    )
    print(f"split sizes:      train={len(train_loader.dataset)} "
          f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    weights = SemanticVTCFDataset.get_class_weights(csv_path, transcripts_dir)
    print(f"class_weights:    {weights.tolist()}")

    batch = next(iter(train_loader))
    print("\nBatch shapes:")
    for key, value in batch.items():
        if hasattr(value, "shape"):
            print(f"  {key}: {tuple(value.shape)}")
        else:
            print(f"  {key}: list(len={len(value)})")

    missing_summary = 0
    for _, row in raw.iterrows():
        vid = str(row["video_id"])
        sp = transcripts_dir / vid / "summary.txt"
        if not sp.exists() or not sp.read_text(encoding="utf-8").strip():
            missing_summary += 1
    print(f"\nmissing_summary:  {missing_summary}/{len(raw)}")
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
