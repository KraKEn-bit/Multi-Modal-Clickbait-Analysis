"""Overfit sanity gate: memorize 20 examples (10/10) before full Phase 3 training."""

from __future__ import annotations

import copy
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.custom_dataset import SemanticCollator, SemanticVTCFDataset, prepare_training_dataframe
from models.fusion_network_v2 import SemanticVTCF, SemanticVTCFLoss
from scripts._paths import DEFAULT_CONFIG_PATH, load_config, resolve_path
from scripts.extend_dataset import map_detection_label

logger = logging.getLogger(__name__)

SANITY_SEED = 42
SAMPLES_PER_CLASS = 10
EPOCHS = 40
LEARNING_RATE = 3e-5
PASS_ACCURACY = 0.95
BATCH_SIZE = 10


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_train_subset_indices(dataset: SemanticVTCFDataset, seed: int) -> list[int]:
    """10 clickbait + 10 non-clickbait from the underlying train split."""
    labels = [
        int(dataset.dataframe.iloc[i]["detection_label_int"])
        for i in range(len(dataset))
    ]
    cb_idx = [i for i, label in enumerate(labels) if label == 1]
    nc_idx = [i for i, label in enumerate(labels) if label == 0]
    if len(cb_idx) < SAMPLES_PER_CLASS or len(nc_idx) < SAMPLES_PER_CLASS:
        raise RuntimeError(
            f"Need {SAMPLES_PER_CLASS} per class; got clickbait={len(cb_idx)} "
            f"non_clickbait={len(nc_idx)}"
        )
    rng = random.Random(seed)
    picked = rng.sample(cb_idx, SAMPLES_PER_CLASS) + rng.sample(nc_idx, SAMPLES_PER_CLASS)
    rng.shuffle(picked)
    return picked


def build_tiny_train_dataset(config: dict) -> tuple[SemanticVTCFDataset, list[int]]:
    csv_path = resolve_path(config["data"]["finding2_verified_csv"])
    transcripts_dir = resolve_path(config["data"]["transcripts_dir"])
    train_cfg = config["training"]
    model_cfg = config["model"]

    raw = pd.read_csv(csv_path)
    raw = raw[raw["summary_source"].astype(str).str.lower() == "gemini"].copy()
    train_df = prepare_training_dataframe(raw, transcripts_dir=transcripts_dir)
    train_df["detection_label_int"] = train_df["human_label"].map(map_detection_label)

    seed = int(train_cfg.get("seed", 42))
    val_ratio = float(train_cfg.get("val_ratio", 0.1))
    test_ratio = float(train_cfg.get("test_ratio", 0.1))
    holdout = val_ratio + test_ratio
    indices = list(range(len(train_df)))
    labels = train_df["detection_label_int"].tolist()
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["text_encoder"])
    full_train = SemanticVTCFDataset(
        csv_path=csv_path,
        transcripts_dir=transcripts_dir,
        text_tokenizer=tokenizer,
        max_title_length=int(model_cfg["max_title_length"]),
        max_ocr_length=int(model_cfg["max_ocr_length"]),
        max_summary_length=int(model_cfg["max_summary_length"]),
        split="train",
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    subset_indices = pick_train_subset_indices(full_train, SANITY_SEED)
    video_ids = [str(full_train.dataframe.iloc[i]["video_id"]) for i in subset_indices]
    logger.info(
        "Tiny set: %s samples (%s clickbait, %s non-clickbait)",
        len(subset_indices),
        sum(
            1
            for i in subset_indices
            if int(full_train.dataframe.iloc[i]["detection_label_int"]) == 1
        ),
        sum(
            1
            for i in subset_indices
            if int(full_train.dataframe.iloc[i]["detection_label_int"]) == 0
        ),
    )
    logger.info("video_ids: %s", video_ids)
    return full_train, subset_indices


def run_epoch(
    model: SemanticVTCF,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: SemanticVTCFLoss,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        labels = batch["detection_label"].to(device)
        logits = model(
            batch["title_input_ids"].to(device),
            batch["title_attention_mask"].to(device),
            batch["ocr_input_ids"].to(device),
            batch["ocr_attention_mask"].to(device),
            batch["summary_input_ids"].to(device),
            batch["summary_attention_mask"].to(device),
        )["detection_logits"]
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += int((preds == labels).sum().item())
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(
    model: SemanticVTCF,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        labels = batch["detection_label"].to(device)
        logits = model(
            batch["title_input_ids"].to(device),
            batch["title_attention_mask"].to(device),
            batch["ocr_input_ids"].to(device),
            batch["ocr_attention_mask"].to(device),
            batch["summary_input_ids"].to(device),
            batch["summary_attention_mask"].to(device),
        )["detection_logits"]
        preds = logits.argmax(dim=-1)
        correct += int((preds == labels).sum().item())
        total += labels.size(0)
    return correct / max(total, 1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    set_seed(SANITY_SEED)
    config = load_config(DEFAULT_CONFIG_PATH)
    config = copy.deepcopy(config)
    config["model"]["dropout"] = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    full_train, subset_indices = build_tiny_train_dataset(config)
    tiny = Subset(full_train, subset_indices)
    collator = SemanticCollator(pad_token_id=full_train.text_tokenizer.pad_token_id)
    loader = DataLoader(
        tiny,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        tiny,
        batch_size=len(tiny),
        shuffle=False,
        collate_fn=collator,
    )

    model = SemanticVTCF(config).to(device)
    loss_fn = SemanticVTCFLoss(class_weights=None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)

    print("\n=== Ablation D overfit sanity (title + OCR + summary) ===")
    print(f"samples={len(tiny)} epochs={EPOCHS} lr={LEARNING_RATE} dropout=0 weight_decay=0")
    print()

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, loader, optimizer, loss_fn, device)
        eval_acc = evaluate(model, eval_loader, device)
        best_acc = max(best_acc, eval_acc)
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS or eval_acc >= PASS_ACCURACY:
            print(
                f"epoch {epoch:>2d} | loss={train_loss:.4f} "
                f"train_acc={train_acc:.3f} eval_acc={eval_acc:.3f}"
            )
        if eval_acc >= PASS_ACCURACY:
            print(f"\nPASS: reached {eval_acc:.1%} accuracy at epoch {epoch}")
            break
    else:
        print(f"\nBest eval accuracy: {best_acc:.1%}")
        if best_acc >= PASS_ACCURACY:
            print(f"PASS: memorized tiny set (>= {PASS_ACCURACY:.0%})")
        else:
            print(
                f"FAIL: could not memorize 20 examples (best={best_acc:.1%}, "
                f"need>={PASS_ACCURACY:.0%}). Check forward pass, loss, labels."
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
