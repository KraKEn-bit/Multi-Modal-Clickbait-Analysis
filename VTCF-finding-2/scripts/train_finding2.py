"""Train Finding-2 SemanticVTCF ablations B/C/D on the clean Phase 1 dataset."""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.custom_dataset import SemanticCollator, SemanticVTCFDataset, get_dataloaders
from models.fusion_network_v2 import SemanticVTCF, SemanticVTCFLoss
from scripts._paths import DEFAULT_CONFIG_PATH, load_config, resolve_path

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

ABLATION_CHOICES = ("title_only", "title_ocr", "title_summary", "title_ocr_summary")
MIN_TRANSCRIPT_WORDS = 5
ENCODER_LR = 2e-5
HEAD_LR = 3e-5
MAX_GRAD_NORM = 1.0

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Finding-2 SemanticVTCF ablations")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--ablation",
        type=str,
        choices=ABLATION_CHOICES,
        required=True,
        help="A=title_only, B=title_ocr, C=title_summary, D=title_ocr_summary",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override config training.epochs (default: use config)",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="2 epochs, max 5 train batches per epoch (pre-flight)",
    )
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def checkpoint_path(ablation: str) -> Path:
    return CHECKPOINT_DIR / f"finding2_{ablation}.pt"


def log_path(ablation: str) -> Path:
    return LOG_DIR / f"finding2_training_{ablation}.csv"


def init_log(ablation: str) -> None:
    path = log_path(ablation)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(
            ["epoch", "train_loss", "val_f1", "val_accuracy", "val_auc"]
        )


def append_log(ablation: str, row: dict[str, Any]) -> None:
    with log_path(ablation).open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(
            [
                row["epoch"],
                f"{row['train_loss']:.6f}",
                f"{row['val_f1']:.6f}",
                f"{row['val_accuracy']:.6f}",
                f"{row['val_auc']:.6f}",
            ]
        )


def apply_ablation_batch(
    batch: dict[str, Any],
    ablation: str,
    pad_token_id: int,
) -> dict[str, Any]:
    """Zero unused streams at the data level (all-zero attention masks)."""
    out = {key: value for key, value in batch.items()}
    title_ids = out["title_input_ids"].clone()
    title_mask = out["title_attention_mask"].clone()
    ocr_ids = out["ocr_input_ids"].clone()
    ocr_mask = out["ocr_attention_mask"].clone()
    summary_ids = out["summary_input_ids"].clone()
    summary_mask = out["summary_attention_mask"].clone()

    if ablation == "title_only":
        ocr_ids = torch.full_like(ocr_ids, pad_token_id)
        ocr_mask = torch.zeros_like(ocr_mask)
        summary_ids = torch.full_like(summary_ids, pad_token_id)
        summary_mask = torch.zeros_like(summary_mask)
    elif ablation == "title_ocr":
        summary_ids = torch.full_like(summary_ids, pad_token_id)
        summary_mask = torch.zeros_like(summary_mask)
    elif ablation == "title_summary":
        ocr_ids = torch.full_like(ocr_ids, pad_token_id)
        ocr_mask = torch.zeros_like(ocr_mask)
    elif ablation == "title_ocr_summary":
        pass
    else:
        raise ValueError(f"Unknown ablation: {ablation}")

    out["title_input_ids"] = title_ids
    out["title_attention_mask"] = title_mask
    out["ocr_input_ids"] = ocr_ids
    out["ocr_attention_mask"] = ocr_mask
    out["summary_input_ids"] = summary_ids
    out["summary_attention_mask"] = summary_mask
    return out


def describe_active_streams(batch: dict[str, Any], ablation: str) -> str:
    title_active = int(batch["title_attention_mask"].sum().item())
    ocr_active = int(batch["ocr_attention_mask"].sum().item())
    summary_active = int(batch["summary_attention_mask"].sum().item())
    return (
        f"ablation={ablation} | active tokens "
        f"title={title_active} ocr={ocr_active} summary={summary_active}"
    )


def build_optimizer(model: SemanticVTCF, weight_decay: float) -> torch.optim.AdamW:
    encoder_params = list(model.text_encoder.parameters())
    head_params = list(model.fusion.parameters()) + list(model.detection_head.parameters())
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": ENCODER_LR},
            {"params": head_params, "lr": HEAD_LR},
        ],
        weight_decay=weight_decay,
    )


def forward_batch(model: SemanticVTCF, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    outputs = model(
        batch["title_input_ids"].to(device),
        batch["title_attention_mask"].to(device),
        batch["ocr_input_ids"].to(device),
        batch["ocr_attention_mask"].to(device),
        batch["summary_input_ids"].to(device),
        batch["summary_attention_mask"].to(device),
    )
    return outputs["detection_logits"]


def train_one_epoch(
    model: SemanticVTCF,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn: SemanticVTCFLoss,
    device: torch.device,
    ablation: str,
    pad_token_id: int,
    *,
    max_batches: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = apply_ablation_batch(batch, ablation, pad_token_id)
        labels = batch["detection_label"].to(device)
        logits = forward_batch(model, batch, device)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: SemanticVTCF,
    loader: DataLoader,
    loss_fn: SemanticVTCFLoss,
    device: torch.device,
    ablation: str,
    pad_token_id: int,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []

    for batch in loader:
        batch = apply_ablation_batch(batch, ablation, pad_token_id)
        labels = batch["detection_label"].to(device)
        logits = forward_batch(model, batch, device)
        loss = loss_fn(logits, labels)
        losses.append(float(loss.item()))
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        y_prob.extend(probs.cpu().tolist())

    val_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    val_acc = accuracy_score(y_true, y_pred)
    try:
        val_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        val_auc = float("nan")

    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_f1": float(val_f1),
        "val_accuracy": float(val_acc),
        "val_auc": float(val_auc),
    }


def save_checkpoint(
    path: Path,
    *,
    model: SemanticVTCF,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    ablation: str,
    best_val_f1: float,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "ablation": ablation,
            "best_val_f1": best_val_f1,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: SemanticVTCF,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint.get("epoch", 0)), float(checkpoint.get("best_val_f1", 0.0))


def main() -> None:
    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    config = load_config(args.config)
    train_cfg = config.get("training", {})
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    device = get_device()
    print(f"Device: {device}")

    epochs = 2 if args.test_mode else int(args.epochs or train_cfg.get("epochs", 10))
    batch_size = int(train_cfg.get("batch_size", 16))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    max_train_batches = 5 if args.test_mode else None

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_encoder"])
    pad_token_id = tokenizer.pad_token_id or 0

    train_loader, val_loader, test_loader = get_dataloaders(
        config_path=args.config,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=0,
        min_transcript_words=MIN_TRANSCRIPT_WORDS,
    )

    print(
        f"Dataset (gemini, transcript_words>{MIN_TRANSCRIPT_WORDS}): "
        f"train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} "
        f"test={len(test_loader.dataset)}"
    )

    class_weights = SemanticVTCFDataset.get_class_weights(
        config["data"]["finding2_verified_csv"],
        config["data"]["transcripts_dir"],
        min_transcript_words=MIN_TRANSCRIPT_WORDS,
    ).to(device)
    print(f"class_weights: {class_weights.tolist()}")

    model = SemanticVTCF(config).to(device)
    loss_fn = SemanticVTCFLoss(class_weights=class_weights)
    optimizer = build_optimizer(model, weight_decay=weight_decay)

    steps_per_epoch = len(train_loader)
    if max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, max_train_batches)
    total_steps = max(steps_per_epoch * epochs, 1)
    warmup_steps = max(int(total_steps * 0.1), 1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    init_log(args.ablation)

    start_epoch = 0
    best_val_f1 = 0.0
    ckpt = args.resume or checkpoint_path(args.ablation)
    if args.resume and ckpt.exists():
        start_epoch, best_val_f1 = load_checkpoint(ckpt, model, optimizer, scheduler, device)
        print(f"Resumed from {ckpt} at epoch {start_epoch}, best_val_f1={best_val_f1:.4f}")

    first_batch = next(iter(train_loader))
    ablated = apply_ablation_batch(first_batch, args.ablation, pad_token_id)
    print("\n=== First-batch stream audit ===")
    print(describe_active_streams(ablated, args.ablation))
    print(f"ablation={args.ablation} | mode={'TEST' if args.test_mode else 'FULL'}")
    print(f"epochs={epochs} batch_size={batch_size} max_train_batches={max_train_batches}")
    print()

    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
            device,
            args.ablation,
            pad_token_id,
            max_batches=max_train_batches,
        )
        metrics = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            args.ablation,
            pad_token_id,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
        }
        append_log(args.ablation, row)
        print(
            f"epoch {epoch:>2d} | train_loss={train_loss:.4f} "
            f"val_f1={metrics['val_f1']:.4f} "
            f"val_acc={metrics['val_accuracy']:.4f} "
            f"val_auc={metrics['val_auc']:.4f}"
        )

        if metrics["val_f1"] >= best_val_f1:
            best_val_f1 = metrics["val_f1"]
            save_checkpoint(
                checkpoint_path(args.ablation),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                ablation=args.ablation,
                best_val_f1=best_val_f1,
                config=config,
            )
            print(f"  -> saved best checkpoint (val_f1={best_val_f1:.4f})")

    print(f"\nDone. Best val_f1={best_val_f1:.4f} | checkpoint={checkpoint_path(args.ablation)}")


if __name__ == "__main__":
    main()
