"""VTCF training script with TDS contrastive loss, ablations, and checkpointing."""

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
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.custom_dataset import VTCFDataset, get_dataloaders
from models.fusion_network import VTCF, VTCFLoss, load_config

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

GRAD_ACCUM_STEPS = 4
MAX_GRAD_NORM = 1.0
FREEZE_EPOCHS = 2
BACKBONE_LR = 1e-5
HEAD_LR = 2e-5

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train the VTCF model")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint for resuming training",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run a short smoke-test training loop",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        choices=["text_only", "vision_only", "full"],
        default="full",
        help="Ablation mode for modality dropout",
    )
    return parser.parse_args()


def training_log_path(ablation: str) -> Path:
    """Return per-ablation CSV log path."""
    return LOG_DIR / f"training_{ablation}.csv"


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Select the best available compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device


def setup_output_dirs() -> None:
    """Create checkpoint and log directories."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def init_training_log(ablation: str) -> None:
    """Initialize CSV log file with headers if it does not exist."""
    log_path = training_log_path(ablation)
    if log_path.exists():
        return

    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "det_loss",
                "contrast_loss",
                "val_loss",
                "val_f1",
                "val_accuracy",
                "val_auc",
                "tds_correlation",
                "tds_mean_clickbait",
                "tds_mean_nonclickbait",
                "lr",
            ]
        )


def append_training_log(ablation: str, row: dict[str, Any]) -> None:
    """Append one epoch of metrics to the training CSV log."""
    with training_log_path(ablation).open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                row["epoch"],
                f"{row['train_loss']:.6f}",
                f"{row['det_loss']:.6f}",
                f"{row['contrast_loss']:.6f}",
                f"{row['val_loss']:.6f}",
                f"{row['val_f1']:.6f}",
                f"{row['val_accuracy']:.6f}",
                f"{row['val_auc']:.6f}",
                f"{row['tds_correlation']:.6f}"
                if row["tds_correlation"] is not None
                else "",
                f"{row['tds_mean_clickbait']:.6f}"
                if row["tds_mean_clickbait"] is not None
                else "",
                f"{row['tds_mean_nonclickbait']:.6f}"
                if row["tds_mean_nonclickbait"] is not None
                else "",
                f"{row['lr']:.8f}",
            ]
        )


def apply_ablation(
    batch: dict[str, Any],
    ablation: str,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply modality ablation to a batch before the forward pass."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    pixel_values = batch["pixel_values"].to(device)

    if ablation == "text_only":
        pixel_values = torch.zeros_like(pixel_values)
    elif ablation == "vision_only":
        input_ids = torch.full_like(input_ids, pad_token_id)
        attention_mask = torch.zeros_like(attention_mask)

    return input_ids, attention_mask, pixel_values


def build_optimizer(
    model: VTCF,
    weight_decay: float,
    freeze_backbones: bool,
    ablation: str = "full",
) -> torch.optim.AdamW:
    """Create AdamW with separate learning rates for backbones and heads."""
    if ablation == "text_only":
        model.configure_text_only_training()
        return torch.optim.AdamW(
            [
                {"params": model.text_encoder.parameters(), "lr": BACKBONE_LR},
                {"params": model.detection_head.parameters(), "lr": HEAD_LR},
            ],
            weight_decay=weight_decay,
        )

    if freeze_backbones:
        model.freeze_backbones()
        return torch.optim.AdamW(
            model.head_parameters(),
            lr=HEAD_LR,
            weight_decay=weight_decay,
        )

    model.unfreeze_backbones()
    return torch.optim.AdamW(
        [
            {"params": model.backbone_parameters(), "lr": BACKBONE_LR},
            {"params": model.head_parameters(), "lr": HEAD_LR},
        ],
        weight_decay=weight_decay,
    )


def run_forward(
    model: VTCF,
    ablation: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
) -> dict[str, torch.Tensor | None]:
    """Run the correct forward path for the active ablation mode."""
    if ablation == "text_only":
        return model.forward_text_only(input_ids, attention_mask)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
    )


def compute_loss_dict(
    loss_fn: VTCFLoss,
    outputs: dict[str, torch.Tensor | None],
    detection_labels: torch.Tensor,
    attribution_labels: torch.Tensor,
    ablation: str,
) -> dict[str, torch.Tensor]:
    """Compute losses; text-only uses detection cross-entropy only."""
    if ablation == "text_only":
        detection_loss = loss_fn.detection_loss_fn(
            outputs["detection_logits"],
            detection_labels,
        )
        zero = torch.zeros((), device=detection_loss.device, dtype=detection_loss.dtype)
        return {
            "total_loss": detection_loss,
            "detection_loss": detection_loss,
            "contrastive_loss": zero,
            "attribution_loss": zero,
        }
    return loss_fn(outputs, detection_labels, attribution_labels)


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    loss_fn: VTCFLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler | None,
    device: torch.device,
    ablation: str,
    pad_token_id: int,
    epoch: int,
    test_mode: bool = False,
) -> dict[str, float]:
    """Run one training epoch with gradient accumulation."""
    model.train()
    use_amp = scaler is not None and device.type == "cuda"

    running_total = 0.0
    running_detection = 0.0
    running_contrastive = 0.0
    running_tds = 0.0
    num_batches = 0

    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)

    for batch_index, batch in enumerate(progress):
        if test_mode and batch_index >= 5:
            break

        input_ids, attention_mask, pixel_values = apply_ablation(
            batch,
            ablation=ablation,
            pad_token_id=pad_token_id,
            device=device,
        )
        detection_labels = batch["detection_label"].to(device)
        attribution_labels = batch["attribution_label"].to(device)

        with autocast(enabled=use_amp):
            outputs = run_forward(
                model,
                ablation=ablation,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )
            loss_dict = compute_loss_dict(
                loss_fn,
                outputs,
                detection_labels,
                attribution_labels,
                ablation=ablation,
            )
            total_loss = loss_dict["total_loss"]
            scaled_loss = total_loss / GRAD_ACCUM_STEPS

        if (
            ablation == "text_only"
            and epoch == 1
            and batch_index % 10 == 0
        ):
            mean_probs = torch.softmax(outputs["detection_logits"], dim=-1).mean(dim=0)
            print(
                f"[text_only debug] batch={batch_index} "
                f"det_loss={loss_dict['detection_loss'].item():.4f} "
                f"total_loss={total_loss.item():.4f} "
                f"contrast_loss={loss_dict['contrastive_loss'].item():.4f} "
                f"mean_probs={mean_probs.detach().cpu().tolist()}"
            )

        if use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if (batch_index + 1) % GRAD_ACCUM_STEPS == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        running_total += total_loss.item()
        running_detection += loss_dict["detection_loss"].item()
        running_contrastive += loss_dict["contrastive_loss"].item()
        tds_tensor = outputs.get("tds_computed")
        if tds_tensor is not None:
            running_tds += tds_tensor.mean().item()
        num_batches += 1

        tds_display = tds_tensor.mean().item() if tds_tensor is not None else 0.0
        progress.set_postfix(
            loss=f"{total_loss.item():.4f}",
            det=f"{loss_dict['detection_loss'].item():.4f}",
            contrast=f"{loss_dict['contrastive_loss'].item():.4f}",
            tds=f"{tds_display:.3f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

    if num_batches > 0 and num_batches % GRAD_ACCUM_STEPS != 0:
        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    if num_batches == 0:
        return {
            "train_loss": 0.0,
            "det_loss": 0.0,
            "contrast_loss": 0.0,
            "batch_tds_mean": 0.0,
        }

    return {
        "train_loss": running_total / num_batches,
        "det_loss": running_detection / num_batches,
        "contrast_loss": running_contrastive / num_batches,
        "batch_tds_mean": running_tds / num_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: VTCFLoss,
    device: torch.device,
    ablation: str,
    pad_token_id: int,
    test_mode: bool = False,
) -> dict[str, Any]:
    """Evaluate the model on the validation split."""
    model.eval()

    all_detection_labels: list[int] = []
    all_detection_probs: list[float] = []
    all_detection_preds: list[int] = []
    all_tds_scores: list[float] = []

    total_loss = 0.0
    num_batches = 0

    for batch_index, batch in enumerate(val_loader):
        if test_mode and batch_index >= 5:
            break

        input_ids, attention_mask, pixel_values = apply_ablation(
            batch,
            ablation=ablation,
            pad_token_id=pad_token_id,
            device=device,
        )
        detection_labels = batch["detection_label"].to(device)
        attribution_labels = batch["attribution_label"].to(device)

        outputs = run_forward(
            model,
            ablation=ablation,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
        loss_dict = compute_loss_dict(
            loss_fn,
            outputs,
            detection_labels,
            attribution_labels,
            ablation=ablation,
        )
        batch_loss = loss_dict["total_loss"]

        detection_logits = outputs["detection_logits"]
        detection_probs = torch.softmax(detection_logits, dim=-1)[:, 1]
        detection_preds = torch.argmax(detection_logits, dim=-1)

        total_loss += batch_loss.item()
        num_batches += 1

        all_detection_labels.extend(detection_labels.cpu().tolist())
        all_detection_probs.extend(detection_probs.cpu().tolist())
        all_detection_preds.extend(detection_preds.cpu().tolist())
        tds_tensor = outputs.get("tds_computed")
        if tds_tensor is not None:
            all_tds_scores.extend(tds_tensor.cpu().numpy().tolist())

    if num_batches == 0:
        return {
            "val_loss": 0.0,
            "val_f1_detection": 0.0,
            "val_accuracy": 0.0,
            "val_auc_detection": float("nan"),
            "tds_correlation": None,
            "tds_mean_clickbait": None,
            "tds_mean_nonclickbait": None,
        }

    detection_labels_array = np.array(all_detection_labels)
    detection_preds_array = np.array(all_detection_preds)

    val_f1_detection = f1_score(
        detection_labels_array,
        detection_preds_array,
        average="macro",
        zero_division=0,
    )
    val_accuracy = accuracy_score(detection_labels_array, detection_preds_array)

    try:
        val_auc = roc_auc_score(detection_labels_array, all_detection_probs)
    except ValueError:
        val_auc = float("nan")

    tds_mean_clickbait = None
    tds_mean_nonclickbait = None
    tds_correlation = None
    if ablation != "text_only" and all_tds_scores:
        tds_array = np.array(all_tds_scores)
        clickbait_mask = detection_labels_array == 1
        non_clickbait_mask = detection_labels_array == 0
        tds_mean_clickbait = (
            float(tds_array[clickbait_mask].mean()) if clickbait_mask.any() else None
        )
        tds_mean_nonclickbait = (
            float(tds_array[non_clickbait_mask].mean()) if non_clickbait_mask.any() else None
        )
        if len(np.unique(detection_labels_array)) > 1:
            tds_correlation = float(pearsonr(detection_labels_array, tds_array)[0])

    return {
        "val_loss": total_loss / num_batches,
        "val_f1_detection": float(val_f1_detection),
        "val_accuracy": float(val_accuracy),
        "val_auc_detection": float(val_auc),
        "tds_correlation": tds_correlation,
        "tds_mean_clickbait": tds_mean_clickbait,
        "tds_mean_nonclickbait": tds_mean_nonclickbait,
    }


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler | None,
    best_val_f1: float,
    config: dict[str, Any],
    ablation_mode: str,
    val_metrics: dict[str, Any] | None = None,
) -> None:
    """Persist a training checkpoint to disk."""
    checkpoint = {
        "epoch": epoch,
        "ablation_mode": ablation_mode,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val_f1": best_val_f1,
        "val_metrics": val_metrics or {},
        "config": config,
    }
    torch.save(checkpoint, path)
    logger.info("Saved checkpoint to %s", path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler | None,
    device: torch.device,
) -> tuple[int, float]:
    """Load a checkpoint and restore training state."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = int(checkpoint["epoch"]) + 1
    best_val_f1 = float(checkpoint.get("best_val_f1", 0.0))
    logger.info("Resumed from %s at epoch %s", path, start_epoch)
    return start_epoch, best_val_f1


def count_optimizer_steps_per_epoch(
    loader_len: int,
    test_mode: bool,
) -> int:
    """Count optimizer steps per epoch after gradient accumulation."""
    num_batches = min(loader_len, 5) if test_mode else loader_len
    return max((num_batches + GRAD_ACCUM_STEPS - 1) // GRAD_ACCUM_STEPS, 1)


def main() -> None:
    """Run the full VTCF training pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args = parse_args()
    config = load_config(args.config)

    if args.test_mode:
        print("\n=== TEST MODE ===\n")

    set_seed(int(config["training"]["seed"]))
    device = get_device()
    setup_output_dirs()
    init_training_log(args.ablation)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_encoder"])
    batch_size = int(config["training"]["batch_size"])
    num_workers = 0 if args.test_mode else 4

    train_loader, val_loader, _ = get_dataloaders(
        config_path=args.config,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if len(train_loader.dataset) == 0:
        raise RuntimeError("Training dataset is empty. Run ingestion before training.")

    class_weights = VTCFDataset.get_class_weights(config["data"]["verified_csv"]).to(device)
    model = VTCF(config).to(device)
    loss_fn = VTCFLoss(class_weights=class_weights).to(device)

    weight_decay = float(config["training"]["weight_decay"])
    epochs = 2 if args.test_mode else int(config["training"]["epochs"])

    scaler = GradScaler(enabled=device.type == "cuda")

    start_epoch = 0
    best_val_f1 = 0.0
    if args.resume is not None:
        placeholder_optimizer = build_optimizer(
            model,
            weight_decay=weight_decay,
            freeze_backbones=False,
            ablation=args.ablation,
        )
        placeholder_scheduler = get_cosine_schedule_with_warmup(
            placeholder_optimizer,
            num_warmup_steps=1,
            num_training_steps=1,
        )
        start_epoch, best_val_f1 = load_checkpoint(
            args.resume,
            model,
            placeholder_optimizer,
            placeholder_scheduler,
            scaler if device.type == "cuda" else None,
            device,
        )

    print(f"Ablation mode: {args.ablation}")
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")
    print(f"Effective batch size: {batch_size * GRAD_ACCUM_STEPS}")

    checkpoint_name = f"best_model_{args.ablation}.pt"
    last_checkpoint_name = f"last_model_{args.ablation}.pt"

    steps_per_epoch = count_optimizer_steps_per_epoch(len(train_loader), args.test_mode)
    total_steps = max(steps_per_epoch * epochs, 1)
    warmup_steps = max(int(total_steps * 0.1), 1)
    warmup_steps = min(warmup_steps, total_steps)

    optimizer = build_optimizer(
        model,
        weight_decay=weight_decay,
        freeze_backbones=True,
        ablation=args.ablation,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    if args.ablation == "text_only":
        print("  Text-only mode: ViT/fusion frozen; training BERT + detection head")
    else:
        print("  Phase 1 (frozen backbones)")

    for epoch in range(start_epoch, epochs):
        if epoch == FREEZE_EPOCHS and args.ablation != "text_only":
            remaining_steps = max(steps_per_epoch * (epochs - epoch), 1)
            optimizer = build_optimizer(
                model,
                weight_decay=weight_decay,
                freeze_backbones=False,
                ablation=args.ablation,
            )
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=max(int(remaining_steps * 0.1), 1),
                num_training_steps=remaining_steps,
            )
            print("  Phase 2 (full fine-tune — backbones unfrozen)")

        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if device.type == "cuda" else None,
            device=device,
            ablation=args.ablation,
            pad_token_id=tokenizer.pad_token_id,
            epoch=epoch + 1,
            test_mode=args.test_mode,
        )

        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            ablation=args.ablation,
            pad_token_id=tokenizer.pad_token_id,
            test_mode=args.test_mode,
        )

        current_lr = scheduler.get_last_lr()[0]
        log_row = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["train_loss"],
            "det_loss": train_metrics["det_loss"],
            "contrast_loss": train_metrics["contrast_loss"],
            "val_loss": val_metrics["val_loss"],
            "val_f1": val_metrics["val_f1_detection"],
            "val_accuracy": val_metrics["val_accuracy"],
            "val_auc": val_metrics["val_auc_detection"],
            "tds_correlation": val_metrics["tds_correlation"],
            "tds_mean_clickbait": val_metrics["tds_mean_clickbait"],
            "tds_mean_nonclickbait": val_metrics["tds_mean_nonclickbait"],
            "lr": current_lr,
        }
        append_training_log(args.ablation, log_row)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_metrics['train_loss']:.4f} | "
            f"det={train_metrics['det_loss']:.4f} | "
            f"contrast={train_metrics['contrast_loss']:.4f} | "
            f"val_f1={val_metrics['val_f1_detection']:.4f} | "
            f"val_auc={val_metrics['val_auc_detection']:.4f} | "
            f"tds_r={val_metrics['tds_correlation']} | "
            f"lr={current_lr:.2e}"
        )
        print(
            "  TDS mean | clickbait="
            f"{val_metrics['tds_mean_clickbait']} | non-clickbait="
            f"{val_metrics['tds_mean_nonclickbait']}"
        )

        save_checkpoint(
            CHECKPOINT_DIR / last_checkpoint_name,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if device.type == "cuda" else None,
            best_val_f1=best_val_f1,
            config=config,
            ablation_mode=args.ablation,
            val_metrics=val_metrics,
        )

        if val_metrics["val_f1_detection"] > best_val_f1:
            best_val_f1 = val_metrics["val_f1_detection"]
            save_checkpoint(
                CHECKPOINT_DIR / checkpoint_name,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler if device.type == "cuda" else None,
                best_val_f1=best_val_f1,
                config=config,
                ablation_mode=args.ablation,
                val_metrics=val_metrics,
            )
            print(f"  New best validation F1: {best_val_f1:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    main()
