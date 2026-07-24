"""VTCF evaluation suite with ablation comparisons and statistical testing."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import chi2 as chi2_dist
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.custom_dataset import ATTRIBUTION_LABELS, VTCFCollator, VTCFDataset, get_dataloaders
from models.fusion_network import VTCF, load_config
from models.interpretability import (
    compute_tds,
    generate_paper_results_table,
    run_tds_analysis_from_csv,
    visualize_attention_storyboard,
)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_HARD_SUBSET_CSV = (
    PROJECT_ROOT / "data" / "baseline_banglabert_model" / "hard_subset_video_ids.csv"
)

logger = logging.getLogger(__name__)

ABLATION_CHECKPOINTS = {
    "text_only": CHECKPOINT_DIR / "best_model_text_only.pt",
    "vision_only": CHECKPOINT_DIR / "best_model_vision_only.pt",
    "full": CHECKPOINT_DIR / "best_model_full.pt",
}

LEGACY_CHECKPOINTS = [
    CHECKPOINT_DIR / "best_model.pt",
    CHECKPOINT_DIR / "last_model.pt",
]

AMBIGUOUS_LOWER = 0.45
AMBIGUOUS_UPPER = 0.55


def get_device() -> torch.device:
    """Select the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def unload_model(model: nn.Module, device: torch.device) -> None:
    """Remove a model from memory before loading another checkpoint."""
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def run_evaluation(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    ablation: str = "full",
    pad_token_id: int = 0,
) -> dict[str, Any]:
    """
    Run full evaluation and return detection, attribution, and TDS metrics.

    Args:
        model: Trained VTCF model.
        dataloader: Evaluation DataLoader.
        device: Target device.
        ablation: Modality ablation mode used during forward pass.
        pad_token_id: Tokenizer pad token id for vision-only ablation.

    Returns:
        Dictionary of metrics and raw prediction artifacts.
    """
    model.eval()

    video_ids: list[str] = []
    true_labels: list[int] = []
    pred_labels: list[int] = []
    pred_probs: list[float] = []
    tds_scores: list[float] = []
    ingestion_tds: list[float] = []
    attribution_true: list[np.ndarray] = []
    attribution_pred: list[np.ndarray] = []

    for batch in tqdm(dataloader, desc=f"Evaluating ({ablation})", leave=False):
        input_ids, attention_mask, pixel_values = apply_ablation(
            batch,
            ablation=ablation,
            pad_token_id=pad_token_id,
            device=device,
        )

        outputs = run_forward(
            model,
            ablation=ablation,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

        detection_logits = outputs["detection_logits"]
        attribution_logits = outputs.get("attribution_logits")
        probabilities = torch.softmax(detection_logits, dim=-1)[:, 1]
        predictions = torch.argmax(detection_logits, dim=-1)

        video_ids.extend(batch["video_id"])
        true_labels.extend(batch["detection_label"].cpu().tolist())
        pred_labels.extend(predictions.cpu().tolist())
        pred_probs.extend(probabilities.cpu().tolist())
        ingestion_tds.extend(batch["tds_score"].cpu().tolist())

        tds_tensor = outputs.get("tds_computed")
        if tds_tensor is not None:
            model_tds = tds_tensor.cpu().numpy()
        elif outputs.get("temporal_visual_matrix") is not None:
            model_tds = compute_tds(outputs["temporal_visual_matrix"]).cpu().numpy()
        else:
            model_tds = np.full(len(batch["video_id"]), np.nan)
        tds_scores.extend(model_tds.tolist())

        if attribution_logits is not None:
            attribution_true.append(batch["attribution_label"].cpu().numpy())
            attribution_pred.append((torch.sigmoid(attribution_logits) > 0.5).cpu().numpy())

    if not true_labels:
        return {
            "detection": {
                "f1_macro": 0.0,
                "f1_weighted": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "auc_roc": float("nan"),
                "accuracy": 0.0,
                "confusion_matrix": [[0, 0], [0, 0]],
            },
            "attribution": {
                "f1_per_class": [0.0] * len(ATTRIBUTION_LABELS),
                "f1_macro": 0.0,
                "hamming_loss": float("nan"),
            },
            "tds_correlation": float("nan"),
            "predictions": {
                "video_ids": [],
                "true_labels": [],
                "pred_labels": [],
                "pred_probs": [],
                "tds_scores": [],
            },
        }

    y_true = np.array(true_labels)
    y_pred = np.array(pred_labels)
    y_prob = np.array(pred_probs)
    tds_array = np.array(tds_scores)

    try:
        auc_roc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc_roc = float("nan")

    if ablation == "text_only" or np.all(np.isnan(tds_array)):
        tds_corr = float("nan")
    else:
        try:
            tds_corr = float(pearsonr(tds_array, y_true)[0])
        except Exception:
            tds_corr = float("nan")

    if attribution_true:
        attr_true = np.vstack(attribution_true)
        attr_pred = np.vstack(attribution_pred)
        attr_f1_per_class = f1_score(
            attr_true,
            attr_pred,
            average=None,
            zero_division=0,
        ).tolist()
        attr_f1_macro = float(f1_score(attr_true, attr_pred, average="macro", zero_division=0))
        attr_hamming = float(hamming_loss(attr_true, attr_pred))
    else:
        attr_f1_per_class = [0.0] * len(ATTRIBUTION_LABELS)
        attr_f1_macro = 0.0
        attr_hamming = float("nan")

    return {
        "detection": {
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "auc_roc": auc_roc,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        },
        "attribution": {
            "f1_per_class": attr_f1_per_class,
            "f1_macro": attr_f1_macro,
            "hamming_loss": attr_hamming,
        },
        "tds_correlation": tds_corr,
        "predictions": {
            "video_ids": video_ids,
            "true_labels": true_labels,
            "pred_labels": pred_labels,
            "pred_probs": pred_probs,
            "tds_scores": tds_scores,
        },
    }


def mcnemar_test(
    model_a_preds: np.ndarray | list[int],
    model_b_preds: np.ndarray | list[int],
    true_labels: np.ndarray | list[int],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> dict[str, Any]:
    """
    Perform McNemar's test on paired classifier predictions.

    Args:
        model_a_preds: Predictions from model A.
        model_b_preds: Predictions from model B.
        true_labels: Ground-truth labels.
        model_a_name: Display name for model A.
        model_b_name: Display name for model B.

    Returns:
        Dictionary with chi-square statistic, p-value, and significance flag.
    """
    y_true = np.asarray(true_labels)
    pred_a = np.asarray(model_a_preds)
    pred_b = np.asarray(model_b_preds)

    model_a_correct = pred_a == y_true
    model_b_correct = pred_b == y_true

    b_count = int(np.sum(model_a_correct & ~model_b_correct))
    c_count = int(np.sum(~model_a_correct & model_b_correct))

    if b_count + c_count == 0:
        chi2_stat = 0.0
        p_value = 1.0
    else:
        chi2_stat = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
        p_value = float(chi2_dist.sf(chi2_stat, df=1))

    significant = p_value < 0.05
    if significant:
        if b_count > c_count:
            print(f"{model_a_name} significantly outperforms {model_b_name} (p={p_value:.4f})")
        elif c_count > b_count:
            print(f"{model_b_name} significantly outperforms {model_a_name} (p={p_value:.4f})")
        else:
            print(
                f"No directional advantage between {model_a_name} and {model_b_name} "
                f"(p={p_value:.4f})"
            )
    else:
        print(
            f"No significant difference between {model_a_name} and {model_b_name} "
            f"(p={p_value:.4f})"
        )

    return {
        "chi2": float(chi2_stat),
        "p_value": float(p_value),
        "significant": significant,
        "discordant_a_correct": b_count,
        "discordant_b_correct": c_count,
    }


def resolve_checkpoint_path(
    checkpoint_path: Path | str,
    condition: str | None = None,
) -> Path:
    """
    Resolve a checkpoint path, falling back to legacy names when needed.

    Args:
        checkpoint_path: Requested checkpoint path.
        condition: Optional ablation condition for additional fallbacks.

    Returns:
        Existing checkpoint path.

    Raises:
        FileNotFoundError: If no matching checkpoint exists.
    """
    path = Path(checkpoint_path)
    if path.exists():
        return path

    candidates: list[Path] = []
    if condition == "full":
        candidates.extend([ABLATION_CHECKPOINTS["full"], *LEGACY_CHECKPOINTS])
    elif condition in ABLATION_CHECKPOINTS:
        candidates.append(ABLATION_CHECKPOINTS[condition])
    else:
        candidates.extend(
            [
                CHECKPOINT_DIR / "best_model_full.pt",
                CHECKPOINT_DIR / "best_model.pt",
                CHECKPOINT_DIR / "last_model.pt",
            ]
        )

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            logger.warning("Checkpoint %s not found; using %s", path, candidate)
            return candidate

    available = sorted(p.name for p in CHECKPOINT_DIR.glob("*.pt")) if CHECKPOINT_DIR.exists() else []
    available_text = ", ".join(available) if available else "none"
    raise FileNotFoundError(
        f"Checkpoint not found: {path}. Available checkpoints: {available_text}"
    )


def load_checkpoint_model(
    checkpoint_path: Path | str,
    config: dict[str, Any],
    device: torch.device,
    condition: str | None = None,
) -> nn.Module:
    """Load a VTCF model from a checkpoint file."""
    resolved_path = resolve_checkpoint_path(checkpoint_path, condition=condition)
    checkpoint = torch.load(resolved_path, map_location=device)
    model = VTCF(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Loaded checkpoint from %s", resolved_path)
    return model


def _align_predictions_by_video_id(
    reference_ids: list[str],
    candidate_ids: list[str],
    candidate_preds: list[int],
) -> list[int]:
    """Align candidate predictions to the reference video_id order."""
    mapping = {
        str(video_id): int(pred)
        for video_id, pred in zip(candidate_ids, candidate_preds)
    }
    return [mapping[str(video_id)] for video_id in reference_ids]


def _build_text_only_confidence_dict(results: dict[str, Any]) -> dict[str, float]:
    """Build a video_id -> clickbait confidence mapping from evaluation results."""
    predictions = results["predictions"]
    return {
        str(video_id): float(prob)
        for video_id, prob in zip(predictions["video_ids"], predictions["pred_probs"])
    }


def run_ablation_suite(
    config_path: Path | str,
    test_dataloader: DataLoader,
    device: torch.device,
    pad_token_id: int,
) -> dict[str, Any]:
    """
    Evaluate all ablation checkpoints sequentially and compare them statistically.

    Models are loaded one at a time to avoid holding multiple checkpoints on GPU.
    """
    config = load_config(config_path)
    suite_results: dict[str, Any] = {
        "conditions": {},
        "mcnemar": {},
        "text_only_confidence": {},
    }

    fallback_full = LEGACY_CHECKPOINTS[0]
    checkpoint_map = dict(ABLATION_CHECKPOINTS)
    if not checkpoint_map["full"].exists() and fallback_full.exists():
        checkpoint_map["full"] = fallback_full
        logger.warning("Using legacy checkpoint for full model: %s", fallback_full)

    condition_labels = {
        "text_only": "Text Only",
        "vision_only": "Vision Only",
        "full": "Full VTCF",
    }

    for condition, checkpoint_path in checkpoint_map.items():
        if not checkpoint_path.exists():
            logger.warning("Missing checkpoint for %s: %s", condition, checkpoint_path)
            continue

        print(f"\nEvaluating {condition_labels[condition]} from {checkpoint_path.name}")
        model = load_checkpoint_model(checkpoint_path, config, device, condition=condition)
        results = run_evaluation(
            model=model,
            dataloader=test_dataloader,
            device=device,
            ablation=condition,
            pad_token_id=pad_token_id,
        )
        suite_results["conditions"][condition] = results

        if condition == "text_only":
            suite_results["text_only_confidence"] = _build_text_only_confidence_dict(results)

        unload_model(model, device)

    if "full" not in suite_results["conditions"]:
        logger.warning("Full model checkpoint unavailable; skipping McNemar comparisons.")
        _print_ablation_table(suite_results)
        return suite_results

    full_results = suite_results["conditions"]["full"]
    full_true = full_results["predictions"]["true_labels"]
    full_ids = full_results["predictions"]["video_ids"]
    full_preds = full_results["predictions"]["pred_labels"]

    for condition in ("text_only", "vision_only"):
        if condition not in suite_results["conditions"]:
            continue

        condition_results = suite_results["conditions"][condition]
        aligned_preds = _align_predictions_by_video_id(
            full_ids,
            condition_results["predictions"]["video_ids"],
            condition_results["predictions"]["pred_labels"],
        )
        suite_results["mcnemar"][condition] = mcnemar_test(
            model_a_preds=aligned_preds,
            model_b_preds=full_preds,
            true_labels=full_true,
            model_a_name=condition_labels[condition],
            model_b_name="Full VTCF",
        )

    _print_ablation_table(suite_results)
    return suite_results


def _print_ablation_table(suite_results: dict[str, Any]) -> None:
    """Print a formatted comparison table for ablation conditions."""
    rows = [
        ("Text Only", "text_only"),
        ("Vision Only", "vision_only"),
        ("Full VTCF", "full"),
    ]

    print("\n| Condition    | F1 Det | F1 Attr | TDS Corr | p-value vs Full |")
    print("|--------------|--------|---------|----------|-----------------|")

    for label, key in rows:
        if key not in suite_results["conditions"]:
            print(f"| {label:<12} |   ---  |   ---   |   ---    |       ---       |")
            continue

        metrics = suite_results["conditions"][key]
        f1_det = metrics["detection"]["f1_macro"]
        f1_attr = metrics["attribution"]["f1_macro"]
        tds_corr = metrics["tds_correlation"]

        if key == "full":
            p_value_display = "---"
        else:
            mcnemar = suite_results.get("mcnemar", {}).get(key, {})
            p_value_display = f"{mcnemar.get('p_value', float('nan')):.3f}"

        print(
            f"| {label:<12} | {f1_det:6.2f} | {f1_attr:7.2f} | {tds_corr:8.2f} | "
            f"{p_value_display:>15} |"
        )


def evaluate_ambiguous_subset(
    full_model: nn.Module,
    full_dataloader: DataLoader,
    text_only_preds_dict: dict[str, float],
    device: torch.device,
    pad_token_id: int,
    lower_bound: float = AMBIGUOUS_LOWER,
    upper_bound: float = AMBIGUOUS_UPPER,
) -> dict[str, Any]:
    """
    Evaluate full VTCF on samples where text-only confidence was near random.

    Args:
        full_model: Trained full VTCF model (only model loaded in memory).
        full_dataloader: Test DataLoader.
        text_only_preds_dict: Mapping of video_id to text-only clickbait confidence.
        device: Target device.
        pad_token_id: Tokenizer pad token id.
        lower_bound: Lower confidence bound for ambiguity.
        upper_bound: Upper confidence bound for ambiguity.

    Returns:
        Dictionary with subset metrics and sample count.
    """
    ambiguous_ids = {
        video_id
        for video_id, confidence in text_only_preds_dict.items()
        if lower_bound <= confidence <= upper_bound
    }

    if not ambiguous_ids:
        print("Ambiguous subset: 0 samples (0.0% of total)")
        return {
            "num_samples": 0,
            "text_only_f1": float("nan"),
            "vtcf_f1": float("nan"),
            "delta": float("nan"),
            "video_ids": [],
        }

    full_model.eval()
    true_labels: list[int] = []
    text_preds: list[int] = []
    vtcf_preds: list[int] = []
    matched_ids: list[str] = []

    for batch in full_dataloader:
        batch_ids = batch["video_id"]
        keep_indices = [
            index
            for index, video_id in enumerate(batch_ids)
            if str(video_id) in ambiguous_ids
        ]
        if not keep_indices:
            continue

        subset_batch = {
            "input_ids": batch["input_ids"][keep_indices],
            "attention_mask": batch["attention_mask"][keep_indices],
            "pixel_values": batch["pixel_values"][keep_indices],
            "detection_label": batch["detection_label"][keep_indices],
        }

        input_ids = subset_batch["input_ids"].to(device)
        attention_mask = subset_batch["attention_mask"].to(device)
        pixel_values = subset_batch["pixel_values"].to(device)

        outputs = full_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
        predictions = torch.argmax(outputs["detection_logits"], dim=-1).cpu().tolist()
        labels = subset_batch["detection_label"].cpu().tolist()

        for index, video_id in enumerate([batch_ids[i] for i in keep_indices]):
            confidence = text_only_preds_dict[str(video_id)]
            true_labels.append(labels[index])
            text_preds.append(1 if confidence >= 0.5 else 0)
            vtcf_preds.append(predictions[index])
            matched_ids.append(str(video_id))

    num_samples = len(true_labels)
    total = len(text_only_preds_dict)
    percentage = (num_samples / total * 100.0) if total else 0.0

    if num_samples == 0:
        print(f"Ambiguous subset: 0 matched samples ({percentage:.1f}% of total)")
        return {
            "num_samples": 0,
            "text_only_f1": float("nan"),
            "vtcf_f1": float("nan"),
            "delta": float("nan"),
            "video_ids": [],
        }

    y_true = np.array(true_labels)
    text_only_f1 = float(f1_score(y_true, np.array(text_preds), average="macro", zero_division=0))
    vtcf_f1 = float(f1_score(y_true, np.array(vtcf_preds), average="macro", zero_division=0))
    delta = vtcf_f1 - text_only_f1

    print(
        f"On Ambiguous Text Subset ({num_samples} samples): "
        f"Text-Only F1={text_only_f1:.3f}, VTCF F1={vtcf_f1:.3f}, Delta={delta:+.3f}"
    )
    print(f"Ambiguous subset: {num_samples} samples ({percentage:.1f}% of total)")

    return {
        "num_samples": num_samples,
        "text_only_f1": text_only_f1,
        "vtcf_f1": vtcf_f1,
        "delta": delta,
        "video_ids": matched_ids,
    }


def generate_report(all_results_dict: dict[str, Any], save_path: Path | str) -> None:
    """
    Write a markdown evaluation report suitable for a paper results section.

    Args:
        all_results_dict: Aggregated evaluation outputs.
        save_path: Destination markdown path.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# VTCF Evaluation Report",
        "",
        "Automated evaluation summary for the Visual-Temporal Contradiction Framework.",
        "",
    ]

    conditions = all_results_dict.get("ablation_suite", {}).get("conditions", {})
    if conditions:
        lines.extend(
            [
                "## Ablation Comparison",
                "",
                "| Condition | F1 Det | F1 Attr | TDS Corr | p-value vs Full |",
                "|-----------|--------|---------|----------|-----------------|",
            ]
        )
        label_map = {
            "text_only": "Text Only",
            "vision_only": "Vision Only",
            "full": "Full VTCF",
        }
        mcnemar = all_results_dict.get("ablation_suite", {}).get("mcnemar", {})
        for key, label in label_map.items():
            if key not in conditions:
                continue
            metrics = conditions[key]
            p_value = "---" if key == "full" else f"{mcnemar.get(key, {}).get('p_value', float('nan')):.4f}"
            lines.append(
                f"| {label} | {metrics['detection']['f1_macro']:.4f} | "
                f"{metrics['attribution']['f1_macro']:.4f} | "
                f"{metrics['tds_correlation']:.4f} | {p_value} |"
            )
        lines.append("")

    single_eval = all_results_dict.get("single_evaluation")
    if single_eval:
        det = single_eval["detection"]
        attr = single_eval["attribution"]
        lines.extend(
            [
                "## Single-Checkpoint Evaluation",
                "",
                f"- Detection F1 (macro): **{det['f1_macro']:.4f}**",
                f"- Detection F1 (weighted): **{det['f1_weighted']:.4f}**",
                f"- Precision (macro): **{det['precision']:.4f}**",
                f"- Recall (macro): **{det['recall']:.4f}**",
                f"- AUC-ROC: **{det['auc_roc']:.4f}**",
                f"- Accuracy: **{det['accuracy']:.4f}**",
                f"- Attribution F1 (macro): **{attr['f1_macro']:.4f}**",
                f"- Hamming Loss: **{attr['hamming_loss']:.4f}**",
                f"- TDS Correlation: **{single_eval['tds_correlation']:.4f}**",
                "",
                "### Detection Confusion Matrix",
                "",
                "```",
                str(det["confusion_matrix"]),
                "```",
                "",
                "### Attribution F1 Per Class",
                "",
            ]
        )
        for label, score in zip(ATTRIBUTION_LABELS, attr["f1_per_class"]):
            lines.append(f"- {label}: **{score:.4f}**")
        lines.append("")

    ambiguous = all_results_dict.get("ambiguous_subset")
    if ambiguous and ambiguous.get("num_samples", 0) > 0:
        lines.extend(
            [
                "## Ambiguous Text Subset",
                "",
                "Samples where the text-only model confidence was between 0.45 and 0.55.",
                "",
                f"- Subset size: **{ambiguous['num_samples']}**",
                f"- Text-Only F1: **{ambiguous['text_only_f1']:.4f}**",
                f"- VTCF F1: **{ambiguous['vtcf_f1']:.4f}**",
                f"- Delta (VTCF - Text-Only): **{ambiguous['delta']:+.4f}**",
                "",
            ]
        )

    mcnemar = all_results_dict.get("ablation_suite", {}).get("mcnemar", {})
    if mcnemar:
        lines.extend(
            [
                "## McNemar's Test",
                "",
                "McNemar's test assesses whether paired prediction differences between",
                "conditions are statistically significant on the same test set (p < 0.05).",
                "",
            ]
        )
        for condition, stats in mcnemar.items():
            lines.append(
                f"- **{condition} vs Full VTCF**: chi2={stats['chi2']:.4f}, "
                f"p={stats['p_value']:.4f}, significant={stats['significant']}"
            )
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- Higher TDS correlation with clickbait labels supports the bait-and-switch hypothesis.",
            "- A positive ambiguous-subset delta indicates the visual branch adds signal when text is unreliable.",
            "",
        ]
    )

    save_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved evaluation report to {save_path}")


def _video_id_from_concat_dataset(dataset: ConcatDataset, index: int) -> str:
    """Resolve a video_id from a ConcatDataset index."""
    if index < 0 or index >= len(dataset):
        raise IndexError(f"Index {index} out of range for dataset of size {len(dataset)}")

    dataset_index = index
    for subset in dataset.datasets:
        if dataset_index < len(subset):
            return str(subset.dataframe.iloc[dataset_index]["video_id"])
        dataset_index -= len(subset)
    raise IndexError(f"Could not resolve video_id for ConcatDataset index {index}")


def _build_hard_subset_dataloader(
    config_path: Path | str,
    tokenizer: AutoTokenizer,
    hard_ids: set[str],
    split: str,
    batch_size: int,
) -> tuple[DataLoader, set[str]]:
    """Build a DataLoader for hard-subset IDs on test-only or all splits."""
    train_loader, val_loader, test_loader = get_dataloaders(
        config_path=config_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=0,
    )

    if split == "test":
        source_datasets: list[VTCFDataset] = [test_loader.dataset]
    elif split == "all":
        source_datasets = [
            train_loader.dataset,
            val_loader.dataset,
            test_loader.dataset,
        ]
    else:
        raise ValueError(f"Unsupported hard-subset split '{split}'. Expected 'test' or 'all'.")

    combined = ConcatDataset(source_datasets)
    indices = [
        index
        for index in range(len(combined))
        if _video_id_from_concat_dataset(combined, index) in hard_ids
    ]
    matched_ids = {
        _video_id_from_concat_dataset(combined, index) for index in indices
    }

    collator = VTCFCollator(pad_token_id=tokenizer.pad_token_id)
    subset_loader = DataLoader(
        Subset(combined, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    return subset_loader, matched_ids


@torch.no_grad()
def evaluate_hard_subset(
    config_path: Path | str,
    hard_subset_csv: Path | str,
    checkpoint_path: Path | str | None = None,
    device: torch.device | None = None,
    split: str = "test",
) -> dict[str, Any]:
    """
    Evaluate Full VTCF on BanglaBERT failure cases from the hard subset CSV.

    These are videos where the standalone BanglaBERT baseline misclassified the title,
    so text-only F1 on this subset is 0 by construction.

    Args:
        split: ``test`` evaluates only hard-subset IDs in the test split (default).
               ``all`` evaluates every hard-subset ID present in train/val/test combined.
    """
    config = load_config(config_path)
    device = device or get_device()
    hard_subset_csv = Path(hard_subset_csv)
    checkpoint_path = Path(
        checkpoint_path or ABLATION_CHECKPOINTS["full"]
    )

    if split not in {"test", "all"}:
        raise ValueError(f"Unsupported hard-subset split '{split}'. Expected 'test' or 'all'.")

    hard_df = pd.read_csv(hard_subset_csv)
    hard_ids = set(hard_df["video_id"].astype(str).tolist())

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_encoder"])
    batch_size = int(config["training"]["batch_size"])
    subset_loader, matched_ids = _build_hard_subset_dataloader(
        config_path=config_path,
        tokenizer=tokenizer,
        hard_ids=hard_ids,
        split=split,
        batch_size=batch_size,
    )

    if len(subset_loader.dataset) == 0:
        raise RuntimeError(
            f"No hard-subset videos matched split='{split}'. "
            f"CSV contains {len(hard_ids)} ids."
        )

    model = load_checkpoint_model(checkpoint_path, config, device, condition="full")
    results = run_evaluation(
        model=model,
        dataloader=subset_loader,
        device=device,
        ablation="full",
        pad_token_id=tokenizer.pad_token_id,
    )
    unload_model(model, device)

    full_f1 = float(results["detection"]["f1_macro"])
    banglabert_f1 = 0.0
    improvement = full_f1 - banglabert_f1
    sample_count = len(subset_loader.dataset)

    box_unicode = (
        "╔══════════════════════════════════════════╗\n"
        f"║      HARD SUBSET EVALUATION (n={sample_count:02d})      ║\n"
        "╠══════════════════════════════════════════╣\n"
        f"║ BanglaBERT (text-only):  F1 = {banglabert_f1:.4f}   ║\n"
        f"║ Full VTCF:               F1 = {full_f1:.4f}   ║\n"
        f"║ Improvement:             +{improvement:.4f}        ║\n"
        "╚══════════════════════════════════════════╝\n"
        "Note: text-only F1=0 by construction\n"
        "(these are exactly the cases it got wrong)"
    )
    box_ascii = (
        "+==========================================+\n"
        f"|      HARD SUBSET EVALUATION (n={sample_count:02d})      |\n"
        "+==========================================+\n"
        f"| BanglaBERT (text-only):  F1 = {banglabert_f1:.4f}   |\n"
        f"| Full VTCF:               F1 = {full_f1:.4f}   |\n"
        f"| Improvement:             +{improvement:.4f}        |\n"
        "+==========================================+\n"
        "Note: text-only F1=0 by construction\n"
        "(these are exactly the cases it got wrong)"
    )
    box_for_file = box_unicode if os.name != "nt" else box_ascii
    print("\n" + box_for_file)

    missing_ids = sorted(hard_ids - matched_ids)
    if split == "all":
        _, test_matched_ids = _build_hard_subset_dataloader(
            config_path=config_path,
            tokenizer=tokenizer,
            hard_ids=hard_ids,
            split="test",
            batch_size=batch_size,
        )
        n_matched_test = len(test_matched_ids)
    else:
        n_matched_test = sample_count

    if missing_ids:
        print(
            f"Hard subset split='{split}': {sample_count}/{len(hard_ids)} CSV ids evaluated; "
            f"{len(missing_ids)} missing from usable dataset."
        )

    output = {
        "split": split,
        "n_csv": len(hard_ids),
        "n_evaluated": sample_count,
        "n_matched_test": n_matched_test,
        "n_missing_from_dataset": len(missing_ids),
        "missing_video_ids": missing_ids,
        "video_ids": sorted(matched_ids),
        "banglabert_f1": banglabert_f1,
        "full_vtcf_f1": full_f1,
        "improvement": improvement,
        "detection": results["detection"],
    }

    output_path = DEFAULT_OUTPUT_DIR / "hard_subset_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"Saved hard subset results to {output_path}")

    return output


@torch.no_grad()
def generate_storyboard_figures(
    config_path: Path | str,
    hard_subset_csv: Path | str,
    output_dir: Path | str,
    device: torch.device | None = None,
) -> list[Path]:
    """Generate Figure 3a/b/c storyboards for paper examples."""
    config = load_config(config_path)
    device = device or get_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = resolve_project_path(config["data"]["frames_dir"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_encoder"])
    _, _, test_loader = get_dataloaders(
        config_path=config_path,
        tokenizer=tokenizer,
        batch_size=1,
        num_workers=0,
    )

    hard_ids = set(pd.read_csv(hard_subset_csv)["video_id"].astype(str).tolist())
    model = load_checkpoint_model(ABLATION_CHECKPOINTS["full"], config, device, condition="full")

    clickbait_example: dict[str, Any] | None = None
    non_clickbait_example: dict[str, Any] | None = None
    hard_example: dict[str, Any] | None = None

    for batch in test_loader:
        video_id = str(batch["video_id"][0])
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        true_label = int(batch["detection_label"][0].item())

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
        pred_label = int(torch.argmax(outputs["detection_logits"], dim=-1)[0].item())
        tds_tensor = outputs.get("tds_computed")
        if tds_tensor is not None:
            tds_score = float(tds_tensor[0].item())
        else:
            tvm = outputs.get("temporal_visual_matrix")
            tds_score = float(compute_tds(tvm)[0].item()) if tvm is not None else float("nan")

        record = {
            "video_id": video_id,
            "true_label": true_label,
            "pred_label": pred_label,
            "tds_score": tds_score,
            "attention_weights": outputs["attention_weights"][0],
        }

        if (
            clickbait_example is None
            and true_label == 1
            and pred_label == 1
        ):
            clickbait_example = record

        if (
            non_clickbait_example is None
            and true_label == 0
            and pred_label == 0
        ):
            non_clickbait_example = record

        if (
            hard_example is None
            and video_id in hard_ids
            and pred_label == true_label
        ):
            hard_example = record

    unload_model(model, device)

    figure_specs = [
        ("3a", clickbait_example, "clickbait_correct"),
        ("3b", non_clickbait_example, "non_clickbait_correct"),
        ("3c", hard_example, "hard_subset_rescued"),
    ]

    saved_paths: list[Path] = []
    for suffix, example, slug in figure_specs:
        if example is None:
            logger.warning("Could not find storyboard example for Figure %s", suffix)
            continue
        save_path = output_dir / f"figure_{suffix}_{slug}_{example['video_id']}.png"
        visualize_attention_storyboard(
            video_id=example["video_id"],
            frames_dir=frames_dir,
            attention_weights=example["attention_weights"],
            tds_score=example["tds_score"],
            label=example["true_label"],
            prediction=example["pred_label"],
            save_path=save_path,
        )
        saved_paths.append(save_path)
        print(f"Storyboard saved -> {save_path}")

    return saved_paths


def run_paper_assets(
    config_path: Path | str,
    hard_subset_csv: Path | str,
    output_dir: Path | str,
    hard_subset_split: str = "all",
) -> None:
    """Run hard-subset eval, full ablation table, and storyboard figures."""
    device = get_device()
    config = load_config(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hard_results = evaluate_hard_subset(
        config_path=config_path,
        hard_subset_csv=hard_subset_csv,
        device=device,
        split=hard_subset_split,
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_encoder"])
    _, _, test_loader = get_dataloaders(
        config_path=config_path,
        tokenizer=tokenizer,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=0,
    )
    eval_loader = test_loader

    ablation_results = run_ablation_suite(
        config_path=config_path,
        test_dataloader=eval_loader,
        device=device,
        pad_token_id=tokenizer.pad_token_id,
    )

    conditions = ablation_results.get("conditions", {})
    mcnemar = ablation_results.get("mcnemar", {})

    def _metric(condition: str, key: str, default: float = float("nan")) -> float:
        if condition not in conditions:
            return default
        return float(conditions[condition]["detection"][key])

    verified_csv = resolve_project_path(config["data"]["verified_csv"])
    tds_df = run_tds_analysis_from_csv(verified_csv, output_dir=output_dir / "visualizations")
    clickbait_mean = (
        float(tds_df.loc[tds_df["label"] == "clickbait", "tds_score"].mean())
        if not tds_df.empty
        else None
    )
    non_clickbait_mean = (
        float(tds_df.loc[tds_df["label"] == "non_clickbait", "tds_score"].mean())
        if not tds_df.empty
        else None
    )
    from scipy import stats

    if not tds_df.empty:
        cb = tds_df.loc[tds_df["label"] == "clickbait", "tds_score"].astype(float)
        ncb = tds_df.loc[tds_df["label"] == "non_clickbait", "tds_score"].astype(float)
        mann_p = float(stats.mannwhitneyu(cb, ncb, alternative="two-sided").pvalue)
    else:
        mann_p = None

    generate_paper_results_table(
        text_only_f1=_metric("text_only", "f1_macro"),
        vision_only_f1=_metric("vision_only", "f1_macro"),
        full_f1=_metric("full", "f1_macro"),
        hard_subset_f1=float(hard_results["full_vtcf_f1"]),
        text_only_acc=_metric("text_only", "accuracy"),
        vision_only_acc=_metric("vision_only", "accuracy"),
        full_acc=_metric("full", "accuracy"),
        mcnemar_p_full_vs_text=float(mcnemar.get("text_only", {}).get("p_value", float("nan"))),
        mcnemar_p_full_vs_vision=float(
            mcnemar.get("vision_only", {}).get("p_value", float("nan"))
        ),
        clickbait_mean_tds=clickbait_mean,
        non_clickbait_mean_tds=non_clickbait_mean,
        mann_whitney_p=mann_p,
        output_path=output_dir / "paper_results_table.txt",
    )

    generate_storyboard_figures(
        config_path=config_path,
        hard_subset_csv=hard_subset_csv,
        output_dir=output_dir / "visualizations",
        device=device,
    )


def resolve_project_path(path: Path | str) -> Path:
    """Resolve a config-relative path against the project root."""
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate VTCF models and generate reports")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "hard_subset", "paper_assets", "storyboards"],
        default="full",
        help="Evaluation mode: full ablation suite, hard subset, paper assets, or storyboards",
    )
    parser.add_argument(
        "--hard-subset-csv",
        type=Path,
        default=DEFAULT_HARD_SUBSET_CSV,
        help="CSV of BanglaBERT failure cases for hard-subset evaluation",
    )
    parser.add_argument(
        "--hard-subset-split",
        type=str,
        choices=["test", "all"],
        default="test",
        help="Hard-subset scope: test split only (n≈4) or all splits (n≈29)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional single checkpoint to evaluate",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--ablation-mode",
        type=str,
        choices=["text_only", "vision_only", "full"],
        default="full",
        help="Ablation mode for single-checkpoint evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for evaluation outputs",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full VTCF evaluation suite."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args = parse_args()
    config = load_config(args.config)
    device = get_device()
    print(f"Using device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "hard_subset":
        evaluate_hard_subset(
            config_path=args.config,
            hard_subset_csv=args.hard_subset_csv,
            checkpoint_path=args.checkpoint,
            device=device,
            split=args.hard_subset_split,
        )
        return

    if args.mode == "paper_assets":
        run_paper_assets(
            config_path=args.config,
            hard_subset_csv=args.hard_subset_csv,
            output_dir=args.output_dir,
            hard_subset_split=args.hard_subset_split,
        )
        return

    if args.mode == "storyboards":
        generate_storyboard_figures(
            config_path=args.config,
            hard_subset_csv=args.hard_subset_csv,
            output_dir=args.output_dir / "visualizations",
            device=device,
        )
        return

    report_path = args.output_dir / "evaluation_report.md"

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_encoder"])
    train_loader, val_loader, test_loader = get_dataloaders(
        config_path=args.config,
        tokenizer=tokenizer,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=0,
    )
    eval_loader = test_loader if len(test_loader.dataset) > 0 else val_loader
    if len(eval_loader.dataset) == 0:
        raise RuntimeError("No evaluation samples available in test or validation splits.")

    split_name = "test" if len(test_loader.dataset) > 0 else "val"
    print(
        f"Evaluating on {split_name} split: {len(eval_loader.dataset)} samples "
        f"(train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)})"
    )
    if len(eval_loader.dataset) < 10:
        print(
            "WARNING: Very small evaluation set — metrics like F1=1.00 or TDS corr=nan "
            "are expected until full ingestion completes."
        )

    all_results: dict[str, Any] = {}

    ablation_results = run_ablation_suite(
        config_path=args.config,
        test_dataloader=eval_loader,
        device=device,
        pad_token_id=tokenizer.pad_token_id,
    )
    all_results["ablation_suite"] = ablation_results

    if ablation_results.get("text_only_confidence") and "full" in ablation_results.get("conditions", {}):
        full_checkpoint = ABLATION_CHECKPOINTS["full"]
        if not full_checkpoint.exists():
            full_checkpoint = resolve_checkpoint_path("best_model_full.pt", condition="full")

        if full_checkpoint.exists():
            full_model = load_checkpoint_model(full_checkpoint, config, device)
            all_results["ambiguous_subset"] = evaluate_ambiguous_subset(
                full_model=full_model,
                full_dataloader=eval_loader,
                text_only_preds_dict=ablation_results["text_only_confidence"],
                device=device,
                pad_token_id=tokenizer.pad_token_id,
            )
            unload_model(full_model, device)

    if args.checkpoint is not None:
        print(f"\nEvaluating single checkpoint: {args.checkpoint}")
        model = load_checkpoint_model(
            args.checkpoint,
            config,
            device,
            condition=args.ablation_mode,
        )
        all_results["single_evaluation"] = run_evaluation(
            model=model,
            dataloader=eval_loader,
            device=device,
            ablation=args.ablation_mode,
            pad_token_id=tokenizer.pad_token_id,
        )
        unload_model(model, device)

    generate_report(all_results, report_path)


if __name__ == "__main__":
    main()
