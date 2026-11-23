"""Evaluate Finding-2 ablations and compare against Finding-1 full VTCF."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import chi2 as chi2_dist
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.custom_dataset import SemanticCollator, SemanticVTCFDataset, get_dataloaders
from models.fusion_network_v2 import SemanticVTCF
from scripts._paths import DEFAULT_CONFIG_PATH, load_config, resolve_path
from scripts.train_finding2 import (
    MIN_TRANSCRIPT_WORDS,
    apply_ablation_batch,
    checkpoint_path,
    forward_batch,
)

FINDING2_ABLATIONS = ("title_only", "title_ocr", "title_summary", "title_ocr_summary")
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_PATH = OUTPUT_DIR / "finding2_evaluation_report.json"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mcnemar_test(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float | bool]:
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b_count = int(np.sum(correct_a & ~correct_b))
    c_count = int(np.sum(~correct_a & correct_b))
    if b_count + c_count == 0:
        chi2_stat, p_value = 0.0, 1.0
    else:
        chi2_stat = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
        p_value = float(chi2_dist.sf(chi2_stat, df=1))
    return {
        "chi2": float(chi2_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "discordant_a_correct": b_count,
        "discordant_b_correct": c_count,
    }


class Finding1CrossEvalDataset(Dataset):
    """Finding-1 VTCF inputs for the exact Finding-2 test split video IDs."""

    def __init__(
        self,
        finding2_test: SemanticVTCFDataset,
        frames_dir: Path,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        k_frames: int = 3,
    ) -> None:
        self.dataframe = finding2_test.dataframe.copy()
        self.frames_dir = frames_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.k_frames = k_frames
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.dataframe)

    def _load_pixels(self, video_id: str) -> torch.Tensor:
        pixels = torch.zeros(self.k_frames, 3, 224, 224, dtype=torch.float32)
        frame_dir = self.frames_dir / video_id
        for index in range(self.k_frames):
            path = frame_dir / f"frame_{index}.png"
            if path.exists():
                image = Image.open(path).convert("RGB")
                pixels[index] = self.transform(image)
        return pixels

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.dataframe.iloc[idx]
        title = str(row.get("title", "") or "")
        encoding = self.tokenizer(
            title,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        video_id = str(row["video_id"])
        return {
            "input_ids": encoding["input_ids"].squeeze(0).long(),
            "attention_mask": encoding["attention_mask"].squeeze(0).long(),
            "pixel_values": self._load_pixels(video_id),
            "detection_label": torch.tensor(int(row["detection_label_int"]), dtype=torch.long),
            "video_id": video_id,
        }


class Finding1Collator:
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
            "detection_label": torch.stack([item["detection_label"] for item in batch]),
            "video_id": [item["video_id"] for item in batch],
        }


@torch.no_grad()
def evaluate_finding2_ablation(
    model: SemanticVTCF,
    loader: DataLoader,
    device: torch.device,
    ablation: str,
    pad_token_id: int,
) -> dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []
    video_ids: list[str] = []

    for batch in loader:
        batch = apply_ablation_batch(batch, ablation, pad_token_id)
        labels = batch["detection_label"].to(device)
        logits = forward_batch(model, batch, device)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        y_prob.extend(probs.cpu().tolist())
        video_ids.extend(batch["video_id"])

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_prob_arr = np.asarray(y_prob)
    try:
        auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    except ValueError:
        auc = float("nan")

    return {
        "n": len(y_true),
        "f1_macro": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "auc": auc,
        "video_ids": video_ids,
        "y_true": y_true_arr.tolist(),
        "y_pred": y_pred_arr.tolist(),
        "y_prob": y_prob_arr.tolist(),
    }


@torch.no_grad()
def evaluate_finding1_full(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []
    video_ids: list[str] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["detection_label"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
        logits = outputs["detection_logits"]
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        y_prob.extend(probs.cpu().tolist())
        video_ids.extend(batch["video_id"])

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_prob_arr = np.asarray(y_prob)
    try:
        auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    except ValueError:
        auc = float("nan")

    return {
        "n": len(y_true),
        "f1_macro": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "auc": auc,
        "video_ids": video_ids,
        "y_true": y_true_arr.tolist(),
        "y_pred": y_pred_arr.tolist(),
        "y_prob": y_prob_arr.tolist(),
    }


def ensemble_metrics(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float]:
    probs = (probs_a + probs_b) / 2.0
    preds = (probs >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y_true, probs))
    except ValueError:
        auc = float("nan")
    return {
        "f1_macro": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "auc": auc,
    }


def load_finding2_model(ablation: str, config: dict[str, Any], device: torch.device) -> SemanticVTCF:
    path = checkpoint_path(ablation)
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    model = SemanticVTCF(config).to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _remap_finding1_vit_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map Finding-1 ViT keys (transformers 4.x) to current ViTModel layout."""
    import re

    layer_re = re.compile(r"^visual_encoder\.vit\.encoder\.layer\.(\d+)\.(.+)$")
    suffix_re = re.compile(r"^(.+)\.(weight|bias)$")
    attention_map = {
        "attention.attention.query": "attention.q_proj",
        "attention.attention.key": "attention.k_proj",
        "attention.attention.value": "attention.v_proj",
        "attention.output.dense": "attention.o_proj",
        "intermediate.dense": "mlp.fc1",
        "output.dense": "mlp.fc2",
        "layernorm_before": "layernorm_before",
        "layernorm_after": "layernorm_after",
    }

    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        match = layer_re.match(key)
        if not match:
            remapped[key] = value
            continue
        layer_idx, rest = match.groups()
        suffix_match = suffix_re.match(rest)
        if suffix_match:
            base, suffix = suffix_match.groups()
            new_base = attention_map.get(base)
            if new_base is None:
                remapped[key] = value
                continue
            new_key = f"visual_encoder.vit.layers.{layer_idx}.{new_base}.{suffix}"
        else:
            new_base = attention_map.get(rest)
            if new_base is None:
                remapped[key] = value
                continue
            new_key = f"visual_encoder.vit.layers.{layer_idx}.{new_base}"
        remapped[new_key] = value
    return remapped


def _load_finding1_fusion_module(vtcf_research_root: Path):
    module_path = vtcf_research_root / "models" / "fusion_network.py"
    spec = importlib.util.spec_from_file_location("vtcf_f1_fusion_network", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Finding-1 fusion network from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_finding1_model(vtcf_research_root: Path, device: torch.device) -> nn.Module:
    fusion_module = _load_finding1_fusion_module(vtcf_research_root)
    f1_config = fusion_module.load_config(vtcf_research_root / "config.yaml")
    model = fusion_module.VTCF(f1_config).to(device)
    ckpt_path = vtcf_research_root / "outputs" / "checkpoints" / "best_model_full.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing Finding-1 checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = _remap_finding1_vit_state_dict(checkpoint["model_state_dict"])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    vit_missing = [key for key in missing if key.startswith("visual_encoder.vit.")]
    fusion_missing = [key for key in missing if not key.startswith("visual_encoder.vit.")]
    if fusion_missing:
        raise RuntimeError(f"Missing non-ViT Finding-1 weights: {fusion_missing[:5]}")
    if vit_missing:
        raise RuntimeError(
            "Finding-1 ViT weights did not load after remapping; "
            f"sample missing keys: {vit_missing[:5]}"
        )
    if unexpected:
        print(f"Warning: ignored unexpected Finding-1 checkpoint keys: {len(unexpected)}")
    model.eval()
    return model


def print_results_table(results: dict[str, dict[str, Any]]) -> None:
    print("\n=== Finding-2 Test Split Results ===")
    print(f"{'Model':<28} {'N':>4} {'F1':>7} {'Acc':>7} {'AUC':>7}")
    print("-" * 58)
    for name, metrics in results.items():
        print(
            f"{name:<28} {metrics['n']:>4} "
            f"{metrics['f1_macro']:>7.4f} "
            f"{metrics['accuracy']:>7.4f} "
            f"{metrics['auc']:>7.4f}"
        )


def main() -> None:
    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Evaluate Finding-2 ablations")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--ablations",
        nargs="+",
        choices=FINDING2_ABLATIONS,
        default=None,
        help="Evaluate only these Finding-2 ablations (default: all)",
    )
    parser.add_argument(
        "--skip-finding1",
        action="store_true",
        help="Skip Finding-1 cross-eval and ensemble",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_encoder"])
    pad_token_id = tokenizer.pad_token_id or 0
    _, _, test_loader = get_dataloaders(
        config_path=args.config,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=0,
        min_transcript_words=MIN_TRANSCRIPT_WORDS,
    )
    test_ds: SemanticVTCFDataset = test_loader.dataset
    collator = SemanticCollator(pad_token_id=pad_token_id)
    print(f"Test split: n={len(test_ds)}")

    results: dict[str, dict[str, Any]] = {}
    raw_preds: dict[str, dict[str, Any]] = {}
    ablations = tuple(args.ablations or FINDING2_ABLATIONS)

    for ablation in ablations:
        print(f"\nEvaluating Finding-2 ablation: {ablation}")
        model = load_finding2_model(ablation, config, device)
        metrics = evaluate_finding2_ablation(
            model, test_loader, device, ablation, pad_token_id
        )
        results[f"finding2_{ablation}"] = metrics
        raw_preds[ablation] = metrics
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    missing_frames = 0
    mcnemar: dict[str, dict[str, float | bool]] = {}
    if not args.skip_finding1:
        vtcf_research_root = resolve_path(config["vtcf_research_root"])
        frames_dir = resolve_path(config["data"]["frames_dir"])
        if not frames_dir.is_absolute():
            frames_dir = (PROJECT_ROOT / frames_dir).resolve()
        if not frames_dir.exists():
            frames_dir = (vtcf_research_root / "data" / "extracted_frames").resolve()

        print("\nEvaluating Finding-1 full VTCF on the same test split...")
        f1_cfg = load_config(vtcf_research_root / "config.yaml")
        f1_tokenizer = AutoTokenizer.from_pretrained(f1_cfg["model"]["text_encoder"])
        f1_dataset = Finding1CrossEvalDataset(
            test_ds,
            frames_dir=frames_dir,
            tokenizer=f1_tokenizer,
            max_length=int(f1_cfg["model"].get("max_title_length", 128)),
            k_frames=int(f1_cfg["model"].get("K_frames", 3)),
        )
        missing_frames = sum(
            1
            for video_id in f1_dataset.dataframe["video_id"].astype(str)
            if not (frames_dir / video_id / "frame_0.png").exists()
        )
        print(
            f"Finding-1 frame coverage on test split: "
            f"{len(f1_dataset) - missing_frames}/{len(f1_dataset)}"
        )

        f1_loader = DataLoader(
            f1_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=Finding1Collator(),
        )
        f1_model = load_finding1_model(vtcf_research_root, device)
        f1_metrics = evaluate_finding1_full(f1_model, f1_loader, device)
        results["finding1_full_vtcf"] = f1_metrics
        del f1_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if "title_ocr_summary" in raw_preds:
            d_probs = np.asarray(raw_preds["title_ocr_summary"]["y_prob"])
            f1_probs = np.asarray(f1_metrics["y_prob"])
            y_true = np.asarray(f1_metrics["y_true"])
            ensemble = ensemble_metrics(d_probs, f1_probs, y_true)
            results["ensemble_D_plus_finding1"] = {"n": len(y_true), **ensemble}

            print("\n=== McNemar pairwise tests (test split) ===")
            name_map = {
                "title_only": "A title-only",
                "title_ocr": "B title+ocr",
                "title_summary": "C title+summary",
                "title_ocr_summary": "D full",
                "finding1": "Finding-1 full",
                "ensemble": "Ensemble D+F1",
            }

            pred_store: dict[str, np.ndarray] = {
                key: np.asarray(raw_preds[key]["y_pred"])
                for key in raw_preds
            }
            pred_store["finding1"] = np.asarray(f1_metrics["y_pred"])
            pred_store["ensemble"] = ((d_probs + f1_probs) / 2.0 >= 0.5).astype(int)

            pairs = [
                ("title_only", "title_ocr"),
                ("title_only", "title_summary"),
                ("title_only", "title_ocr_summary"),
                ("title_ocr", "title_summary"),
                ("title_ocr", "title_ocr_summary"),
                ("title_summary", "title_ocr_summary"),
                ("title_ocr_summary", "finding1"),
                ("title_ocr_summary", "ensemble"),
                ("finding1", "ensemble"),
            ]
            for a, b in pairs:
                if a not in pred_store or b not in pred_store:
                    continue
                stats = mcnemar_test(pred_store[a], pred_store[b], y_true)
                key = f"{a}_vs_{b}"
                mcnemar[key] = stats
                print(
                    f"{name_map.get(a, a)} vs {name_map.get(b, b)}: "
                    f"p={stats['p_value']:.4f} significant={stats['significant']}"
                )

    print_results_table(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "test_n": len(test_ds),
        "results": {
            key: {k: v for k, v in value.items() if k not in {"video_ids", "y_true", "y_pred", "y_prob"}}
            for key, value in results.items()
        },
        "mcnemar": mcnemar,
        "missing_finding1_frames": missing_frames,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
