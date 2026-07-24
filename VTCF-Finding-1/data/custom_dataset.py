"""PyTorch Dataset and DataLoader utilities for VTCF multimodal training."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import PreTrainedTokenizerBase

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

logger = logging.getLogger(__name__)

ATTRIBUTION_LABELS = [
    "exaggeration",
    "curiosity_gap",
    "emotional_trigger",
    "misleading",
]


def load_config(config_path: Path | str) -> dict[str, Any]:
    """Load YAML configuration from disk."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_project_path(path: Path | str) -> Path:
    """Resolve a config-relative path against the project root."""
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def map_detection_label(label: Any) -> int:
    """Map raw label strings to binary detection targets: 1=clickbait, 0=non-clickbait."""
    if pd.isna(label):
        raise ValueError("Cannot map a null detection label.")

    text = str(label).strip().lower().replace("-", "_")
    if "non_clickbait" in text or "not_clickbait" in text:
        return 0
    if "non" in text and "clickbait" in text:
        return 0
    if "not" in text and "clickbait" in text:
        return 0
    if text == "clickbait" or text.endswith("_clickbait") or text == "1":
        return 1
    if "clickbait" in text:
        return 1
    return 0


def encode_attribution_label(tactic_label: Any, num_classes: int = 4) -> torch.Tensor:
    """Encode tactic attribution as a multi-hot vector."""
    vector = torch.zeros(num_classes, dtype=torch.float32)
    if pd.isna(tactic_label):
        return vector

    raw = str(tactic_label).strip()
    if not raw:
        return vector

    tokens = [token.strip().lower().replace("-", "_") for token in raw.split(",")]
    for token in tokens:
        for index, class_name in enumerate(ATTRIBUTION_LABELS[:num_classes]):
            if token == class_name or class_name in token:
                vector[index] = 1.0
    return vector


def default_image_transform() -> transforms.Compose:
    """Return the default ViT-compatible image preprocessing pipeline."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _build_split_indices(
    labels: list[int],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[int]]:
    """Create reproducible stratified train/val/test index splits."""
    indices = list(range(len(labels)))
    holdout_ratio = val_ratio + test_ratio

    if len(indices) == 0:
        return {"train": [], "val": [], "test": []}

    if holdout_ratio <= 0 or len(indices) < 3:
        return {"train": indices, "val": [], "test": []}

    stratify_labels = labels if len(set(labels)) > 1 else None

    try:
        train_indices, holdout_indices = train_test_split(
            indices,
            test_size=holdout_ratio,
            random_state=seed,
            stratify=stratify_labels,
        )
    except ValueError:
        logger.warning(
            "Stratified train/holdout split failed for n=%s; assigning all samples to train.",
            len(indices),
        )
        return {"train": indices, "val": [], "test": []}

    if not holdout_indices:
        return {"train": train_indices, "val": [], "test": []}

    relative_test_ratio = test_ratio / holdout_ratio if holdout_ratio > 0 else 0.0
    if relative_test_ratio <= 0 or len(holdout_indices) < 2:
        return {
            "train": train_indices,
            "val": holdout_indices,
            "test": [],
        }

    holdout_labels = [labels[index] for index in holdout_indices]
    holdout_stratify = holdout_labels if len(set(holdout_labels)) > 1 else None

    try:
        val_indices, test_indices = train_test_split(
            holdout_indices,
            test_size=relative_test_ratio,
            random_state=seed,
            stratify=holdout_stratify,
        )
    except ValueError:
        logger.warning(
            "Stratified val/test split failed for n=%s holdout; using val-only holdout.",
            len(holdout_indices),
        )
        return {
            "train": train_indices,
            "val": holdout_indices,
            "test": [],
        }

    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


def resolve_frame_dir(row: pd.Series, frames_dir: Path) -> Path:
    """Resolve the directory containing extracted frames for one CSV row."""
    video_id = str(row["video_id"])
    default_dir = frames_dir / video_id

    frame_dir = row.get("frame_dir")
    if isinstance(frame_dir, str) and frame_dir.strip():
        candidate = resolve_project_path(frame_dir)
        if (candidate / "frame_0.png").exists():
            return candidate

    return default_dir


def all_frames_exist(row: pd.Series, frames_dir: Path, k: int) -> bool:
    """Return True when all K frame PNG files exist on disk for a row."""
    frame_dir = resolve_frame_dir(row, frames_dir)
    return all(
        os.path.exists(frame_dir / f"frame_{frame_index}.png")
        for frame_index in range(k)
    )


def prepare_training_dataframe(
    dataframe: pd.DataFrame,
    frames_dir: Path | str,
    k: int,
) -> pd.DataFrame:
    """Filter CSV rows to live samples with valid labels and complete frame sets."""
    resolved_frames_dir = resolve_project_path(frames_dir)
    filtered = dataframe.copy()

    if "audit_status" in filtered.columns:
        before = len(filtered)
        filtered = filtered[
            filtered["audit_status"].astype(str).str.lower() == "live"
        ].copy()
        logger.info(
            "Retained %s/%s live rows after audit_status filtering.",
            len(filtered),
            before,
        )

    before = len(filtered)
    filtered = filtered[filtered["title"].notna()].copy()
    filtered = filtered[filtered["title"].astype(str).str.strip().ne("")].copy()
    filtered = filtered[filtered["label"].notna()].copy()
    logger.info(
        "Retained %s/%s rows after dropping null or empty title/label.",
        len(filtered),
        before,
    )

    before = len(filtered)
    frame_mask = filtered.apply(
        lambda row: all_frames_exist(row, resolved_frames_dir, k),
        axis=1,
    )
    filtered = filtered[frame_mask].copy()
    logger.info(
        "Retained %s/%s rows with all %s frame files present on disk.",
        len(filtered),
        before,
        k,
    )

    return filtered.sort_values("video_id").reset_index(drop=True)


class VTCFDataset(Dataset):
    """Multimodal dataset pairing Bangla headlines with K independent video frames."""

    def __init__(
        self,
        csv_path: Path | str,
        frames_dir: Path | str,
        text_tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        K: int = 3,
        transform: Callable | None = None,
        split: str = "train",
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.csv_path = resolve_project_path(csv_path)
        self.frames_dir = resolve_project_path(frames_dir)
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length
        self.K = K
        self.transform = transform or default_image_transform()
        self.split = split
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split '{split}'. Expected train, val, or test.")

        raw_dataframe = pd.read_csv(self.csv_path)
        dataframe = prepare_training_dataframe(
            raw_dataframe,
            frames_dir=self.frames_dir,
            k=self.K,
        )
        dataframe["detection_label_int"] = dataframe["label"].map(map_detection_label)

        split_indices = _build_split_indices(
            labels=dataframe["detection_label_int"].tolist(),
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        selected_indices = split_indices[split]
        self.dataframe = dataframe.iloc[selected_indices].reset_index(drop=True)

        logger.info(
            "Initialized VTCFDataset split=%s with %s samples.",
            split,
            len(self.dataframe),
        )

    def __len__(self) -> int:
        """Return the number of samples in the current split."""
        return len(self.dataframe)

    def _load_frame_tensor(self, frame_path: Path) -> torch.Tensor:
        """Load and transform a single frame, returning shape (3, 224, 224)."""
        image = Image.open(frame_path).convert("RGB")
        return self.transform(image)

    def _load_pixel_values(self, row: pd.Series) -> torch.Tensor:
        """Load K independent frame tensors with shape (K, 3, 224, 224)."""
        frame_dir = resolve_frame_dir(row, self.frames_dir)
        video_id = str(row["video_id"])
        pixel_values = torch.zeros(self.K, 3, 224, 224, dtype=torch.float32)

        for frame_index in range(self.K):
            frame_path = frame_dir / f"frame_{frame_index}.png"
            if frame_path.exists():
                pixel_values[frame_index] = self._load_frame_tensor(frame_path)
            else:
                logger.warning(
                    "Missing frame for video_id=%s at %s; using zero tensor.",
                    video_id,
                    frame_path,
                )

        return pixel_values

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one multimodal training example."""
        row = self.dataframe.iloc[idx]
        title = row.get("title", "")
        if pd.isna(title):
            title = ""

        encoding = self.text_tokenizer(
            str(title),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        tds_value = row.get("tds_score", 0.0)
        if pd.isna(tds_value):
            tds_value = 0.0

        return {
            "input_ids": encoding["input_ids"].squeeze(0).long(),
            "attention_mask": encoding["attention_mask"].squeeze(0).long(),
            "pixel_values": self._load_pixel_values(row),
            "detection_label": torch.tensor(
                map_detection_label(row.get("label")),
                dtype=torch.long,
            ),
            "attribution_label": encode_attribution_label(row.get("tactic_label")),
            "video_id": str(row["video_id"]),
            "tds_score": torch.tensor(float(tds_value), dtype=torch.float32),
        }

    @classmethod
    def get_class_weights(
        cls,
        csv_path: Path | str,
        frames_dir: Path | str | None = None,
        k: int = 3,
    ) -> torch.Tensor:
        """Return inverse-frequency class weights for the detection head."""
        csv_path = resolve_project_path(csv_path)
        if frames_dir is None:
            config = load_config(DEFAULT_CONFIG_PATH)
            frames_dir = resolve_project_path(config["data"]["frames_dir"])
            k = int(config["model"]["K_frames"])

        raw_dataframe = pd.read_csv(csv_path)
        dataframe = prepare_training_dataframe(raw_dataframe, frames_dir=frames_dir, k=k)
        labels = dataframe["label"].map(map_detection_label).tolist()
        if not labels:
            return torch.ones(2, dtype=torch.float32)

        label_tensor = torch.tensor(labels, dtype=torch.long)
        class_counts = torch.bincount(label_tensor, minlength=2).float()
        class_counts = torch.clamp(class_counts, min=1.0)
        weights = len(labels) / (len(class_counts) * class_counts)
        return weights


class VTCFCollator:
    """Batch collator with dynamic padding for token sequences."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a list of samples into a batched dictionary."""
        max_sequence_length = max(item["input_ids"].shape[0] for item in batch)

        input_ids = []
        attention_masks = []
        pixel_values = []
        detection_labels = []
        attribution_labels = []
        video_ids = []
        tds_scores = []

        for item in batch:
            sequence_length = item["input_ids"].shape[0]
            pad_length = max_sequence_length - sequence_length

            padded_input_ids = torch.cat(
                [
                    item["input_ids"],
                    torch.full((pad_length,), self.pad_token_id, dtype=torch.long),
                ]
            )
            padded_attention_mask = torch.cat(
                [
                    item["attention_mask"],
                    torch.zeros(pad_length, dtype=torch.long),
                ]
            )

            input_ids.append(padded_input_ids)
            attention_masks.append(padded_attention_mask)
            pixel_values.append(item["pixel_values"])
            detection_labels.append(item["detection_label"])
            attribution_labels.append(item["attribution_label"])
            video_ids.append(item["video_id"])
            tds_scores.append(item["tds_score"])

        return {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_masks, dim=0),
            "pixel_values": torch.stack(pixel_values, dim=0),
            "detection_label": torch.stack(detection_labels, dim=0),
            "attribution_label": torch.stack(attribution_labels, dim=0),
            "video_id": video_ids,
            "tds_score": torch.stack(tds_scores, dim=0),
        }


def get_dataloaders(
    config_path: Path | str,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 16,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders from config settings."""
    config = load_config(config_path)

    csv_path = resolve_project_path(config["data"]["verified_csv"])
    frames_dir = resolve_project_path(config["data"]["frames_dir"])
    frame_size = int(config["data"]["frame_size"])
    k_frames = int(config["model"]["K_frames"])
    seed = int(config["training"]["seed"])

    if frame_size != 224:
        logger.warning(
            "Config frame_size=%s differs from ViT default 224; dataset still emits 224x224 tensors.",
            frame_size,
        )

    common_kwargs = {
        "csv_path": csv_path,
        "frames_dir": frames_dir,
        "text_tokenizer": tokenizer,
        "K": k_frames,
        "seed": seed,
    }

    train_dataset = VTCFDataset(split="train", **common_kwargs)
    val_dataset = VTCFDataset(split="val", **common_kwargs)
    test_dataset = VTCFDataset(split="test", **common_kwargs)

    collator = VTCFCollator(pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )

    return train_loader, val_loader, test_loader


def _print_batch_shapes(batch: dict[str, Any]) -> None:
    """Print tensor shapes for a collated batch."""
    print("\n=== VTCF DataLoader Sanity Check ===")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key:20s} shape={tuple(value.shape)} dtype={value.dtype}")
        elif isinstance(value, list):
            print(f"{key:20s} list(len={len(value)}) sample={value[:2]}")
        else:
            print(f"{key:20s} type={type(value).__name__}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    from transformers import AutoTokenizer

    config = load_config(DEFAULT_CONFIG_PATH)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_encoder"])

    train_loader, val_loader, test_loader = get_dataloaders(
        config_path=DEFAULT_CONFIG_PATH,
        tokenizer=tokenizer,
        batch_size=2,
        num_workers=0,
    )

    print(f"train samples: {len(train_loader.dataset)}")
    print(f"val samples:   {len(val_loader.dataset)}")
    print(f"test samples:  {len(test_loader.dataset)}")

    class_weights = VTCFDataset.get_class_weights(config["data"]["verified_csv"])
    print(f"class weights: {class_weights.tolist()}")

    batch = next(iter(train_loader))
    _print_batch_shapes(batch)
