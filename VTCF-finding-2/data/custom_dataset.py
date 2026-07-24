"""Phase 2 PyTorch dataset: title + hook OCR + LLM summary → fusion_network_v2."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from scripts._paths import DEFAULT_CONFIG_PATH, load_config, resolve_path
from scripts.extend_dataset import map_detection_label

logger = logging.getLogger(__name__)


def _build_split_indices(
    labels: list[int],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[int]]:
    indices = list(range(len(labels)))
    holdout_ratio = val_ratio + test_ratio

    if not indices:
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
        logger.warning("Stratified split failed for n=%s; all samples → train.", len(indices))
        return {"train": indices, "val": [], "test": []}

    if not holdout_indices:
        return {"train": train_indices, "val": [], "test": []}

    relative_test_ratio = test_ratio / holdout_ratio if holdout_ratio > 0 else 0.0
    if relative_test_ratio <= 0 or len(holdout_indices) < 2:
        return {"train": train_indices, "val": holdout_indices, "test": []}

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
        return {"train": train_indices, "val": holdout_indices, "test": []}

    return {"train": train_indices, "val": val_indices, "test": test_indices}


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def prepare_training_dataframe(
    dataframe: pd.DataFrame,
    *,
    transcripts_dir: Path,
    require_gemini: bool = True,
) -> pd.DataFrame:
    """Filter finding2_verified rows to training-ready samples."""
    filtered = dataframe.copy()
    before = len(filtered)

    if "usable_for_training" in filtered.columns:
        usable = filtered["usable_for_training"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
        filtered = filtered[usable].copy()

    if require_gemini and "summary_source" in filtered.columns:
        filtered = filtered[
            filtered["summary_source"].astype(str).str.lower() == "gemini"
        ].copy()

    filtered = filtered[filtered["human_label"].notna()].copy()
    filtered = filtered[filtered["title"].notna()].copy()
    filtered = filtered[filtered["title"].astype(str).str.strip().ne("")].copy()

    missing: list[str] = []
    keep_rows: list[pd.Series] = []
    for _, row in filtered.iterrows():
        video_id = str(row["video_id"])
        summary_path = Path(str(row.get("summary_path", "")))
        if not summary_path.exists():
            summary_path = transcripts_dir / video_id / "summary.txt"
        hook_path = transcripts_dir / video_id / "hook_ocr.txt"
        if not summary_path.exists() or not _read_text(summary_path):
            missing.append(video_id)
            continue
        keep_rows.append(row)

    if missing:
        logger.warning(
            "Dropped %s rows with missing/empty summary (e.g. %s).",
            len(missing),
            missing[:3],
        )

    result = pd.DataFrame(keep_rows).reset_index(drop=True)
    logger.info(
        "Training dataframe: %s/%s rows after filtering.",
        len(result),
        before,
    )
    return result


class SemanticVTCFDataset(Dataset):
    """Text-only dataset for SemanticVTCF (title, hook OCR, summary)."""

    def __init__(
        self,
        csv_path: Path | str,
        transcripts_dir: Path | str,
        text_tokenizer: PreTrainedTokenizerBase,
        *,
        max_title_length: int = 128,
        max_ocr_length: int = 128,
        max_summary_length: int = 256,
        split: str = "train",
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        require_gemini: bool = True,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split '{split}'.")

        self.transcripts_dir = resolve_path(transcripts_dir)
        self.text_tokenizer = text_tokenizer
        self.max_title_length = max_title_length
        self.max_ocr_length = max_ocr_length
        self.max_summary_length = max_summary_length
        self.split = split

        raw = pd.read_csv(resolve_path(csv_path))
        dataframe = prepare_training_dataframe(
            raw,
            transcripts_dir=self.transcripts_dir,
            require_gemini=require_gemini,
        )
        dataframe["detection_label_int"] = dataframe["human_label"].map(map_detection_label)

        split_indices = _build_split_indices(
            labels=dataframe["detection_label_int"].tolist(),
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        self.dataframe = dataframe.iloc[split_indices[split]].reset_index(drop=True)
        logger.info("SemanticVTCFDataset split=%s samples=%s", split, len(self.dataframe))

    def __len__(self) -> int:
        return len(self.dataframe)

    def _tokenize(self, text: str, max_length: int) -> dict[str, torch.Tensor]:
        encoding = self.text_tokenizer(
            text or "",
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0).long(),
            "attention_mask": encoding["attention_mask"].squeeze(0).long(),
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.dataframe.iloc[idx]
        video_id = str(row["video_id"])
        title = str(row.get("title", "") or "")
        hook_ocr = _read_text(self.transcripts_dir / video_id / "hook_ocr.txt")

        summary_path = Path(str(row.get("summary_path", "")))
        if not summary_path.exists():
            summary_path = self.transcripts_dir / video_id / "summary.txt"
        summary = _read_text(summary_path)

        title_enc = self._tokenize(title, self.max_title_length)
        ocr_enc = self._tokenize(hook_ocr, self.max_ocr_length)
        summary_enc = self._tokenize(summary, self.max_summary_length)

        return {
            "title_input_ids": title_enc["input_ids"],
            "title_attention_mask": title_enc["attention_mask"],
            "ocr_input_ids": ocr_enc["input_ids"],
            "ocr_attention_mask": ocr_enc["attention_mask"],
            "summary_input_ids": summary_enc["input_ids"],
            "summary_attention_mask": summary_enc["attention_mask"],
            "detection_label": torch.tensor(
                int(row["detection_label_int"]),
                dtype=torch.long,
            ),
            "video_id": video_id,
            "hook_ocr_usable": bool(row.get("hook_ocr_usable", False)),
        }

    @classmethod
    def get_class_weights(
        cls,
        csv_path: Path | str,
        transcripts_dir: Path | str,
        *,
        require_gemini: bool = True,
    ) -> torch.Tensor:
        raw = pd.read_csv(resolve_path(csv_path))
        dataframe = prepare_training_dataframe(
            raw,
            transcripts_dir=resolve_path(transcripts_dir),
            require_gemini=require_gemini,
        )
        labels = dataframe["human_label"].map(map_detection_label).tolist()
        if not labels:
            return torch.ones(2, dtype=torch.float32)

        label_tensor = torch.tensor(labels, dtype=torch.long)
        class_counts = torch.bincount(label_tensor, minlength=2).float().clamp(min=1.0)
        return len(labels) / (len(class_counts) * class_counts)


class SemanticCollator:
    """Batch collator for three independent text streams."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "title_input_ids": torch.stack([b["title_input_ids"] for b in batch]),
            "title_attention_mask": torch.stack([b["title_attention_mask"] for b in batch]),
            "ocr_input_ids": torch.stack([b["ocr_input_ids"] for b in batch]),
            "ocr_attention_mask": torch.stack([b["ocr_attention_mask"] for b in batch]),
            "summary_input_ids": torch.stack([b["summary_input_ids"] for b in batch]),
            "summary_attention_mask": torch.stack(
                [b["summary_attention_mask"] for b in batch]
            ),
            "detection_label": torch.stack([b["detection_label"] for b in batch]),
            "video_id": [b["video_id"] for b in batch],
        }
        return out


def get_dataloaders(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    tokenizer: PreTrainedTokenizerBase | None = None,
    *,
    batch_size: int | None = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    config = load_config(config_path)
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_cfg.get("text_encoder"))

    csv_path = config["data"]["finding2_verified_csv"]
    transcripts_dir = config["data"]["transcripts_dir"]
    seed = int(train_cfg.get("seed", 42))
    val_ratio = float(train_cfg.get("val_ratio", 0.1))
    test_ratio = float(train_cfg.get("test_ratio", 0.1))
    bs = int(batch_size or train_cfg.get("batch_size", 8))

    common = {
        "csv_path": csv_path,
        "transcripts_dir": transcripts_dir,
        "text_tokenizer": tokenizer,
        "max_title_length": int(model_cfg.get("max_title_length", 128)),
        "max_ocr_length": int(model_cfg.get("max_ocr_length", 128)),
        "max_summary_length": int(model_cfg.get("max_summary_length", 256)),
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
    }

    train_ds = SemanticVTCFDataset(split="train", **common)
    val_ds = SemanticVTCFDataset(split="val", **common)
    test_ds = SemanticVTCFDataset(split="test", **common)
    collator = SemanticCollator(pad_token_id=tokenizer.pad_token_id)

    loader_kwargs = {
        "batch_size": bs,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collator,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from transformers import AutoTokenizer

    cfg = load_config()
    tok = AutoTokenizer.from_pretrained(cfg["model"]["text_encoder"])
    train_loader, val_loader, test_loader = get_dataloaders(tokenizer=tok, num_workers=0)
    print(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")
    weights = SemanticVTCFDataset.get_class_weights(
        cfg["data"]["finding2_verified_csv"],
        cfg["data"]["transcripts_dir"],
    )
    print(f"class_weights={weights.tolist()}")
    batch = next(iter(train_loader))
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {tuple(value.shape)}")
