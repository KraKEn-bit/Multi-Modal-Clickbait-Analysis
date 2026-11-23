"""Phase 2 PyTorch dataset: title + hook OCR + LLM summary → fusion_network_v2."""

from __future__ import annotations

import json
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

VALID_SPLITS = frozenset({"train", "val", "test"})
DEFAULT_SPLIT_CSV = Path(__file__).resolve().parent / "finding2_split.csv"


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


def build_split_assignments(
    dataframe: pd.DataFrame,
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    label_column: str = "detection_label_int",
) -> pd.DataFrame:
    """Return video_id → split assignments for a filtered training dataframe."""
    if label_column not in dataframe.columns:
        raise KeyError(f"Missing label column '{label_column}' in training dataframe.")

    split_indices = _build_split_indices(
        labels=dataframe[label_column].tolist(),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    rows: list[dict[str, str]] = []
    for split_name, indices in split_indices.items():
        for index in indices:
            rows.append(
                {
                    "video_id": str(dataframe.iloc[index]["video_id"]),
                    "split": split_name,
                }
            )
    return pd.DataFrame(rows)


def _load_persisted_split_csv(split_csv_path: Path) -> pd.DataFrame:
    split_df = pd.read_csv(split_csv_path)
    required = {"video_id", "split"}
    missing = required - set(split_df.columns)
    if missing:
        raise ValueError(f"Split CSV missing columns: {sorted(missing)}")

    split_df = split_df.copy()
    split_df["video_id"] = split_df["video_id"].astype(str)
    split_df["split"] = split_df["split"].astype(str).str.lower()
    invalid = sorted(set(split_df["split"]) - VALID_SPLITS)
    if invalid:
        raise ValueError(f"Split CSV has invalid split values: {invalid}")

    duplicates = split_df["video_id"][split_df["video_id"].duplicated()].tolist()
    if duplicates:
        raise ValueError(
            f"Split CSV has duplicate video_id entries (e.g. {duplicates[:3]})."
        )
    return split_df


def _apply_split_to_dataframe(
    dataframe: pd.DataFrame,
    *,
    split: str,
    split_csv_path: Path | None,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    if split_csv_path is not None and split_csv_path.exists():
        split_df = _load_persisted_split_csv(split_csv_path)
        dataset_ids = set(dataframe["video_id"].astype(str))
        split_ids = set(split_df["video_id"])
        missing_ids = sorted(dataset_ids - split_ids)
        extra_ids = sorted(split_ids - dataset_ids)
        if missing_ids:
            raise ValueError(
                "Persisted split CSV is missing assignments for "
                f"{len(missing_ids)} training-ready video(s) "
                f"(e.g. {missing_ids[:3]}). Regenerate the split file or "
                "remove it to fall back to sklearn splitting."
            )
        if extra_ids:
            logger.warning(
                "Persisted split CSV contains %s video_id(s) not in the "
                "current training frame (ignored).",
                len(extra_ids),
            )
        merged = dataframe.merge(split_df, on="video_id", how="inner")
        subset = merged[merged["split"] == split].drop(columns=["split"])
        return subset.reset_index(drop=True)

    split_indices = _build_split_indices(
        labels=dataframe["detection_label_int"].tolist(),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    return dataframe.iloc[split_indices[split]].reset_index(drop=True)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def prepare_training_dataframe(
    dataframe: pd.DataFrame,
    *,
    transcripts_dir: Path,
    require_gemini: bool = True,
    min_transcript_words: int = 0,
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
        meta_path = transcripts_dir / video_id / "metadata.json"
        transcript_words = 0
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                transcript_words = int(meta.get("transcript_word_count", 0) or 0)
            except (json.JSONDecodeError, TypeError, ValueError):
                transcript_words = 0
        if min_transcript_words > 0 and transcript_words <= min_transcript_words:
            missing.append(video_id)
            continue
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
        min_transcript_words: int = 0,
        split_csv_path: Path | str | None = None,
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
            min_transcript_words=min_transcript_words,
        )
        dataframe["detection_label_int"] = dataframe["human_label"].map(map_detection_label)

        resolved_split_csv = (
            resolve_path(split_csv_path) if split_csv_path is not None else None
        )
        self.dataframe = _apply_split_to_dataframe(
            dataframe,
            split=split,
            split_csv_path=resolved_split_csv,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        source = (
            f"persisted:{resolved_split_csv.name}"
            if resolved_split_csv is not None and resolved_split_csv.exists()
            else f"sklearn(seed={seed})"
        )
        logger.info(
            "SemanticVTCFDataset split=%s samples=%s source=%s",
            split,
            len(self.dataframe),
            source,
        )

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
        min_transcript_words: int = 0,
    ) -> torch.Tensor:
        raw = pd.read_csv(resolve_path(csv_path))
        dataframe = prepare_training_dataframe(
            raw,
            transcripts_dir=resolve_path(transcripts_dir),
            require_gemini=require_gemini,
            min_transcript_words=min_transcript_words,
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
    min_transcript_words: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    config = load_config(config_path)
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_cfg.get("text_encoder"))

    csv_path = config["data"]["finding2_verified_csv"]
    transcripts_dir = config["data"]["transcripts_dir"]
    split_csv_path = config.get("data", {}).get("finding2_split_csv")
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
        "min_transcript_words": min_transcript_words,
        "split_csv_path": split_csv_path,
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
