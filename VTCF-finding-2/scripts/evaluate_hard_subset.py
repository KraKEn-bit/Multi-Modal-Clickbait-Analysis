"""Evaluate Finding-2 ablations on Finding-1's BanglaBERT hard subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.custom_dataset import SemanticCollator, _read_text, prepare_training_dataframe
from scripts._paths import DEFAULT_CONFIG_PATH, load_config, resolve_path
from scripts.evaluate_finding2 import evaluate_finding2_ablation, load_finding2_model
from scripts.extend_dataset import map_detection_label
from scripts.train_finding2 import MIN_TRANSCRIPT_WORDS

REPORT_PATH = PROJECT_ROOT / "outputs" / "hard_subset_evaluation_report.json"


class HardSubsetDataset(Dataset):
    """SemanticVTCF samples for an explicit list of video IDs."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transcripts_dir: Path,
        text_tokenizer,
        *,
        max_title_length: int = 128,
        max_ocr_length: int = 128,
        max_summary_length: int = 256,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.transcripts_dir = transcripts_dir
        self.text_tokenizer = text_tokenizer
        self.max_title_length = max_title_length
        self.max_ocr_length = max_ocr_length
        self.max_summary_length = max_summary_length

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
        }


def _has_complete_finding2_row(
    row: pd.Series,
    transcripts_dir: Path,
    *,
    min_transcript_words: int,
) -> tuple[bool, str]:
    video_id = str(row["video_id"])
    transcript_path = Path(str(row.get("transcript_path", "")))
    if not transcript_path.exists():
        transcript_path = transcripts_dir / video_id / "raw_transcript.txt"
    summary_path = Path(str(row.get("summary_path", "")))
    if not summary_path.exists():
        summary_path = transcripts_dir / video_id / "summary.txt"
    hook_path = transcripts_dir / video_id / "hook_ocr.txt"
    meta_path = transcripts_dir / video_id / "metadata.json"

    if not transcript_path.exists() or not _read_text(transcript_path):
        return False, "missing_transcript"
    if not summary_path.exists() or not _read_text(summary_path):
        return False, "missing_summary"
    if not hook_path.exists():
        return False, "missing_hook_ocr_file"

    transcript_words = 0
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            transcript_words = int(meta.get("transcript_word_count", 0) or 0)
        except (json.JSONDecodeError, TypeError, ValueError):
            transcript_words = 0
    if transcript_words <= min_transcript_words:
        return False, f"transcript_words<={min_transcript_words}"
    return True, "ok"


def build_hard_subset_frame(
    hard_csv: Path,
    verified_csv: Path,
    transcripts_dir: Path,
    *,
    min_transcript_words: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    hard_df = pd.read_csv(hard_csv)
    hard_df["video_id"] = hard_df["video_id"].astype(str)
    hard_ids = set(hard_df["video_id"])

    verified = pd.read_csv(verified_csv)
    verified["video_id"] = verified["video_id"].astype(str)
    training_ready = prepare_training_dataframe(
        verified,
        transcripts_dir=transcripts_dir,
        require_gemini=True,
        min_transcript_words=min_transcript_words,
    )

    in_verified = hard_df.merge(training_ready, on="video_id", how="inner", suffixes=("_hard", ""))
    complete_rows: list[pd.Series] = []
    incomplete: list[dict[str, str]] = []
    for _, row in in_verified.iterrows():
        ok, reason = _has_complete_finding2_row(
            row,
            transcripts_dir,
            min_transcript_words=min_transcript_words,
        )
        if ok:
            complete_rows.append(row)
        else:
            incomplete.append({"video_id": str(row["video_id"]), "reason": reason})

    if complete_rows:
        frame = pd.DataFrame(complete_rows).reset_index(drop=True)
        frame["detection_label_int"] = frame["human_label"].map(map_detection_label)
    else:
        frame = pd.DataFrame()

    meta = {
        "hard_subset_n": len(hard_ids),
        "in_finding2_verified_gemini_words": len(in_verified),
        "complete_finding2_n": len(frame),
        "missing_from_finding2": sorted(hard_ids - set(training_ready["video_id"].astype(str))),
        "in_verified_incomplete_data": incomplete,
        "evaluated_video_ids": sorted(frame["video_id"].astype(str).tolist()) if len(frame) else [],
    }
    return frame, meta


def print_table(rows: list[tuple[str, int, float]]) -> None:
    print("\n=== Hard Subset Comparison ===")
    print(f"{'Model':<34} {'n':>4} {'F1':>8}")
    print("-" * 50)
    for name, n, f1 in rows:
        print(f"{name:<34} {n:>4} {f1:>8.4f}")


def main() -> None:
    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Evaluate Finding-2 on BanglaBERT hard subset")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--hard-csv", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=["title_only", "title_summary", "title_ocr_summary"],
    )
    args = parser.parse_args()

    config = load_config(args.config)
    vtcf_root = resolve_path(config["vtcf_research_root"])
    hard_csv = args.hard_csv or (vtcf_root / "data" / "baseline_banglabert_model" / "hard_subset_video_ids.csv")
    verified_csv = resolve_path(config["data"]["finding2_verified_csv"])
    transcripts_dir = resolve_path(config["data"]["transcripts_dir"])

    frame, subset_meta = build_hard_subset_frame(
        hard_csv,
        verified_csv,
        transcripts_dir,
        min_transcript_words=MIN_TRANSCRIPT_WORDS,
    )

    print("Hard subset cross-reference")
    print(f"  Hard subset total: {subset_meta['hard_subset_n']}")
    print(f"  In finding2_verified (gemini, words>{MIN_TRANSCRIPT_WORDS}): {subset_meta['in_finding2_verified_gemini_words']}")
    print(f"  Complete Finding-2 data (transcript+summary+OCR file): {subset_meta['complete_finding2_n']}")
    if subset_meta["missing_from_finding2"]:
        print(f"  Missing from Finding-2 verified: {subset_meta['missing_from_finding2']}")
    if subset_meta["in_verified_incomplete_data"]:
        print(f"  Verified but incomplete: {subset_meta['in_verified_incomplete_data']}")

    n_eval = len(frame)
    if n_eval == 0:
        print("\nNo hard-subset videos with complete Finding-2 data to evaluate.")
        return

    f1_results: dict[str, dict[str, float]] = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_encoder"])
    pad_token_id = tokenizer.pad_token_id or 0
    model_cfg = config.get("model", {})

    dataset = HardSubsetDataset(
        frame,
        transcripts_dir,
        tokenizer,
        max_title_length=int(model_cfg.get("max_title_length", 128)),
        max_ocr_length=int(model_cfg.get("max_ocr_length", 128)),
        max_summary_length=int(model_cfg.get("max_summary_length", 256)),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=SemanticCollator(pad_token_id=pad_token_id),
    )

    per_ablation: dict[str, Any] = {}
    for ablation in args.ablations:
        print(f"\nEvaluating {ablation} on hard subset (n={n_eval})...")
        model = load_finding2_model(ablation, config, device)
        metrics = evaluate_finding2_ablation(
            model, loader, device, ablation, pad_token_id
        )
        per_ablation[ablation] = metrics
        f1_results[ablation] = {
            "f1_macro": metrics["f1_macro"],
            "accuracy": metrics["accuracy"],
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Finding-1 reference (n=29 on frames dataset)
    f1_ref_path = vtcf_root / "outputs" / "hard_subset_results.json"
    f1_n = 29
    f1_f1 = 1.0
    if f1_ref_path.exists():
        ref = json.loads(f1_ref_path.read_text(encoding="utf-8"))
        f1_n = int(ref.get("n_evaluated", 29))
        f1_f1 = float(ref.get("full_vtcf_f1", 1.0))

    table_rows = [
        ("BanglaBERT standalone", n_eval, 0.0),
        ("Finding-1 Full VTCF (visual)", f1_n, f1_f1),
        ("Finding-2 A (title_only)", n_eval, f1_results["title_only"]["f1_macro"]),
        ("Finding-2 C (title_summary)", n_eval, f1_results["title_summary"]["f1_macro"]),
        ("Finding-2 D (title_ocr_summary)", n_eval, f1_results["title_ocr_summary"]["f1_macro"]),
    ]
    print_table(table_rows)

    report = {
        "hard_subset_csv": str(hard_csv),
        "subset_meta": subset_meta,
        "banglabert_standalone": {"n": n_eval, "f1_macro": 0.0},
        "finding1_full_vtcf_reference": {"n": f1_n, "f1_macro": f1_f1},
        "finding2": {
            key: {
                "n": metrics["n"],
                "f1_macro": metrics["f1_macro"],
                "accuracy": metrics["accuracy"],
                "video_ids": metrics["video_ids"],
                "y_true": metrics["y_true"],
                "y_pred": metrics["y_pred"],
            }
            for key, metrics in per_ablation.items()
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
