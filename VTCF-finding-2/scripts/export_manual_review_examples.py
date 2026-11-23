"""Export 10 training-ready examples with title_only vs title_summary predictions."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.custom_dataset import get_dataloaders, prepare_training_dataframe
from models.fusion_network_v2 import SemanticVTCF
from scripts._paths import DEFAULT_CONFIG_PATH, load_config, resolve_path
from scripts.extend_dataset import map_detection_label
from scripts.train_finding2 import (
    MIN_TRANSCRIPT_WORDS,
    apply_ablation_batch,
    checkpoint_path,
    forward_batch,
)

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "manual_review_10_examples.txt"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def human_label_display(label: object) -> str:
    text = str(label).strip().lower().replace("-", "_")
    if "non_clickbait" in text or "not_clickbait" in text:
        return "Not Clickbait"
    if "non" in text and "clickbait" in text:
        return "Not Clickbait"
    if "clickbait" in text:
        return "Clickbait"
    return str(label)


def pred_name(pred_int: int) -> str:
    return "Clickbait" if pred_int == 1 else "Not Clickbait"


def load_model(ablation: str, config: dict, device: torch.device) -> SemanticVTCF:
    path = checkpoint_path(ablation)
    model = SemanticVTCF(config).to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def collect_test_predictions(
    config: dict,
    device: torch.device,
    pad_token_id: int,
) -> pd.DataFrame:
    _, _, test_loader = get_dataloaders(
        config_path=DEFAULT_CONFIG_PATH,
        batch_size=16,
        num_workers=0,
        min_transcript_words=MIN_TRANSCRIPT_WORDS,
    )
    model_a = load_model("title_only", config, device)
    model_c = load_model("title_summary", config, device)

    rows: list[dict] = []
    with torch.no_grad():
        for batch in test_loader:
            batch_a = apply_ablation_batch(batch, "title_only", pad_token_id)
            batch_c = apply_ablation_batch(batch, "title_summary", pad_token_id)
            logits_a = forward_batch(model_a, batch_a, device)
            logits_c = forward_batch(model_c, batch_c, device)
            prob_a = torch.softmax(logits_a, dim=-1)[:, 1]
            prob_c = torch.softmax(logits_c, dim=-1)[:, 1]
            pred_a = logits_a.argmax(dim=-1)
            pred_c = logits_c.argmax(dim=-1)

            for index, video_id in enumerate(batch["video_id"]):
                rows.append(
                    {
                        "video_id": str(video_id),
                        "label_int": int(batch["detection_label"][index].item()),
                        "pred_a": int(pred_a[index].item()),
                        "conf_a": float(prob_a[index].item()),
                        "pred_c": int(pred_c[index].item()),
                        "conf_c": float(prob_c[index].item()),
                    }
                )
    result = pd.DataFrame(rows)
    result["agree"] = result["pred_a"] == result["pred_c"]
    return result


def pick_examples(predictions: pd.DataFrame) -> list[str]:
    """Select 5 clickbait + 5 non-clickbait with 2-3 A/C disagreements when available."""
    selected: list[str] = []
    used: set[str] = set()

    def add_from(pool: pd.DataFrame, count: int) -> None:
        for video_id in pool["video_id"]:
            if len(selected) >= count:
                return
            if video_id in used:
                continue
            selected.append(video_id)
            used.add(video_id)

    disagree = predictions[~predictions["agree"]]
    agree = predictions[predictions["agree"]]

    disagree_target = min(3, len(disagree))
    if disagree_target:
        per_label = max(1, disagree_target // 2)
        add_from(disagree[disagree["label_int"] == 1], len(selected) + per_label)
        add_from(disagree[disagree["label_int"] == 0], len(selected) + disagree_target)

    for label_int in (1, 0):
        need = 5 - sum(
            1
            for video_id in selected
            if int(predictions.loc[predictions["video_id"] == video_id, "label_int"].iloc[0])
            == label_int
        )
        if need <= 0:
            continue
        pool = agree[agree["label_int"] == label_int]
        pool = pool[~pool["video_id"].isin(used)]
        add_from(pool, len(selected) + need)

    for label_int in (1, 0):
        need = 5 - sum(
            1
            for video_id in selected
            if int(predictions.loc[predictions["video_id"] == video_id, "label_int"].iloc[0])
            == label_int
        )
        if need <= 0:
            continue
        pool = predictions[predictions["label_int"] == label_int]
        pool = pool[~pool["video_id"].isin(used)]
        add_from(pool, len(selected) + need)

    return selected[:10]


def read_hook_ocr(transcripts_dir: Path, video_id: str) -> str:
    path = transcripts_dir / video_id / "hook_ocr.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def read_summary(row: pd.Series, transcripts_dir: Path, video_id: str) -> str:
    summary_path = Path(str(row.get("summary_path", "")))
    if not summary_path.exists():
        summary_path = transcripts_dir / video_id / "summary.txt"
    if not summary_path.exists():
        return ""
    return summary_path.read_text(encoding="utf-8").strip()


def format_example(
    index: int,
    row: pd.Series,
    pred_row: pd.Series,
    transcripts_dir: Path,
) -> str:
    video_id = str(row["video_id"])
    title = str(row.get("title", "") or "")
    human_label = human_label_display(row.get("human_label", ""))
    hook_ocr = read_hook_ocr(transcripts_dir, video_id)
    summary = read_summary(row, transcripts_dir, video_id)

    lines = [
        f"{'=' * 72}",
        f"Example {index} of 10",
        f"{'=' * 72}",
        "",
        f"video_id: {video_id}",
        "",
        "title (Bangla):",
        title if title else "(empty)",
        "",
        f"human_label: {human_label}",
        "",
        "hook_ocr:",
        hook_ocr if hook_ocr else "(none)",
        "",
        "Gemini summary:",
        summary if summary else "(empty)",
        "",
        (
            "Model A (title_only) prediction: "
            f"{pred_name(int(pred_row['pred_a']))} "
            f"(confidence: {pred_row['conf_a']:.4f})"
        ),
        (
            "Model C (title+summary) prediction: "
            f"{pred_name(int(pred_row['pred_c']))} "
            f"(confidence: {pred_row['conf_c']:.4f})"
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_encoder"])
    pad_token_id = tokenizer.pad_token_id or 0

    csv_path = resolve_path(config["data"]["finding2_verified_csv"])
    transcripts_dir = resolve_path(config["data"]["transcripts_dir"])
    raw = pd.read_csv(csv_path)
    training_df = prepare_training_dataframe(
        raw,
        transcripts_dir=transcripts_dir,
        require_gemini=True,
        min_transcript_words=MIN_TRANSCRIPT_WORDS,
    )
    training_df = training_df.set_index("video_id", drop=False)

    predictions = collect_test_predictions(config, device, pad_token_id)
    selected_ids = pick_examples(predictions)
    pred_by_id = predictions.set_index("video_id")

    disagree_count = int((~predictions["agree"]).sum())
    selected_preds = predictions[predictions["video_id"].isin(selected_ids)]
    selected_disagree = int((~selected_preds["agree"]).sum())

    sections: list[str] = [
        "Finding-2 Manual Review — 10 Examples",
        "Source: finding2_verified.csv (summary_source=gemini, transcript_word_count > 5)",
        "All examples drawn from the frozen test split (finding2_split.csv).",
        "Predictions use checkpoints: Model A = title_only, Model C = title+summary",
        "",
        f"Test-set A vs C disagreements (total available): {disagree_count} of {len(predictions)}",
        f"Included in this file: {selected_disagree} disagree, {len(selected_ids) - selected_disagree} agree",
        "",
    ]

    for index, video_id in enumerate(selected_ids, start=1):
        row = training_df.loc[video_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        pred_row = pred_by_id.loc[video_id]
        sections.append(format_example(index, row, pred_row, transcripts_dir))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(sections), encoding="utf-8")
    print(f"Saved -> {OUTPUT_PATH}")
    print(f"Selected video_ids: {', '.join(selected_ids)}")


if __name__ == "__main__":
    main()
