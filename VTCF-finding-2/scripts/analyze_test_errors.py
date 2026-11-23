"""Identify per-video test-split errors for Finding-2 ablations A-D."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.custom_dataset import get_dataloaders
from scripts._paths import DEFAULT_CONFIG_PATH, load_config
from scripts.evaluate_finding2 import evaluate_finding2_ablation, load_finding2_model
from scripts.train_finding2 import MIN_TRANSCRIPT_WORDS

LABELS = {0: "non_clickbait", 1: "clickbait"}


def main() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_encoder"])
    pad_token_id = tokenizer.pad_token_id or 0
    _, _, test_loader = get_dataloaders(
        config_path=DEFAULT_CONFIG_PATH,
        tokenizer=tokenizer,
        batch_size=16,
        num_workers=0,
        min_transcript_words=MIN_TRANSCRIPT_WORDS,
    )

    ablations = ["title_only", "title_ocr", "title_summary", "title_ocr_summary"]
    results: dict[str, dict] = {}
    for ablation in ablations:
        model = load_finding2_model(ablation, config, device)
        results[ablation] = evaluate_finding2_ablation(
            model, test_loader, device, ablation, pad_token_id
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    a = results["title_only"]
    a_errors = [
        (vid, yt, yp, prob)
        for vid, yt, yp, prob in zip(a["video_ids"], a["y_true"], a["y_pred"], a["y_prob"])
        if yt != yp
    ]

    print("=== title_only (A) errors on test split ===")
    print(f"count: {len(a_errors)}")
    for vid, yt, yp, prob in a_errors:
        print(f"  {vid}: true={LABELS[yt]} pred={LABELS[yp]} p_clickbait={prob:.4f}")

    print("\n=== Error sets by ablation ===")
    a_err_set = {e[0] for e in a_errors}
    err_by_ab: dict[str, set[str]] = {}
    for ablation in ablations:
        m = results[ablation]
        err_set = {
            vid
            for vid, yt, yp in zip(m["video_ids"], m["y_true"], m["y_pred"])
            if yt != yp
        }
        err_by_ab[ablation] = err_set
        overlap = sorted(err_set & a_err_set) if ablation != "title_only" else sorted(err_set)
        print(f"{ablation}: n_errors={len(err_set)} ids={sorted(err_set) or ['none']}")
        if ablation != "title_only":
            print(f"  overlap_with_A: {overlap or ['none']}")

    bcd = [err_by_ab["title_ocr"], err_by_ab["title_summary"], err_by_ab["title_ocr_summary"]]
    common_bcd = set.intersection(*bcd) if bcd else set()
    print(f"\nB ∩ C ∩ D errors: {sorted(common_bcd) or ['none']}")

    if a_errors:
        print("\n=== Predictions on A-error video(s) ===")
        for vid, yt, yp_a, prob_a in a_errors:
            print(f"video_id={vid} | true={LABELS[yt]} | A pred={LABELS[yp_a]} (p={prob_a:.4f})")
            for ablation in ["title_ocr", "title_summary", "title_ocr_summary"]:
                m = results[ablation]
                idx = m["video_ids"].index(vid)
                yp = m["y_pred"][idx]
                prob = m["y_prob"][idx]
                status = "CORRECT" if yp == yt else "WRONG"
                print(f"  {ablation}: pred={LABELS[yp]} p={prob:.4f} [{status}]")


if __name__ == "__main__":
    main()
