"""
Recompute semantic_divergence using quality-gated hook promise text.

Reads existing raw_transcript.txt, summary.txt, and hook_ocr.txt — does not
re-run ASR or LLM summarization.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._paths import load_config, resolve_path
from scripts.phase0_spike import print_go_no_go_box, print_spot_check_pairs, summarize_rows
from scripts.semantic_features import (
    is_usable_hook_ocr,
    resolve_promise_text,
    semantic_divergence,
)

logger = logging.getLogger(__name__)

SPIKE_SUMMARY_FIELDS = [
    "video_id",
    "title",
    "label",
    "speech_coverage_percent",
    "transcript_word_count",
    "transcript_is_empty",
    "ocr_text_detected",
    "ocr_char_count",
    "hook_ocr_usable",
    "hook_promise_source",
    "semantic_divergence",
    "summary_source",
    "processing_time_seconds",
    "whisper_confidence",
    "transcript",
    "summary",
    "ocr_text",
]


def setup_logging() -> None:
    log_path = PROJECT_ROOT / "outputs" / "logs" / "recompute_divergence.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> None:
    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    setup_logging()
    config = load_config()
    spike_csv = resolve_path(config["data"]["spike_csv"])
    spike_transcripts_dir = resolve_path(config["data"]["spike_transcripts_dir"])
    bert_model = str(config.get("model", {}).get("text_encoder", "sagorsarker/bangla-bert-base"))

    sample_df = pd.read_csv(spike_csv)
    rows: list[dict] = []
    title_only = 0
    title_plus_ocr = 0

    for _, record in sample_df.iterrows():
        video_id = str(record["video_id"])
        title = str(record.get("title", ""))
        human_label = str(record.get("label", ""))
        out_dir = spike_transcripts_dir / video_id

        raw_transcript = (
            (out_dir / "raw_transcript.txt").read_text(encoding="utf-8").strip()
            if (out_dir / "raw_transcript.txt").exists()
            else ""
        )
        summary = (
            (out_dir / "summary.txt").read_text(encoding="utf-8").strip()
            if (out_dir / "summary.txt").exists()
            else ""
        )
        hook_ocr = (
            (out_dir / "hook_ocr.txt").read_text(encoding="utf-8").strip()
            if (out_dir / "hook_ocr.txt").exists()
            else ""
        )

        meta_path = out_dir / "metadata.json"
        metadata: dict = {}
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        promise_text, hook_promise_source = resolve_promise_text(title, hook_ocr)
        divergence = semantic_divergence(promise_text, summary, model_name=bert_model)
        hook_ocr_usable = is_usable_hook_ocr(hook_ocr)

        if hook_promise_source == "title+ocr":
            title_plus_ocr += 1
        else:
            title_only += 1

        metadata.update(
            {
                "semantic_divergence": divergence,
                "hook_ocr_usable": hook_ocr_usable,
                "hook_promise_source": hook_promise_source,
            }
        )
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        row = {
            "video_id": video_id,
            "title": title,
            "label": human_label,
            "speech_coverage_percent": metadata.get("speech_coverage_percent", 0.0),
            "transcript_word_count": len(raw_transcript.split()) if raw_transcript else 0,
            "transcript_is_empty": not raw_transcript.strip(),
            "ocr_text_detected": metadata.get("ocr_text_detected", bool(hook_ocr)),
            "ocr_char_count": len(hook_ocr),
            "hook_ocr_usable": hook_ocr_usable,
            "hook_promise_source": hook_promise_source,
            "semantic_divergence": divergence,
            "summary_source": metadata.get("summary_source", ""),
            "processing_time_seconds": metadata.get("processing_time_seconds", 0.0),
            "whisper_confidence": metadata.get("whisper_confidence"),
            "transcript": raw_transcript,
            "summary": summary,
            "ocr_text": hook_ocr,
        }
        rows.append(row)
        logger.info(
            "DONE %s promise=%s div=%s",
            video_id,
            hook_promise_source,
            f"{divergence:.3f}" if divergence is not None else "N/A",
        )

    summary = summarize_rows(rows, target_total=len(sample_df), config=config)
    summary_csv = PROJECT_ROOT / "outputs" / "spike_results" / "spike_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPIKE_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "Promise sources: title=%s title+ocr=%s (raw hook_ocr.txt unchanged)",
        title_only,
        title_plus_ocr,
    )
    print_spot_check_pairs(rows)
    print_go_no_go_box(summary)
    logger.info("Updated %s | Decision: %s", summary_csv, summary["decision"])


if __name__ == "__main__":
    main()
