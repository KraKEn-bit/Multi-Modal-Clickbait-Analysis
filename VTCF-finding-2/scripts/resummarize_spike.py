"""
Re-summarize spike transcripts, recompute semantic divergence, refresh GO/NO-GO.

Reads data/spike_transcripts/*/raw_transcript.txt and hook_ocr.txt.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._env import load_dotenv
from scripts._paths import load_config, resolve_path
from scripts.phase0_spike import (
    GO_THRESHOLD_USABLE_RATE,
    is_usable_transcript,
    print_go_no_go_box,
    print_spot_check_pairs,
    summarize_rows,
)
from scripts.semantic_features import (
    is_usable_hook_ocr,
    resolve_promise_text,
    semantic_divergence,
    summarize_transcript,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-summarize spike transcripts")
    parser.add_argument(
        "--retry-fallback-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip videos that already have summary_source=gemini (default: on)",
    )
    return parser.parse_args()


def should_resummarize(metadata: dict, *, retry_fallback_only: bool) -> bool:
    if not retry_fallback_only:
        return True
    source = str(metadata.get("summary_source", "")).strip().lower()
    return source != "gemini"


def setup_logging() -> None:
    log_path = PROJECT_ROOT / "outputs" / "logs" / "resummarize_spike.log"
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
    args = parse_args()
    load_dotenv()
    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    setup_logging()
    config = load_config()
    spike_csv = resolve_path(config["data"]["spike_csv"])
    spike_transcripts_dir = resolve_path(config["data"]["spike_transcripts_dir"])
    llm_cfg = config.get("llm", {})
    bert_model = str(config.get("model", {}).get("text_encoder", "sagorsarker/bangla-bert-base"))
    logger.info(
        "Gemini models: primary=%s fallback=%s retry_fallback_only=%s",
        llm_cfg.get("gemini_model"),
        llm_cfg.get("gemini_fallback_model"),
        args.retry_fallback_only,
    )

    sample_df = pd.read_csv(spike_csv)
    rows: list[dict] = []
    resummarized = 0
    skipped = 0

    for _, record in sample_df.iterrows():
        video_id = str(record["video_id"])
        title = str(record.get("title", ""))
        human_label = str(record.get("label", ""))
        out_dir = spike_transcripts_dir / video_id

        raw_path = out_dir / "raw_transcript.txt"
        ocr_path = out_dir / "hook_ocr.txt"
        raw_transcript = raw_path.read_text(encoding="utf-8").strip() if raw_path.exists() else ""
        hook_ocr = ocr_path.read_text(encoding="utf-8").strip() if ocr_path.exists() else ""

        meta_path = out_dir / "metadata.json"
        metadata = {}
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        summary_path = out_dir / "summary.txt"
        if should_resummarize(metadata, retry_fallback_only=args.retry_fallback_only):
            summary, summary_source, llm_model_used = summarize_transcript(raw_transcript, llm_cfg)
            resummarized += 1
        else:
            summary = (
                summary_path.read_text(encoding="utf-8").strip() if summary_path.exists() else ""
            )
            summary_source = str(metadata.get("summary_source", "gemini"))
            skipped += 1
            logger.info("SKIP %s (already gemini)", video_id)
        promise_text, hook_promise_source = resolve_promise_text(title, hook_ocr)
        divergence = semantic_divergence(promise_text, summary, model_name=bert_model)

        metadata.update(
            {
                "summary_source": summary_source,
                "semantic_divergence": divergence,
                "hook_ocr_usable": is_usable_hook_ocr(hook_ocr),
                "hook_promise_source": hook_promise_source,
                "transcript_word_count": len(raw_transcript.split()) if raw_transcript else 0,
            }
        )
        (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        row = {
            "video_id": video_id,
            "title": title,
            "label": human_label,
            "speech_coverage_percent": metadata.get("speech_coverage_percent", 0.0),
            "transcript_word_count": metadata.get("transcript_word_count", 0),
            "transcript_is_empty": not raw_transcript.strip(),
            "ocr_text_detected": metadata.get("ocr_text_detected", bool(hook_ocr)),
            "ocr_char_count": len(hook_ocr),
            "hook_ocr_usable": is_usable_hook_ocr(hook_ocr),
            "hook_promise_source": hook_promise_source,
            "semantic_divergence": divergence,
            "summary_source": summary_source,
            "processing_time_seconds": metadata.get("processing_time_seconds", 0.0),
            "whisper_confidence": metadata.get("whisper_confidence"),
            "transcript": raw_transcript,
            "summary": summary,
            "ocr_text": hook_ocr,
        }
        rows.append(row)
        logger.info(
            "DONE %s summary=%s words=%s div=%s",
            video_id,
            summary_source,
            row["transcript_word_count"],
            f"{divergence:.3f}" if divergence is not None else "N/A",
        )

    target_total = len(sample_df)
    summary = summarize_rows(rows, target_total=target_total, config=config)

    results_dir = PROJECT_ROOT / "outputs" / "spike_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = results_dir / "spike_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPIKE_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    gemini_count = sum(1 for row in rows if row.get("summary_source") == "gemini")
    fallback_count = sum(1 for row in rows if row.get("summary_source") == "extractive_fallback")
    logger.info(
        "Summary sources: gemini=%s extractive_fallback=%s (resummarized=%s skipped=%s)",
        gemini_count,
        fallback_count,
        resummarized,
        skipped,
    )
    if fallback_count:
        logger.warning(
            "%s videos still on extractive_fallback — re-run when API quota resets",
            fallback_count,
        )

    print_spot_check_pairs(rows)
    print_go_no_go_box(summary)
    logger.info("Updated %s | Decision: %s", summary_csv, summary["decision"])


if __name__ == "__main__":
    main()
