"""
Phase 0 feasibility spike: generate text features + diagnostic divergence check.

ARCHITECTURE NOTE:
- Generates title, hook OCR, raw transcript, LLM summary (feature generation only).
- Human labels from BaitBuster-Bangla are NEVER overwritten.
- semantic_divergence is a DIAGNOSTIC sanity check (correlates with human labels),
  not the final clickbait classifier.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._env import load_dotenv as _load_dotenv_file
from scripts._paths import DEFAULT_CONFIG_PATH, ensure_dirs, load_config, resolve_path
from scripts.extract_speech import PipelineFailure, get_ocr_reader, process_video_pipeline
from scripts.semantic_features import (
    is_usable_hook_ocr,
    resolve_promise_text,
    semantic_divergence,
    summarize_transcript,
)

logger = logging.getLogger(__name__)

SPIKE_SEED = 42
FULL_SAMPLE_PER_LABEL = 25
TEST_SAMPLE_TOTAL = 5
USABLE_MIN_WORDS = 5
USABLE_MIN_SPEECH_COVERAGE = 20.0
GO_THRESHOLD_USABLE_RATE = 0.70
MANNWHITNEY_ALPHA = 0.05


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)


def has_hook_frame(frames_dir: Path, video_id: str) -> bool:
    return (frames_dir / str(video_id) / "frame_0.png").exists()


def normalize_label(label: str) -> str:
    text = str(label).strip().lower().replace("-", "_")
    if "non_clickbait" in text or ("non" in text and "clickbait" in text):
        return "non_clickbait"
    if "clickbait" in text:
        return "clickbait"
    return text


def label_to_binary(label: str) -> int | None:
    norm = normalize_label(label)
    if norm == "clickbait":
        return 1
    if norm == "non_clickbait":
        return 0
    return None


def select_spike_sample(
    verified_csv: Path,
    frames_dir: Path,
    output_csv: Path,
    per_label: int,
    seed: int = SPIKE_SEED,
) -> pd.DataFrame:
    dataframe = pd.read_csv(verified_csv)
    dataframe = dataframe[dataframe["audit_status"].astype(str).str.lower() == "live"].copy()
    dataframe = dataframe[dataframe["title"].notna() & dataframe["label"].notna()].copy()
    dataframe["video_id"] = dataframe["video_id"].astype(str)
    dataframe = dataframe[
        dataframe["video_id"].map(lambda vid: has_hook_frame(frames_dir, vid))
    ].reset_index(drop=True)
    dataframe["label_norm"] = dataframe["label"].map(normalize_label)

    clickbait = dataframe[dataframe["label_norm"] == "clickbait"]
    non_clickbait = dataframe[dataframe["label_norm"] == "non_clickbait"]
    if len(clickbait) < per_label or len(non_clickbait) < per_label:
        raise RuntimeError(
            f"Insufficient pool: clickbait={len(clickbait)}, "
            f"non_clickbait={len(non_clickbait)}, need {per_label} each"
        )

    sample = pd.concat(
        [
            clickbait.sample(n=per_label, random_state=seed),
            non_clickbait.sample(n=per_label, random_state=seed),
        ],
        ignore_index=True,
    ).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    out = sample[["video_id", "title", "label"]].copy()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    logger.info("Spike sample saved -> %s (%s rows)", output_csv, len(out))
    return out


def is_usable_transcript(row: dict) -> bool:
    return (
        int(row.get("transcript_word_count", 0)) > USABLE_MIN_WORDS
        and float(row.get("speech_coverage_percent", 0.0)) > USABLE_MIN_SPEECH_COVERAGE
    )


def append_failures(failures_path: Path, failures: list[PipelineFailure]) -> None:
    if not failures:
        return
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not failures_path.exists()
    with failures_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "step", "error_message"])
        if write_header:
            writer.writeheader()
        for failure in failures:
            writer.writerow(
                {
                    "video_id": failure.video_id,
                    "step": failure.step,
                    "error_message": failure.error_message,
                }
            )


def save_video_artifacts(
    output_dir: Path,
    *,
    raw_transcript: str,
    summary: str,
    hook_ocr: str,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw_transcript.txt").write_text(raw_transcript, encoding="utf-8")
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    (output_dir / "hook_ocr.txt").write_text(hook_ocr, encoding="utf-8")
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def compute_divergence_stats(rows: list[dict]) -> dict:
    cb_vals = [
        float(r["semantic_divergence"])
        for r in rows
        if r.get("semantic_divergence") is not None and label_to_binary(r.get("label", "")) == 1
    ]
    nc_vals = [
        float(r["semantic_divergence"])
        for r in rows
        if r.get("semantic_divergence") is not None and label_to_binary(r.get("label", "")) == 0
    ]

    mean_cb = sum(cb_vals) / len(cb_vals) if cb_vals else None
    mean_nc = sum(nc_vals) / len(nc_vals) if nc_vals else None
    direction_correct = (
        mean_cb is not None and mean_nc is not None and mean_cb > mean_nc
    )

    if len(cb_vals) < 2 or len(nc_vals) < 2:
        return {
            "mann_whitney_p": None,
            "direction_correct": direction_correct,
            "mean_divergence_clickbait": mean_cb,
            "mean_divergence_non_clickbait": mean_nc,
        }

    stat = mannwhitneyu(cb_vals, nc_vals, alternative="greater")
    return {
        "mann_whitney_p": float(stat.pvalue),
        "direction_correct": direction_correct,
        "mean_divergence_clickbait": mean_cb,
        "mean_divergence_non_clickbait": mean_nc,
    }


def print_spot_check_pairs(rows: list[dict], seed: int = SPIKE_SEED) -> None:
    eligible = [r for r in rows if r.get("transcript") and r.get("summary")]
    if not eligible:
        print("\n[Spot check] No transcript/summary pairs available.\n")
        return
    sample_count = min(5, len(eligible))
    picks = random.Random(seed).sample(eligible, k=sample_count)
    print("\n" + "=" * 60)
    print("SUMMARY QUALITY SPOT CHECK (manual review)")
    print("=" * 60)
    for idx, row in enumerate(picks, start=1):
        print(f"\n--- Sample {idx} | {row['video_id']} | human_label={row['label']} ---")
        print(f"TITLE: {row.get('title', '')[:200]}")
        print(f"TRANSCRIPT (first 400 chars): {str(row.get('transcript', ''))[:400]}")
        print(f"SUMMARY ({row.get('summary_source', '?')}): {row.get('summary', '')}")
        print(f"semantic_divergence={row.get('semantic_divergence')} (diagnostic only)")
    print("\n" + "=" * 60 + "\n")


def print_go_no_go_box(summary: dict) -> None:
    total = summary["total"]
    usable = summary["usable_transcripts"]
    usable_pct = summary["usable_transcript_rate"] * 100
    ocr_detected = summary["ocr_detected"]
    ocr_pct = summary["ocr_detected_rate"] * 100
    p_val = summary.get("mann_whitney_p")
    p_str = f"{p_val:.3f}" if p_val is not None else "N/A"
    direction = "YES" if summary.get("direction_correct") else "NO"
    decision = summary["decision"]

    print()
    print("+------------------------------------------+")
    print("|       PHASE 0 SPIKE -- GO/NO-GO          |")
    print("+------------------------------------------+")
    print(f"| Usable transcripts:      {usable:>2}/{total:<2} ({usable_pct:>5.1f}%)   |")
    print(f"| OCR detected:            {ocr_detected:>2}/{total:<2} ({ocr_pct:>5.1f}%)   |")
    print(f"| Divergence Mann-Whitney:  p = {p_str:<8s}   |")
    print(f"| Divergence direction correct           |")
    print(f"|   (clickbait > non-clickbait): {direction:<5s}       |")
    print("+------------------------------------------+")
    print(f"| DECISION: {decision:<31s}|")
    print("| (GO if usable > 70% AND direction=YES)   |")
    print("+------------------------------------------+")
    print()


def summarize_rows(rows: list[dict], target_total: int, config: dict) -> dict:
    processed = len(rows)
    usable = sum(1 for row in rows if is_usable_transcript(row))
    ocr_detected = sum(1 for row in rows if row.get("ocr_text_detected"))
    usable_rate = usable / processed if processed else 0.0

    div_stats = compute_divergence_stats(rows)
    direction_ok = bool(div_stats.get("direction_correct"))
    p_val = div_stats.get("mann_whitney_p")
    p_ok = p_val is not None and p_val < MANNWHITNEY_ALPHA

    go_usable = usable_rate > float(
        config.get("spike", {}).get("go_threshold_usable_rate", GO_THRESHOLD_USABLE_RATE)
    )
    decision = "GO" if go_usable and direction_ok else "NO-GO"

    return {
        "total": target_total,
        "processed": processed,
        "usable_transcripts": usable,
        "usable_transcript_rate": usable_rate,
        "ocr_detected": ocr_detected,
        "ocr_detected_rate": ocr_detected / processed if processed else 0.0,
        "mann_whitney_p": p_val,
        "mann_whitney_significant": p_ok,
        "direction_correct": direction_ok,
        "mean_divergence_clickbait": div_stats.get("mean_divergence_clickbait"),
        "mean_divergence_non_clickbait": div_stats.get("mean_divergence_non_clickbait"),
        "decision": decision,
    }


def parse_max_transcribe_seconds(speech_cfg: dict) -> float | None:
    value = speech_cfg.get("max_transcribe_seconds")
    if value is None:
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 0 text-feature feasibility spike")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Process 5 videos (3 clickbait + 2 non-clickbait)",
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Regenerate data/spike_videos.csv (50 stratified rows)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_dotenv_file()
    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    config = load_config(args.config)
    ensure_dirs(config)

    log_path = PROJECT_ROOT / "outputs" / "logs" / "phase0_spike.log"
    setup_logging(log_path)
    logger.info("Phase 0 spike start (test_mode=%s)", args.test_mode)

    verified_csv = resolve_path(config["data"]["verified_csv"])
    frames_dir = resolve_path(config["data"]["frames_dir"])
    spike_csv = resolve_path(config["data"]["spike_csv"])
    spike_transcripts_dir = resolve_path(config["data"]["spike_transcripts_dir"])
    temp_root = resolve_path(config["data"]["spike_audio_dir"])
    cookies_path = resolve_path(config["data"]["youtube_cookies"])
    cookies_file = cookies_path if cookies_path.exists() else None

    results_dir = PROJECT_ROOT / "outputs" / "spike_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = results_dir / "spike_summary.csv"
    failures_csv = results_dir / "failures.csv"
    if failures_csv.exists():
        failures_csv.unlink()

    target_total = TEST_SAMPLE_TOTAL if args.test_mode else FULL_SAMPLE_PER_LABEL * 2
    if not spike_csv.exists() or (args.resample and not args.test_mode):
        sample_df = select_spike_sample(
            verified_csv, frames_dir, spike_csv, FULL_SAMPLE_PER_LABEL, SPIKE_SEED
        )
    else:
        sample_df = pd.read_csv(spike_csv)
        logger.info("Loaded spike sample %s (%s rows)", spike_csv, len(sample_df))

    if args.test_mode:
        cb = sample_df[sample_df["label"].map(normalize_label) == "clickbait"].head(3)
        nc = sample_df[sample_df["label"].map(normalize_label) == "non_clickbait"].head(2)
        sample_df = pd.concat([cb, nc], ignore_index=True)

    logger.info("Pre-loading EasyOCR...")
    get_ocr_reader()

    speech_cfg = config.get("speech", {})
    llm_cfg = config.get("llm", {})
    bert_model = str(config.get("model", {}).get("text_encoder", "sagorsarker/bangla-bert-base"))
    max_transcribe = parse_max_transcribe_seconds(speech_cfg)

    rows: list[dict] = []
    for _, record in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Phase 0 spike"):
        video_id = str(record["video_id"])
        title = str(record.get("title", ""))
        human_label = str(record.get("label", ""))
        video_frames = frames_dir / video_id
        failures: list[PipelineFailure] = []

        try:
            result, failures = process_video_pipeline(
                video_id=video_id,
                frames_dir=video_frames,
                temp_root=temp_root,
                cookies_file=cookies_file,
                sample_rate=int(speech_cfg.get("sample_rate", 16000)),
                vad_threshold=float(speech_cfg.get("vad_threshold", 0.35)),
                min_speech_ms=int(speech_cfg.get("min_speech_segment_ms", 250)),
                use_demucs=bool(speech_cfg.get("use_demucs", False)),
                language=str(speech_cfg.get("whisper_language", "bn")),
                max_transcribe_seconds=max_transcribe,
                hf_model=str(speech_cfg.get("whisper_hf_model", "")),
                fallback_model=str(speech_cfg.get("whisper_fallback_model", "medium")),
                asr_backend=str(speech_cfg.get("asr_backend", "hf_whisper")),
                wav2vec_model=str(speech_cfg.get("wav2vec_model", "")),
            )
        except Exception as exc:
            logger.exception("Pipeline failed for %s", video_id)
            failures.append(PipelineFailure(video_id, "pipeline", str(exc)))
            result = {
                "transcript": "",
                "transcript_word_count": 0,
                "speech_coverage_percent": 0.0,
                "ocr_text": "",
                "ocr_text_detected": False,
                "ocr_char_count": 0,
                "processing_time_seconds": 0.0,
                "whisper_confidence": None,
            }

        append_failures(failures_csv, failures)

        raw_transcript = str(result.get("transcript", ""))
        hook_ocr = str(result.get("ocr_text", ""))

        try:
            summary, summary_source, llm_model_used = summarize_transcript(raw_transcript, llm_cfg)
        except Exception as exc:
            logger.warning("Summary failed for %s: %s", video_id, exc)
            failures.append(PipelineFailure(video_id, "llm_summary", str(exc)))
            summary, summary_source = "", "failed"

        promise_text, hook_promise_source = resolve_promise_text(title, hook_ocr)
        try:
            divergence = semantic_divergence(promise_text, summary, model_name=bert_model)
        except Exception as exc:
            logger.warning("Divergence failed for %s: %s", video_id, exc)
            failures.append(PipelineFailure(video_id, "semantic_divergence", str(exc)))
            divergence = None

        metadata = {
            "video_id": video_id,
            "title": title,
            "human_label": human_label,
            "speech_coverage_percent": result.get("speech_coverage_percent", 0.0),
            "semantic_divergence": divergence,
            "transcript_word_count": result.get("transcript_word_count", 0),
            "ocr_text_detected": result.get("ocr_text_detected", False),
            "hook_ocr_usable": is_usable_hook_ocr(hook_ocr),
            "hook_promise_source": hook_promise_source,
            "summary_source": summary_source,
            "whisper_confidence": result.get("whisper_confidence"),
            "processing_time_seconds": result.get("processing_time_seconds", 0.0),
            "diagnostic_note": (
                "semantic_divergence is sanity-check only; human_label is ground truth"
            ),
        }

        try:
            save_video_artifacts(
                spike_transcripts_dir / video_id,
                raw_transcript=raw_transcript,
                summary=summary,
                hook_ocr=hook_ocr,
                metadata=metadata,
            )
        except Exception as exc:
            logger.warning("Artifact save failed for %s: %s", video_id, exc)
            failures.append(PipelineFailure(video_id, "save_artifacts", str(exc)))

        row = {
            "video_id": video_id,
            "title": title,
            "label": human_label,
            "speech_coverage_percent": result.get("speech_coverage_percent", 0.0),
            "transcript_word_count": result.get("transcript_word_count", 0),
            "transcript_is_empty": not raw_transcript.strip(),
            "ocr_text_detected": result.get("ocr_text_detected", False),
            "ocr_char_count": result.get("ocr_char_count", 0),
            "hook_ocr_usable": is_usable_hook_ocr(hook_ocr),
            "hook_promise_source": hook_promise_source,
            "processing_time_seconds": result.get("processing_time_seconds", 0.0),
            "whisper_confidence": result.get("whisper_confidence"),
            "semantic_divergence": divergence,
            "summary_source": summary_source,
            "transcript": raw_transcript,
            "summary": summary,
            "ocr_text": hook_ocr,
        }
        rows.append(row)

        logger.info(
            "DONE %s | human=%s words=%s cov=%.1f%% div=%s ocr=%s summary=%s",
            video_id,
            human_label,
            row["transcript_word_count"],
            row["speech_coverage_percent"],
            f"{divergence:.3f}" if divergence is not None else "N/A",
            row["ocr_text_detected"],
            summary_source,
        )

    summary = summarize_rows(rows, target_total=target_total, config=config)

    fieldnames = [
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
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print_spot_check_pairs(rows, seed=SPIKE_SEED)
    print_go_no_go_box(summary)

    logger.info("Summary CSV -> %s", summary_csv)
    logger.info("Artifacts -> %s", spike_transcripts_dir)
    logger.info("Failures -> %s", failures_csv)
    logger.info("Decision: %s", summary["decision"])


if __name__ == "__main__":
    main()
