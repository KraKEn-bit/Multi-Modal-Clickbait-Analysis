"""Diagnose and extract Finding-2 data for BanglaBERT hard-subset gaps."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._env import load_dotenv
from scripts._paths import DEFAULT_CONFIG_PATH, ensure_dirs, load_config, resolve_path
from scripts.evaluate_hard_subset import build_hard_subset_frame
from scripts.extract_speech import get_ocr_reader
from scripts.phase1_extract import (
    VERIFIED_FIELDS,
    DailyQuotaExceeded,
    acquire_run_lock,
    build_gemini_limiter,
    is_complete_gemini,
    load_row_from_disk,
    process_one_video,
    setup_logging,
)
from scripts.train_finding2 import MIN_TRANSCRIPT_WORDS

LOG_PATH = PROJECT_ROOT / "outputs" / "logs" / "extract_hard_subset.log"
HARD_CSV_REL = Path("data") / "baseline_banglabert_model" / "hard_subset_video_ids.csv"
DIAGNOSE_PATH = PROJECT_ROOT / "outputs" / "hard_subset_gap_diagnosis.json"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def hard_label_to_human(label: int | str) -> str:
    return "clickbait" if int(label) == 1 else "non_clickbait"


def diagnose_missing_stages(
    missing_ids: list[str],
    transcripts_dir: Path,
    verified_csv: Path,
    subset_csv: Path,
) -> dict:
    verified_ids = set()
    subset_ids = set()
    if verified_csv.exists():
        verified_ids = set(pd.read_csv(verified_csv)["video_id"].astype(str))
    if subset_csv.exists():
        subset_ids = set(pd.read_csv(subset_csv)["video_id"].astype(str))

    stage_counts: dict[str, int] = {}
    per_video: list[dict] = []

    for video_id in missing_ids:
        out_dir = transcripts_dir / video_id
        raw_path = out_dir / "raw_transcript.txt"
        summary_path = out_dir / "summary.txt"
        hook_path = out_dir / "hook_ocr.txt"
        meta_path = out_dir / "metadata.json"

        transcript_words = 0
        summary_source = ""
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                transcript_words = int(meta.get("transcript_word_count", 0) or 0)
                summary_source = str(meta.get("summary_source", ""))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        raw_text = _read_text(raw_path)
        summary_text = _read_text(summary_path)
        hook_exists = hook_path.exists()

        if not out_dir.exists():
            stage = "never_started_no_transcript_dir"
        elif not raw_path.exists() or not raw_text:
            stage = "never_downloaded_or_no_transcript"
        elif transcript_words <= MIN_TRANSCRIPT_WORDS:
            stage = "excluded_transcript_word_count_le_5"
        elif not summary_path.exists() or not summary_text:
            stage = "has_transcript_no_gemini_summary"
        elif summary_source.lower() != "gemini":
            stage = "has_summary_not_gemini"
        elif not hook_exists:
            stage = "missing_hook_ocr_file"
        elif video_id not in verified_ids:
            stage = "complete_on_disk_not_in_verified_csv"
        else:
            stage = "other_complete_issue"

        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        per_video.append(
            {
                "video_id": video_id,
                "stage": stage,
                "transcript_word_count": transcript_words,
                "summary_source": summary_source,
                "in_finding2_subset": video_id in subset_ids,
                "in_finding2_verified": video_id in verified_ids,
            }
        )

    return {
        "missing_n": len(missing_ids),
        "stage_counts": stage_counts,
        "per_video": per_video,
    }


def load_hard_missing_ids(hard_csv: Path, verified_csv: Path, transcripts_dir: Path) -> list[str]:
    hard_df = pd.read_csv(hard_csv)
    hard_ids = sorted(hard_df["video_id"].astype(str).unique().tolist())
    _, meta = build_hard_subset_frame(
        hard_csv,
        verified_csv,
        transcripts_dir,
        min_transcript_words=MIN_TRANSCRIPT_WORDS,
    )
    complete = set(meta.get("evaluated_video_ids", []))
    return [vid for vid in hard_ids if vid not in complete]


def build_hard_queue(hard_csv: Path, video_ids: list[str]) -> pd.DataFrame:
    hard_df = pd.read_csv(hard_csv)
    hard_df["video_id"] = hard_df["video_id"].astype(str)
    queue = hard_df[hard_df["video_id"].isin(video_ids)].copy()
    queue["label"] = queue["label"].map(hard_label_to_human)
    queue["from_spike"] = False
    return queue.reset_index(drop=True)


def merge_verified_rows(verified_csv: Path, new_rows: list[dict]) -> None:
    existing_rows: list[dict] = []
    if verified_csv.exists():
        existing = pd.read_csv(verified_csv)
        update_ids = {str(row["video_id"]) for row in new_rows}
        existing_rows = existing[~existing["video_id"].astype(str).isin(update_ids)].to_dict("records")

    merged = existing_rows + new_rows
    verified_csv.parent.mkdir(parents=True, exist_ok=True)
    with verified_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=VERIFIED_FIELDS)
        writer.writeheader()
        for row in merged:
            writer.writerow({field: row.get(field, "") for field in VERIFIED_FIELDS})


def print_diagnosis(report: dict) -> None:
    print("\n=== Hard-subset gap diagnosis ===")
    print(f"Missing complete Finding-2 data: {report['missing_n']}")
    print("Stage counts:")
    for stage, count in sorted(report["stage_counts"].items(), key=lambda item: (-item[1], item[0])):
        print(f"  {stage}: {count}")


def main() -> None:
    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Hard-subset gap diagnosis and targeted extraction")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--hard-csv", type=Path, default=None)
    parser.add_argument("--diagnose-only", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    setup_logging(LOG_PATH)
    config = load_config(args.config)
    ensure_dirs(config)

    vtcf_root = resolve_path(config["vtcf_research_root"])
    hard_csv = args.hard_csv or (vtcf_root / HARD_CSV_REL)
    verified_csv = resolve_path(config["data"]["finding2_verified_csv"])
    subset_csv = resolve_path(config["data"]["finding2_subset_csv"])
    transcripts_dir = resolve_path(config["data"]["transcripts_dir"])

    missing_ids = load_hard_missing_ids(hard_csv, verified_csv, transcripts_dir)
    diagnosis = diagnose_missing_stages(missing_ids, transcripts_dir, verified_csv, subset_csv)
    DIAGNOSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DIAGNOSE_PATH.write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
    print_diagnosis(diagnosis)
    print(f"\nSaved diagnosis -> {DIAGNOSE_PATH}")

    if args.diagnose_only or not args.extract:
        return

    if not missing_ids:
        print("Nothing to extract.")
        return

    acquire_run_lock()
    queue = build_hard_queue(hard_csv, missing_ids)
    limiter = build_gemini_limiter(
        config.get("llm", {}),
        resolve_path(config["data"]["llm_call_log"]),
        config.get("phase1", {}),
    )
    get_ocr_reader()

    transcripts_dir = resolve_path(config["data"]["transcripts_dir"])
    frames_root = resolve_path(config["data"]["frames_dir"])
    audio_dir = resolve_path(config["data"]["phase1_audio_dir"])
    spike_transcripts_dir = resolve_path(config["data"]["spike_transcripts_dir"])

    new_rows: list[dict] = []
    resume = not args.no_resume
    try:
        for _, record in tqdm(queue.iterrows(), total=len(queue), desc="HardSubset"):
            video_id = str(record["video_id"])
            if resume and is_complete_gemini(transcripts_dir / video_id):
                row = load_row_from_disk(
                    video_id,
                    str(record.get("title", "")),
                    str(record.get("label", "")),
                    False,
                    transcripts_dir / video_id,
                    config,
                )
                if row:
                    new_rows.append(row)
                    merge_verified_rows(verified_csv, new_rows)
                continue
            try:
                row = process_one_video(
                    record,
                    config=config,
                    transcripts_dir=transcripts_dir,
                    frames_dir=frames_root,
                    audio_dir=audio_dir,
                    spike_transcripts_dir=spike_transcripts_dir,
                    llm_cfg=config.get("llm", {}),
                    limiter=limiter,
                    resume=resume,
                )
            except DailyQuotaExceeded as exc:
                print(f"Stopping: {exc}")
                break
            if row:
                new_rows.append(row)
                merge_verified_rows(verified_csv, new_rows)
    finally:
        print(f"Extracted/updated {len(new_rows)} hard-subset rows in {verified_csv}")


if __name__ == "__main__":
    main()
