"""
Re-transcribe all Phase 0 spike videos with the configured ASR backend.

Updates raw_transcript.txt + metadata.json per video; does not re-summarize
(use resummarize_spike.py after this).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._env import load_dotenv
from scripts._paths import ensure_dirs, load_config, resolve_path
from scripts.extract_speech import PipelineFailure, process_video_pipeline

logger = logging.getLogger(__name__)

PROGRESS_PATH = PROJECT_ROOT / "outputs" / "spike_results" / "retranscribe_progress.json"
MIN_RESUME_WORDS = 1


def setup_logging() -> None:
    log_path = PROJECT_ROOT / "outputs" / "logs" / "retranscribe_spike.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-run ASR for spike sample")
    parser.add_argument(
        "--videos",
        nargs="+",
        help="Optional subset of video IDs (default: all rows in spike_csv)",
    )
    parser.add_argument(
        "--retry-failed-only",
        action="store_true",
        help="Only videos with empty/zero-word transcripts in spike_summary.csv",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip videos already transcribed with the current ASR backend (default: on)",
    )
    return parser.parse_args()


def load_progress() -> dict:
    if not PROGRESS_PATH.exists():
        return {"completed": [], "asr_backend": None, "hf_model": None}
    try:
        return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"completed": [], "asr_backend": None, "hf_model": None}


def save_progress(
    *,
    completed: list[str],
    asr_backend: str,
    hf_model: str,
    last_video_id: str | None = None,
) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed": completed,
        "asr_backend": asr_backend,
        "hf_model": hf_model,
        "last_video_id": last_video_id,
        "updated_at": pd.Timestamp.now().isoformat(),
    }
    PROGRESS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def is_video_done(
    out_dir: Path,
    *,
    asr_backend: str,
    hf_model: str,
    min_words: int = MIN_RESUME_WORDS,
) -> bool:
    """True when a prior run saved a non-empty transcript for this backend."""
    meta_path = out_dir / "metadata.json"
    raw_path = out_dir / "raw_transcript.txt"
    if not meta_path.exists() or not raw_path.exists():
        return False
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if str(metadata.get("asr_backend", "")) != asr_backend:
        return False
    if hf_model and str(metadata.get("hf_model", "")) not in ("", hf_model):
        return False
    word_count = int(metadata.get("transcript_word_count", 0) or 0)
    if word_count < min_words:
        return False
    if not raw_path.read_text(encoding="utf-8").strip():
        return False
    # OpenAI fallback sets whisper_confidence; pure HF leaves it null.
    if asr_backend == "hf_whisper" and metadata.get("whisper_confidence") is not None:
        return False
    return True


def main() -> None:
    load_dotenv()
    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    args = parse_args()
    setup_logging()
    config = load_config()
    ensure_dirs(config)

    spike_csv = resolve_path(config["data"]["spike_csv"])
    frames_dir = resolve_path(config["data"]["frames_dir"])
    spike_transcripts_dir = resolve_path(config["data"]["spike_transcripts_dir"])
    temp_root = resolve_path(config["data"]["spike_audio_dir"])
    cookies_path = resolve_path(config["data"]["youtube_cookies"])
    cookies_file = cookies_path if cookies_path.exists() else None
    speech_cfg = config.get("speech", {})

    sample_df = pd.read_csv(spike_csv)
    video_ids = [str(v) for v in sample_df["video_id"].tolist()]

    if args.retry_failed_only:
        summary_csv = PROJECT_ROOT / "outputs" / "spike_results" / "spike_summary.csv"
        if summary_csv.exists():
            summary = pd.read_csv(summary_csv)
            failed = summary[
                summary["transcript_word_count"].fillna(0).astype(int) == 0
            ]["video_id"].astype(str).tolist()
            video_ids = [v for v in video_ids if v in failed]
            logger.info("Retry-failed-only: %s videos", len(video_ids))

    if args.videos:
        wanted = set(args.videos)
        video_ids = [v for v in video_ids if v in wanted]

    asr_backend = str(speech_cfg.get("asr_backend", "hf_whisper"))
    hf_model = str(speech_cfg.get("whisper_hf_model", ""))
    progress = load_progress()
    completed_ids: list[str] = []

    if args.resume:
        pending: list[str] = []
        for video_id in video_ids:
            out_dir = spike_transcripts_dir / video_id
            if is_video_done(out_dir, asr_backend=asr_backend, hf_model=hf_model):
                completed_ids.append(video_id)
                continue
            pending.append(video_id)
        if completed_ids:
            logger.info(
                "Resume: skipping %s already-done videos (%s pending)",
                len(completed_ids),
                len(pending),
            )
        video_ids = pending

    logger.info(
        "Re-transcribing %s videos with backend=%s hf=%s wav2vec=%s (resume=%s)",
        len(video_ids),
        asr_backend,
        hf_model,
        speech_cfg.get("wav2vec_model"),
        args.resume,
    )

    max_transcribe = speech_cfg.get("max_transcribe_seconds")
    max_transcribe = float(max_transcribe) if max_transcribe is not None else None

    for video_id in tqdm(video_ids, desc="Re-transcribe"):
        row = sample_df[sample_df["video_id"].astype(str) == video_id].iloc[0]
        title = str(row.get("title", ""))
        human_label = str(row.get("label", ""))
        video_frames = frames_dir / video_id

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
                asr_backend=asr_backend,
                wav2vec_model=str(speech_cfg.get("wav2vec_model", "")),
            )
        except Exception as exc:
            logger.exception("Pipeline failed for %s", video_id)
            failures = [PipelineFailure(video_id, "pipeline", str(exc))]
            result = {
                "transcript": "",
                "transcript_word_count": 0,
                "speech_coverage_percent": 0.0,
                "ocr_text": "",
                "ocr_text_detected": False,
                "whisper_confidence": None,
                "processing_time_seconds": 0.0,
            }

        out_dir = spike_transcripts_dir / video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_transcript = str(result.get("transcript", ""))
        hook_ocr = str(result.get("ocr_text", ""))
        (out_dir / "raw_transcript.txt").write_text(raw_transcript, encoding="utf-8")
        (out_dir / "hook_ocr.txt").write_text(hook_ocr, encoding="utf-8")

        meta_path = out_dir / "metadata.json"
        metadata = {}
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "video_id": video_id,
                "title": title,
                "human_label": human_label,
                "speech_coverage_percent": result.get("speech_coverage_percent", 0.0),
                "transcript_word_count": result.get("transcript_word_count", 0),
                "ocr_text_detected": result.get("ocr_text_detected", False),
                "whisper_confidence": result.get("whisper_confidence"),
                "processing_time_seconds": result.get("processing_time_seconds", 0.0),
                "asr_backend": asr_backend,
                "hf_model": hf_model,
            }
        )
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        if failures:
            logger.warning(
                "%s failures: %s",
                video_id,
                "; ".join(f"{f.step}:{f.error_message[:80]}" for f in failures),
            )
        logger.info(
            "DONE %s words=%s cov=%.1f%%",
            video_id,
            metadata["transcript_word_count"],
            float(metadata.get("speech_coverage_percent", 0.0)),
        )

        if int(metadata.get("transcript_word_count", 0) or 0) >= MIN_RESUME_WORDS:
            completed_ids.append(video_id)
            save_progress(
                completed=completed_ids,
                asr_backend=asr_backend,
                hf_model=hf_model,
                last_video_id=video_id,
            )


if __name__ == "__main__":
    main()
