"""
Phase 1 feature extraction: stratified subset → ASR + OCR + Gemini summary.

Reuses Phase 0 spike artifacts (no re-ASR for those 50). human_label is copied
unchanged from BaitBuster-Bangla annotations only.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._env import load_dotenv
from scripts._paths import DEFAULT_CONFIG_PATH, ensure_dirs, load_config, resolve_path
from scripts.extend_dataset import filter_usable, map_detection_label
from scripts.extract_speech import PipelineFailure, get_ocr_reader, process_video_pipeline
from scripts.phase0_spike import is_usable_transcript, normalize_label, setup_logging
from scripts.semantic_features import (
    SUMMARY_PROMPT,
    get_gemini_model_rotation,
    is_usable_hook_ocr,
    resolve_promise_text,
    semantic_divergence,
    summarize_transcript,
)

logger = logging.getLogger(__name__)

LOG_PATH = PROJECT_ROOT / "outputs" / "logs" / "extract_speech_phase1.log"
FAILURES_PATH = PROJECT_ROOT / "outputs" / "logs" / "phase1_failures.csv"
PROGRESS_PATH = PROJECT_ROOT / "outputs" / "logs" / "phase1_progress.json"
RUN_LOCK_PATH = PROJECT_ROOT / "outputs" / "logs" / "phase1_run.lock"
VERIFIED_FIELDS = [
    "video_id",
    "title",
    "human_label",
    "promise_text",
    "transcript_path",
    "summary_path",
    "speech_coverage_percent",
    "semantic_divergence",
    "summary_source",
    "usable_for_training",
    "hook_promise_source",
    "hook_ocr_usable",
    "from_spike",
]

LLM_LOG_FIELDS = [
    "timestamp",
    "video_id",
    "model_name",
    "model_version",
    "prompt_hash",
    "summary_source",
    "success",
    "error_message",
]


class DailyQuotaExceeded(Exception):
    """Raised when Gemini daily quota for this run is exhausted."""


class GeminiRateLimiter:
    def __init__(
        self,
        log_path: Path,
        model_limits: dict[str, int],
        min_interval_s: float,
        model_chain: list[str] | None = None,
    ) -> None:
        self.log_path = log_path
        self.model_limits = model_limits
        self.model_chain = model_chain or list(model_limits.keys())
        self.min_interval_s = min_interval_s
        self._last_call_at: float = 0.0
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=LLM_LOG_FIELDS).writeheader()

    def calls_today_for(self, model: str) -> int:
        if not self.log_path.exists():
            return 0
        today = date.today().isoformat()
        count = 0
        with self.log_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                ts = str(row.get("timestamp", ""))
                if not ts.startswith(today):
                    continue
                if str(row.get("success", "")).lower() != "true":
                    continue
                if str(row.get("model_version", "")) == model:
                    count += 1
        return count

    def calls_today(self) -> int:
        return sum(self.calls_today_for(model) for model in self.model_chain)

    def can_call(self, model: str) -> bool:
        limit = self.model_limits.get(model, 0)
        if limit <= 0:
            return True
        return self.calls_today_for(model) < limit

    def remaining_today(self) -> int:
        total = 0
        for model in self.model_chain:
            limit = self.model_limits.get(model, 0)
            if limit <= 0:
                continue
            total += max(0, limit - self.calls_today_for(model))
        return total

    def next_available_model(self) -> str | None:
        for model in self.model_chain:
            if self.can_call(model):
                return model
        return None

    def assert_can_call(self) -> None:
        if self.next_available_model() is None:
            raise DailyQuotaExceeded(
                f"All Gemini model daily limits reached: {self.model_limits}"
            )

    def wait_turn(self) -> None:
        elapsed = time.perf_counter() - self._last_call_at
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)

    def log_call(
        self,
        *,
        video_id: str,
        model_name: str,
        model_version: str,
        prompt_hash: str,
        summary_source: str,
        success: bool,
        error_message: str = "",
    ) -> None:
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=LLM_LOG_FIELDS)
            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(),
                    "video_id": video_id,
                    "model_name": model_name,
                    "model_version": model_version,
                    "prompt_hash": prompt_hash,
                    "summary_source": summary_source,
                    "success": success,
                    "error_message": error_message[:500],
                }
            )
        self._last_call_at = time.perf_counter()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 speech + summary extraction")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Process only phase1.test_mode_count new (non-spike) videos",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip videos with existing gemini summary.txt (default: on)",
    )
    parser.add_argument(
        "--rebuild-subset",
        action="store_true",
        help="Regenerate data/finding2_subset.csv",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="Total videos in subset (default: phase1.target_total from config)",
    )
    return parser.parse_args()


def append_failures(failures: list[PipelineFailure]) -> None:
    if not failures:
        return
    FAILURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not FAILURES_PATH.exists()
    with FAILURES_PATH.open("a", newline="", encoding="utf-8") as handle:
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


def build_finding2_subset(
    config: dict,
    *,
    target_total: int,
    seed: int,
    output_path: Path,
) -> pd.DataFrame:
    verified_csv = resolve_path(config["data"]["verified_csv"])
    frames_dir = resolve_path(config["data"]["frames_dir"])
    spike_csv = resolve_path(config["data"]["spike_csv"])

    verified = pd.read_csv(verified_csv)
    verified = filter_usable(verified, frames_dir)
    verified["video_id"] = verified["video_id"].astype(str)
    verified["label_norm"] = verified["label"].map(normalize_label)
    verified = verified[verified["label_norm"].isin(["clickbait", "non_clickbait"])].copy()

    spike_df = pd.read_csv(spike_csv)
    spike_ids = set(spike_df["video_id"].astype(str))

    spike_in_verified = verified[verified["video_id"].isin(spike_ids)].copy()
    missing_spike = spike_ids - set(spike_in_verified["video_id"])
    if missing_spike:
        extra = spike_df[spike_df["video_id"].astype(str).isin(missing_spike)].copy()
        extra["label_norm"] = extra["label"].map(normalize_label)
        spike_in_verified = pd.concat([spike_in_verified, extra], ignore_index=True)

    need_new = max(target_total - len(spike_in_verified), 0)
    pool = verified[~verified["video_id"].isin(spike_ids)].copy()
    pool["detection_label"] = pool["label"].map(map_detection_label)

    if need_new > len(pool):
        raise RuntimeError(
            f"Cannot sample {need_new} videos; pool has only {len(pool)} after excluding spike"
        )

    if need_new > 0:
        sampled, _ = train_test_split(
            pool,
            train_size=need_new,
            random_state=seed,
            stratify=pool["detection_label"],
        )
        subset = pd.concat([spike_in_verified, sampled], ignore_index=True)
    else:
        subset = spike_in_verified.head(target_total).copy()

    subset = subset.drop_duplicates(subset=["video_id"]).reset_index(drop=True)
    subset["from_spike"] = subset["video_id"].isin(spike_ids)
    out = subset[["video_id", "title", "label", "from_spike"]].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    logger.info(
        "Subset saved %s | total=%s spike=%s new=%s",
        output_path,
        len(out),
        int(out["from_spike"].sum()),
        int((~out["from_spike"]).sum()),
    )
    return out


def import_spike_artifacts(
    video_id: str,
    spike_dir: Path,
    out_dir: Path,
) -> bool:
    """Copy Phase 0 spike outputs into data/transcripts/{id}/."""
    src = spike_dir / video_id
    if not src.exists():
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("raw_transcript.txt", "summary.txt", "hook_ocr.txt", "metadata.json"):
        s = src / name
        if s.exists():
            shutil.copy2(s, out_dir / name)
    return (out_dir / "summary.txt").exists()


def is_complete_gemini(out_dir: Path) -> bool:
    summary_path = out_dir / "summary.txt"
    meta_path = out_dir / "metadata.json"
    if not summary_path.exists() or not summary_path.read_text(encoding="utf-8").strip():
        return False
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return str(meta.get("summary_source", "")).lower() == "gemini"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_run_lock(lock_path: Path = RUN_LOCK_PATH) -> None:
    """Prevent two GPU-heavy phase1_extract instances (causes CUDA crashes)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            old_pid = int(lock_path.read_text(encoding="utf-8").strip())
        except ValueError:
            old_pid = -1
        if _pid_alive(old_pid):
            raise SystemExit(
                f"Another phase1_extract.py is already running (PID {old_pid}). "
                "Wait for it to finish or stop it before starting a new run."
            )
        lock_path.unlink(missing_ok=True)
    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    atexit.register(release_run_lock, lock_path)


def release_run_lock(lock_path: Path = RUN_LOCK_PATH) -> None:
    if not lock_path.exists():
        return
    try:
        if int(lock_path.read_text(encoding="utf-8").strip()) == os.getpid():
            lock_path.unlink(missing_ok=True)
    except (ValueError, OSError):
        pass


def load_cached_pipeline_result(out_dir: Path) -> dict | None:
    """Reuse ASR/OCR on disk when summarization failed or power cut mid-run."""
    raw_path = out_dir / "raw_transcript.txt"
    if not raw_path.exists():
        return None
    raw = raw_path.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    hook_ocr = ""
    if (out_dir / "hook_ocr.txt").exists():
        hook_ocr = (out_dir / "hook_ocr.txt").read_text(encoding="utf-8").strip()
    metadata: dict = {}
    if (out_dir / "metadata.json").exists():
        try:
            metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
    word_count = len(raw.split())
    coverage = float(metadata.get("speech_coverage_percent", 0.0))
    if not is_usable_transcript(
        {
            "transcript_word_count": word_count,
            "speech_coverage_percent": coverage,
        }
    ):
        return None
    return {
        "transcript": raw,
        "ocr_text": hook_ocr,
        "speech_coverage_percent": coverage,
        "transcript_word_count": word_count,
        "ocr_text_detected": metadata.get("ocr_text_detected", bool(hook_ocr)),
        "ocr_char_count": metadata.get("ocr_char_count", len(hook_ocr)),
        "whisper_confidence": metadata.get("whisper_confidence"),
        "processing_time_seconds": metadata.get("processing_time_seconds", 0.0),
    }


def write_progress_checkpoint(
    *,
    subset_df: pd.DataFrame,
    transcripts_dir: Path,
    limiter: GeminiRateLimiter,
    target_total: int,
    last_video_id: str,
) -> None:
    done = sum(
        1
        for vid in subset_df["video_id"].astype(str)
        if is_complete_gemini(transcripts_dir / vid)
    )
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(
        json.dumps(
            {
                "updated_at": datetime.now().isoformat(),
                "last_video_id": last_video_id,
                "complete_gemini": done,
                "target_total": target_total,
                "remaining": target_total - done,
                "gemini_calls_today": limiter.calls_today(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def build_gemini_limiter(llm_cfg: dict, llm_log: Path, phase1_cfg: dict) -> GeminiRateLimiter:
    rotation = get_gemini_model_rotation(llm_cfg)
    model_limits = {model: limit for model, limit in rotation}
    model_chain = [model for model, _ in rotation]
    return GeminiRateLimiter(
        llm_log,
        model_limits=model_limits,
        min_interval_s=float(phase1_cfg.get("gemini_min_interval_s", 6)),
        model_chain=model_chain,
    )


def summarize_with_logging(
    video_id: str,
    transcript: str,
    llm_cfg: dict,
    limiter: GeminiRateLimiter,
) -> tuple[str, str, str]:
    """Returns (summary, summary_source, llm_model_used)."""
    primary = str(llm_cfg.get("gemini_model", "gemini-3.1-flash-lite"))
    cleaned = transcript.strip()
    if not cleaned:
        return "", "empty_transcript", primary

    limiter.assert_can_call()
    limiter.wait_turn()
    prompt = SUMMARY_PROMPT.format(transcript=cleaned[:12000])
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

    try:
        summary, source, model_used = summarize_transcript(
            cleaned,
            llm_cfg,
            can_use_model=limiter.can_call,
        )
        if source == "gemini" and not limiter.can_call(model_used):
            raise DailyQuotaExceeded(f"Model {model_used} quota exhausted during call.")
        limiter.log_call(
            video_id=video_id,
            model_name="gemini" if source == "gemini" else source,
            model_version=model_used,
            prompt_hash=prompt_hash,
            summary_source=source,
            success=source == "gemini",
            error_message="" if source == "gemini" else f"fallback={source}",
        )
        if source != "gemini":
            limiter.assert_can_call()
        return summary, source, model_used
    except DailyQuotaExceeded:
        raise
    except Exception as exc:
        limiter.log_call(
            video_id=video_id,
            model_name="gemini",
            model_version=primary,
            prompt_hash=prompt_hash,
            summary_source="error",
            success=False,
            error_message=str(exc),
        )
        raise


def save_video_outputs(
    out_dir: Path,
    *,
    raw_transcript: str,
    summary: str,
    hook_ocr: str,
    metadata: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_transcript.txt").write_text(raw_transcript, encoding="utf-8")
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    (out_dir / "hook_ocr.txt").write_text(hook_ocr, encoding="utf-8")
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def process_one_video(
    record: pd.Series,
    *,
    config: dict,
    transcripts_dir: Path,
    frames_dir: Path,
    audio_dir: Path,
    spike_transcripts_dir: Path,
    llm_cfg: dict,
    limiter: GeminiRateLimiter,
    resume: bool,
) -> dict | None:
    video_id = str(record["video_id"])
    title = str(record.get("title", ""))
    human_label = str(record.get("label", ""))
    from_spike = bool(record.get("from_spike", False))
    out_dir = transcripts_dir / video_id

    if resume and is_complete_gemini(out_dir):
        logger.info("SKIP %s (resume: gemini summary exists)", video_id)
        return load_row_from_disk(
            video_id, title, human_label, from_spike, out_dir, config
        )

    if from_spike and import_spike_artifacts(video_id, spike_transcripts_dir, out_dir):
        if is_complete_gemini(out_dir):
            logger.info("IMPORTED spike artifacts for %s", video_id)
            return load_row_from_disk(
                video_id, title, human_label, True, out_dir, config
            )

    speech_cfg = config.get("speech", {})
    cookies_path = resolve_path(config["data"]["youtube_cookies"])
    cookies_file = cookies_path if cookies_path.exists() else None

    cached = load_cached_pipeline_result(out_dir) if resume else None
    if cached:
        logger.info("RESUME %s from cached transcript (skip ASR)", video_id)
        result = cached
        failures: list[PipelineFailure] = []
    else:
        result, failures = process_video_pipeline(
            video_id=video_id,
            frames_dir=frames_dir / video_id,
            temp_root=audio_dir,
            cookies_file=cookies_file,
            sample_rate=int(speech_cfg.get("sample_rate", 16000)),
            vad_threshold=float(speech_cfg.get("vad_threshold", 0.35)),
            min_speech_ms=int(speech_cfg.get("min_speech_segment_ms", 250)),
            use_demucs=bool(speech_cfg.get("use_demucs", False)),
            language=str(speech_cfg.get("whisper_language", "bn")),
            max_transcribe_seconds=None,
            hf_model=str(speech_cfg.get("whisper_hf_model", "")),
            fallback_model=str(speech_cfg.get("whisper_fallback_model", "medium")),
            asr_backend=str(speech_cfg.get("asr_backend", "hf_whisper")),
            wav2vec_model=str(speech_cfg.get("wav2vec_model", "")),
        )
    append_failures(failures)

    raw_transcript = str(result.get("transcript", ""))
    hook_ocr = str(result.get("ocr_text", ""))
    promise_text, hook_promise_source = resolve_promise_text(title, hook_ocr)

    try:
        summary, summary_source, llm_model_used = summarize_with_logging(
            video_id, raw_transcript, llm_cfg, limiter
        )
    except DailyQuotaExceeded:
        raise
    except Exception as exc:
        logger.warning("Summary failed for %s: %s", video_id, exc)
        summary, summary_source, llm_model_used = "", "failed", str(
            llm_cfg.get("gemini_model", "")
        )

    bert_model = str(config.get("model", {}).get("text_encoder", "sagorsarker/bangla-bert-base"))
    divergence = (
        semantic_divergence(promise_text, summary, model_name=bert_model)
        if summary.strip()
        else None
    )

    metadata = {
        "video_id": video_id,
        "title": title,
        "human_label": human_label,
        "speech_coverage_percent": result.get("speech_coverage_percent", 0.0),
        "transcript_word_count": result.get("transcript_word_count", 0),
        "ocr_text_detected": result.get("ocr_text_detected", False),
        "ocr_char_count": result.get("ocr_char_count", 0),
        "hook_ocr_usable": is_usable_hook_ocr(hook_ocr),
        "hook_promise_source": hook_promise_source,
        "semantic_divergence": divergence,
        "summary_source": summary_source,
        "llm_model_used": llm_model_used,
        "whisper_confidence": result.get("whisper_confidence"),
        "processing_time_seconds": result.get("processing_time_seconds", 0.0),
        "asr_backend": speech_cfg.get("asr_backend"),
        "hf_model": speech_cfg.get("whisper_hf_model"),
        "diagnostic_note": "human_label is ground truth from BaitBuster-Bangla",
    }
    save_video_outputs(
        out_dir,
        raw_transcript=raw_transcript,
        summary=summary,
        hook_ocr=hook_ocr,
        metadata=metadata,
    )

    row = build_row_dict(
        video_id=video_id,
        title=title,
        human_label=human_label,
        from_spike=from_spike,
        out_dir=out_dir,
        promise_text=promise_text,
        hook_promise_source=hook_promise_source,
        hook_ocr_usable=is_usable_hook_ocr(hook_ocr),
        metadata=metadata,
        divergence=divergence,
        summary_source=summary_source,
    )
    logger.info(
        "DONE %s summary=%s words=%s div=%s",
        video_id,
        summary_source,
        metadata["transcript_word_count"],
        f"{divergence:.3f}" if divergence is not None else "N/A",
    )
    return row


def build_row_dict(
    *,
    video_id: str,
    title: str,
    human_label: str,
    from_spike: bool,
    out_dir: Path,
    promise_text: str,
    hook_promise_source: str,
    hook_ocr_usable: bool,
    metadata: dict,
    divergence: float | None,
    summary_source: str,
) -> dict:
    row = {
        "video_id": video_id,
        "title": title,
        "human_label": human_label,
        "promise_text": promise_text,
        "transcript_path": str(out_dir / "raw_transcript.txt"),
        "summary_path": str(out_dir / "summary.txt"),
        "speech_coverage_percent": metadata.get("speech_coverage_percent", 0.0),
        "semantic_divergence": divergence,
        "summary_source": summary_source,
        "hook_promise_source": hook_promise_source,
        "hook_ocr_usable": hook_ocr_usable,
        "from_spike": from_spike,
    }
    row["usable_for_training"] = is_usable_transcript(
        {
            "transcript_word_count": metadata.get("transcript_word_count", 0),
            "speech_coverage_percent": metadata.get("speech_coverage_percent", 0.0),
        }
    )
    return row


def load_row_from_disk(
    video_id: str,
    title: str,
    human_label: str,
    from_spike: bool,
    out_dir: Path,
    config: dict,
) -> dict | None:
    if not out_dir.exists():
        return None
    raw = (
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
    metadata: dict = {}
    if (out_dir / "metadata.json").exists():
        metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))

    title = str(metadata.get("title", title))
    human_label = str(metadata.get("human_label", human_label))
    promise_text, hook_promise_source = resolve_promise_text(title, hook_ocr)
    bert_model = str(config.get("model", {}).get("text_encoder", "sagorsarker/bangla-bert-base"))
    divergence = metadata.get("semantic_divergence")
    if divergence is None and summary.strip():
        divergence = semantic_divergence(promise_text, summary, model_name=bert_model)
        metadata["semantic_divergence"] = divergence
        metadata["hook_promise_source"] = hook_promise_source
        metadata["hook_ocr_usable"] = is_usable_hook_ocr(hook_ocr)
        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return build_row_dict(
        video_id=video_id,
        title=title,
        human_label=human_label,
        from_spike=from_spike,
        out_dir=out_dir,
        promise_text=promise_text,
        hook_promise_source=hook_promise_source,
        hook_ocr_usable=is_usable_hook_ocr(hook_ocr),
        metadata={
            **metadata,
            "transcript_word_count": len(raw.split()) if raw else 0,
            "speech_coverage_percent": metadata.get("speech_coverage_percent", 0.0),
        },
        divergence=divergence,
        summary_source=str(metadata.get("summary_source", "")),
    )


def write_verified_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=VERIFIED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_progress_summary(
    subset_df: pd.DataFrame,
    transcripts_dir: Path,
    limiter: GeminiRateLimiter,
    target_total: int,
) -> None:
    done = sum(
        1
        for vid in subset_df["video_id"].astype(str)
        if is_complete_gemini(transcripts_dir / vid)
    )
    remaining = target_total - done
    calls_today = limiter.calls_today()
    capacity = limiter.remaining_today() + calls_today
    eta_days = (remaining / capacity) if capacity > 0 and remaining > 0 else 0
    day_n = 1
    logger.info(
        "Day %s progress: %s/%s complete | %s summaries remaining | "
        "Gemini today %s (remaining %s) | ETA ~%.1f days at quota rate",
        day_n,
        done,
        target_total,
        remaining,
        calls_today,
        limiter.remaining_today(),
        eta_days,
    )


def main() -> None:
    args = parse_args()
    load_dotenv()
    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    setup_logging(LOG_PATH)
    acquire_run_lock()
    config = load_config(args.config)
    ensure_dirs(config)

    phase1_cfg = config.get("phase1", {})
    target_total = int(args.target or phase1_cfg.get("target_total", 400))
    seed = int(config.get("data", {}).get("subset_seed", 42))
    test_count = int(phase1_cfg.get("test_mode_count", 20))

    subset_csv = resolve_path(config["data"]["finding2_subset_csv"])
    verified_csv = resolve_path(config["data"]["finding2_verified_csv"])
    transcripts_dir = resolve_path(config["data"]["transcripts_dir"])
    spike_transcripts_dir = resolve_path(config["data"]["spike_transcripts_dir"])
    frames_root = resolve_path(config["data"]["frames_dir"])
    audio_dir = resolve_path(config["data"]["phase1_audio_dir"])
    llm_log = resolve_path(config["data"]["llm_call_log"])

    if args.rebuild_subset or not subset_csv.exists():
        subset_df = build_finding2_subset(
            config, target_total=target_total, seed=seed, output_path=subset_csv
        )
    else:
        subset_df = pd.read_csv(subset_csv)
        subset_df["video_id"] = subset_df["video_id"].astype(str)
        if len(subset_df) < target_total:
            logger.info(
                "Expanding subset %s → %s videos (rebuild)",
                len(subset_df),
                target_total,
            )
            subset_df = build_finding2_subset(
                config, target_total=target_total, seed=seed, output_path=subset_csv
            )

    queue = subset_df.copy()
    if args.test_mode:
        queue = queue[~queue["from_spike"].astype(bool)].head(test_count)
        logger.info("TEST MODE: processing %s new (non-spike) videos", len(queue))

    limiter = build_gemini_limiter(config.get("llm", {}), llm_log, phase1_cfg)

    get_ocr_reader()
    rows_by_id: dict[str, dict] = {}

    for _, record in subset_df.iterrows():
        vid = str(record["video_id"])
        if args.resume and is_complete_gemini(transcripts_dir / vid):
            row = load_row_from_disk(
                vid,
                str(record.get("title", "")),
                str(record.get("label", "")),
                bool(record.get("from_spike", False)),
                transcripts_dir / vid,
                config,
            )
            if row:
                rows_by_id[vid] = row

    try:
        for _, record in tqdm(queue.iterrows(), total=len(queue), desc="Phase1"):
            video_id = str(record["video_id"])
            if args.resume and video_id in rows_by_id:
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
                    resume=args.resume,
                )
            except DailyQuotaExceeded as exc:
                logger.warning("Stopping run: %s", exc)
                break
            if row:
                rows_by_id[video_id] = row
                ordered_rows = [
                    rows_by_id[str(v)]
                    for v in subset_df["video_id"].astype(str)
                    if str(v) in rows_by_id
                ]
                write_verified_csv(ordered_rows, verified_csv)
                write_progress_checkpoint(
                    subset_df=subset_df,
                    transcripts_dir=transcripts_dir,
                    limiter=limiter,
                    target_total=target_total,
                    last_video_id=video_id,
                )
    finally:
        for _, record in subset_df.iterrows():
            vid = str(record["video_id"])
            if vid in rows_by_id:
                continue
            if is_complete_gemini(transcripts_dir / vid):
                row = load_row_from_disk(
                    vid,
                    str(record.get("title", "")),
                    str(record.get("label", "")),
                    bool(record.get("from_spike", False)),
                    transcripts_dir / vid,
                    config,
                )
                if row:
                    rows_by_id[vid] = row

        ordered_rows = [rows_by_id[str(v)] for v in subset_df["video_id"].astype(str) if str(v) in rows_by_id]
        write_verified_csv(ordered_rows, verified_csv)
        print_progress_summary(subset_df, transcripts_dir, limiter, target_total)

        gemini_n = sum(1 for r in ordered_rows if r.get("summary_source") == "gemini")
        logger.info(
            "Verified CSV -> %s | rows=%s gemini=%s extractive/other=%s",
            verified_csv,
            len(ordered_rows),
            gemini_n,
            len(ordered_rows) - gemini_n,
        )


if __name__ == "__main__":
    main()
