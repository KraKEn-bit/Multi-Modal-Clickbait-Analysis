"""VTCF video ingestion pipeline: filter, audit, download, scene-detect, and frame extraction."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

HUMAN_ANNOTATION_COLUMNS = [
    "is_human_annotated",
    "human_label",
    "annotation_source",
    "label_source",
    "annotator",
]

FRAME_ROLE_NAMES = {
    0: "hook",
    1: "context",
    2: "delivery",
}


def resolve_video_cache_dir(config: dict[str, Any]) -> Path:
    """Return the directory for cached video downloads."""
    configured = config.get("data", {}).get("video_cache_dir")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return Path(tempfile.gettempdir()) / "vtcf_videos"


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load YAML configuration and resolve project-relative paths."""
    path = config_path or DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    for key in ("raw_parquet", "input_csv", "verified_csv", "frames_dir"):
        config["data"][key] = str(PROJECT_ROOT / config["data"][key])

    video_cache = config["data"].get("video_cache_dir")
    if video_cache:
        cache_path = Path(video_cache)
        config["data"]["video_cache_dir"] = str(
            cache_path if cache_path.is_absolute() else PROJECT_ROOT / cache_path
        )

    return config


def setup_logging(log_path: Path) -> logging.Logger:
    """Configure logging to console and a persistent log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("vtcf.ingestion")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def explore_dataset(df: pd.DataFrame) -> None:
    """Print exploratory summary of the raw dataset before any filtering."""
    print("\n" + "=" * 72)
    print("DATASET EXPLORATION")
    print("=" * 72)

    print("\n--- Column Names ---")
    for column in df.columns:
        print(f"  {column}")

    print("\n--- Dtypes ---")
    print(df.dtypes.to_string())

    label_like_columns = [
        column
        for column in df.columns
        if any(token in column.lower() for token in ("label", "annotated", "human"))
    ]
    print("\n--- Value Counts (label / annotated / human columns) ---")
    if label_like_columns:
        for column in label_like_columns:
            print(f"\n[{column}]")
            print(df[column].value_counts(dropna=False).to_string())
    else:
        print("  No matching columns found.")

    print("\n--- Non-Null Counts ---")
    print(df.notna().sum().to_string())

    print("\n--- First 3 Rows ---")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        preview = df.head(3).to_string(index=False)
        try:
            print(preview)
        except UnicodeEncodeError:
            print(preview.encode("ascii", errors="replace").decode("ascii"))

    print("=" * 72 + "\n")


def _find_human_annotation_column(df: pd.DataFrame) -> str:
    """Locate the human-annotation column using known names and fuzzy fallbacks."""
    for candidate in HUMAN_ANNOTATION_COLUMNS:
        if candidate in df.columns:
            return candidate

    fuzzy_matches = [
        column
        for column in df.columns
        if "human" in column.lower()
        and ("label" in column.lower() or "annot" in column.lower())
    ]
    if len(fuzzy_matches) == 1:
        return fuzzy_matches[0]
    if len(fuzzy_matches) > 1:
        return fuzzy_matches[0]

    print("Available columns:", list(df.columns))
    raise ValueError(
        "No human annotation column found. Tried: "
        f"{HUMAN_ANNOTATION_COLUMNS}. "
        "Please specify the correct column name manually."
    )


def filter_human_annotated(
    df: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Filter rows to human-annotated samples and persist the subset CSV."""
    column = _find_human_annotation_column(df)
    logger.info("Using human annotation column: %s", column)

    series = df[column]
    if column == "is_human_annotated":
        mask = series.fillna(False).astype(bool)
    elif column in {"annotation_source", "label_source", "annotator"}:
        mask = series.astype(str).str.lower().str.contains("human", na=False)
    else:
        mask = series.notna() & (series.astype(str).str.strip() != "")

    filtered = df.loc[mask].copy()
    total_rows = len(df)
    filtered_rows = len(filtered)

    logger.info(
        "Filtered to %s human-annotated rows from %s total rows.",
        filtered_rows,
        total_rows,
    )
    print(f"Filtered to {filtered_rows} human-annotated rows from {total_rows} total rows.")
    return filtered


def normalize_detection_label(value: Any) -> str:
    """Map raw annotation strings to canonical detection labels."""
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    if "not" in text and "clickbait" in text:
        return "non_clickbait"
    if "clickbait" in text:
        return "clickbait"
    return str(value).strip()


def resolve_label_columns(df: pd.DataFrame) -> tuple[str, str | None]:
    """Return (detection_label_col, tactic_label_col) from available columns."""
    detection_col = None
    for candidate in ("human_labeled", "human_label", "label"):
        if candidate in df.columns:
            detection_col = candidate
            break
    if detection_col is None:
        detection_col = _find_human_annotation_column(df)

    tactic_col = None
    for candidate in ("tactic_label", "attribution_label", "tactic", "attribution"):
        if candidate in df.columns:
            tactic_col = candidate
            break

    return detection_col, tactic_col


def build_youtube_url(video_id: str) -> str:
    """Construct a canonical YouTube watch URL from a video ID."""
    return f"https://www.youtube.com/watch?v={video_id}"


def build_ytdlp_command(
    url: str,
    cookies_browser: str | None = None,
    cookies_file: Path | str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build a yt-dlp command with optional browser or file-based cookies."""
    command = [
        "yt-dlp",
        # YouTube requires EJS challenge solving (see yt-dlp wiki/EJS).
        "--js-runtimes",
        "node",
    ]
    if cookies_file:
        command.extend(["--cookies", str(cookies_file)])
    elif cookies_browser:
        command.extend(["--cookies-from-browser", cookies_browser])
    if extra_args:
        command.extend(extra_args)
    command.append(url)
    return command


def audit_video_url(
    video_id: str,
    cookies_browser: str | None = None,
    cookies_file: Path | str | None = None,
) -> dict[str, str]:
    """Simulate a yt-dlp fetch to determine whether a video is accessible."""
    url = build_youtube_url(video_id)
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        command = build_ytdlp_command(
            url,
            cookies_browser=cookies_browser,
            cookies_file=cookies_file,
            extra_args=["--simulate", "--quiet"],
        )
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return {
                "video_id": video_id,
                "status": "live",
                "error_message": "",
                "timestamp": timestamp,
            }

        error_message = (result.stderr or result.stdout or "non-zero return code").strip()
        return {
            "video_id": video_id,
            "status": "dead",
            "error_message": error_message,
            "timestamp": timestamp,
        }
    except (subprocess.SubprocessError, OSError) as exc:
        return {
            "video_id": video_id,
            "status": "dead",
            "error_message": str(exc),
            "timestamp": timestamp,
        }


def audit_urls(
    df: pd.DataFrame,
    audit_log_path: Path,
    max_workers: int,
    logger: logging.Logger,
    cookies_browser: str | None = None,
    cookies_file: Path | str | None = None,
) -> pd.DataFrame:
    """Audit all video URLs concurrently and write an audit log CSV."""
    video_ids = df["video_id"].astype(str).tolist()
    logger.info("Starting URL audit for %s videos with %s workers.", len(video_ids), max_workers)

    records: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                audit_video_url,
                video_id,
                cookies_browser,
                cookies_file,
            ): video_id
            for video_id in video_ids
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="URL audit"):
            records.append(future.result())

    audit_df = pd.DataFrame(records)
    audit_df.to_csv(audit_log_path, index=False)
    logger.info("Audit log saved to %s", audit_log_path)

    live_count = int((audit_df["status"] == "live").sum())
    total_count = len(audit_df)
    survival_rate = (live_count / total_count * 100.0) if total_count else 0.0
    summary = f"{live_count}/{total_count} videos live ({survival_rate:.1f}%)"
    logger.info(summary)
    print(summary)

    return audit_df


def fetch_video_metadata(
    video_id: str,
    cookies_browser: str | None = None,
    cookies_file: Path | str | None = None,
) -> dict[str, Any]:
    """Fetch yt-dlp JSON metadata for a YouTube video."""
    import json

    url = build_youtube_url(video_id)
    command = build_ytdlp_command(
        url,
        cookies_browser=cookies_browser,
        cookies_file=cookies_file,
        extra_args=["--dump-json", "--skip-download", "--quiet"],
    )
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            (result.stderr or result.stdout or "yt-dlp metadata fetch failed").strip()
        )
    return json.loads(result.stdout)


def download_video(
    video_id: str,
    temp_dir: Path,
    cookies_browser: str | None = None,
    cookies_file: Path | str | None = None,
    sleep_seconds: float = 0.0,
) -> tuple[str, bool, str]:
    """Download a single video at <=360p if not already cached locally."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_path = temp_dir / f"{video_id}.mp4"

    if output_path.exists() and output_path.stat().st_size > 0:
        return video_id, True, "skipped_existing"

    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    url = build_youtube_url(video_id)
    command = build_ytdlp_command(
        url,
        cookies_browser=cookies_browser,
        cookies_file=cookies_file,
        extra_args=[
            "-f",
            "bestvideo[height<=360][ext=mp4]/best[height<=360]",
            "-o",
            str(output_path),
            "--quiet",
            "--no-progress",
            "--sleep-interval",
            "1",
            "--max-sleep-interval",
            "5",
        ],
    )

    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and output_path.exists():
            return video_id, True, "downloaded"

        error_message = (result.stderr or result.stdout or "download failed").strip()
        return video_id, False, error_message
    except (subprocess.SubprocessError, OSError) as exc:
        return video_id, False, str(exc)


def download_videos(
    video_ids: list[str],
    temp_dir: Path,
    max_workers: int,
    logger: logging.Logger,
    cookies_browser: str | None = None,
    cookies_file: Path | str | None = None,
    sleep_seconds: float = 0.0,
) -> dict[str, Path]:
    """Download live videos concurrently and return successful local paths."""
    logger.info(
        "Starting downloads for %s videos with %s workers.",
        len(video_ids),
        max_workers,
    )

    downloaded: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                download_video,
                video_id,
                temp_dir,
                cookies_browser,
                cookies_file,
                sleep_seconds if max_workers == 1 else 0.0,
            ): video_id
            for video_id in video_ids
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            video_id, success, message = future.result()
            output_path = temp_dir / f"{video_id}.mp4"
            if success and output_path.exists():
                downloaded[video_id] = output_path
                logger.info("Video %s ready (%s).", video_id, message)
            else:
                logger.warning("Video %s download failed: %s", video_id, message)

    return downloaded


def get_video_duration_seconds(video_path: Path) -> float:
    """Return total video duration in seconds using OpenCV metadata."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return 0.0

    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()

    if fps <= 0:
        return 0.0
    return float(frame_count / fps)


def clamp_timestamp(
    timestamp_sec: float,
    duration_sec: float,
    margin_sec: float = 0.5,
) -> float:
    """Clamp a seek timestamp to a readable in-bounds position."""
    if duration_sec <= 0:
        return max(timestamp_sec, 0.0)
    upper_bound = max(duration_sec - margin_sec, 0.0)
    return min(max(timestamp_sec, 0.0), upper_bound)


def normalize_timestamps(
    timestamps: list[float],
    duration_sec: float,
    margin_sec: float = 0.5,
) -> list[float]:
    """Clamp all sampling timestamps to readable positions within the video."""
    return [clamp_timestamp(ts, duration_sec, margin_sec) for ts in timestamps]


def detect_scene_timestamps(
    video_path: Path,
    k_frames: int,
    threshold: float,
    logger: logging.Logger,
) -> tuple[list[float], str]:
    """Detect scene boundaries and derive K sampling timestamps."""
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    if len(scene_list) >= k_frames:
        candidate_times: list[float] = [scene[0].get_seconds() for scene in scene_list]
        candidate_times.append(scene_list[-1][1].get_seconds())

        if k_frames == 1:
            timestamps = [candidate_times[0]]
        else:
            indices = np.linspace(0, len(candidate_times) - 1, num=k_frames)
            timestamps = [candidate_times[int(round(index))] for index in indices]

        duration = get_video_duration_seconds(video_path)
        return normalize_timestamps(timestamps, duration), "scene_based"

    duration = get_video_duration_seconds(video_path)
    if duration <= 0:
        logger.warning("Could not determine duration for %s; defaulting to t=0.", video_path)
        return [0.0] * k_frames, "fallback_zero"

    margin = 0.5
    readable_duration = max(duration - margin, 0.0)
    if k_frames == 1:
        timestamps = [0.0]
    else:
        timestamps = [
            readable_duration * index / (k_frames - 1) for index in range(k_frames)
        ]

    logger.info(
        "Fewer than %s scenes in %s; using uniform temporal sampling.",
        k_frames,
        video_path.name,
    )
    return timestamps, "uniform_fallback"


def extract_frame_at_timestamp(
    video_path: Path,
    timestamp_sec: float,
    duration_sec: float | None = None,
) -> np.ndarray | None:
    """Seek to a timestamp and return a BGR frame array."""
    if duration_sec is None:
        duration_sec = get_video_duration_seconds(video_path)

    seek_times = [
        clamp_timestamp(timestamp_sec, duration_sec),
        clamp_timestamp(duration_sec - 1.0, duration_sec),
    ]
    seen: set[float] = set()

    for seek_time in seek_times:
        if seek_time in seen:
            continue
        seen.add(seek_time)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            continue

        capture.set(cv2.CAP_PROP_POS_MSEC, seek_time * 1000.0)
        success, frame = capture.read()

        if not success or frame is None:
            fps = capture.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                frame_index = max(int(seek_time * fps), 0)
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()

        capture.release()
        if success and frame is not None:
            return frame

    return None


def save_resized_frame(
    frame_bgr: np.ndarray,
    output_path: Path,
    frame_size: int,
) -> None:
    """Convert BGR to RGB, resize, and persist as PNG."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    image = image.resize((frame_size, frame_size), Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def sample_and_save_frames(
    video_path: Path,
    video_id: str,
    frames_dir: Path,
    k_frames: int,
    frame_size: int,
    threshold: float,
    logger: logging.Logger,
) -> Path:
    """Detect scenes, sample K frames, resize, and save PNGs for one video."""
    timestamps, strategy = detect_scene_timestamps(
        video_path=video_path,
        k_frames=k_frames,
        threshold=threshold,
        logger=logger,
    )
    duration_sec = get_video_duration_seconds(video_path)
    output_dir = frames_dir / video_id

    for index, timestamp in enumerate(timestamps):
        frame = extract_frame_at_timestamp(
            video_path,
            timestamp,
            duration_sec=duration_sec,
        )
        if frame is None:
            logger.warning(
                "Failed to extract frame %s for %s at %.2fs.",
                index,
                video_id,
                timestamp,
            )
            continue

        role = FRAME_ROLE_NAMES.get(index, f"frame_{index}")
        logger.info(
            "Saving %s frame_%s.png (%s) at %.2fs using %s sampling.",
            video_id,
            index,
            role,
            timestamp,
            strategy,
        )
        save_resized_frame(frame, output_dir / f"frame_{index}.png", frame_size)

    return output_dir


def compute_tds_score(frame_dir: Path, k_frames: int) -> float:
    """Compute temporal divergence score between first and last extracted frames."""
    first_path = frame_dir / "frame_0.png"
    last_path = frame_dir / f"frame_{k_frames - 1}.png"

    if not first_path.exists() or not last_path.exists():
        return float("nan")

    first = np.array(Image.open(first_path).convert("RGB"), dtype=np.float32).flatten()
    last = np.array(Image.open(last_path).convert("RGB"), dtype=np.float32).flatten()

    first_norm = first / (np.linalg.norm(first) + 1e-8)
    last_norm = last / (np.linalg.norm(last) + 1e-8)
    cosine_similarity = float(np.dot(first_norm, last_norm))
    return 1.0 - cosine_similarity


def frames_already_extracted(frame_dir: Path, k_frames: int) -> bool:
    """Return True when all K frame PNGs exist for a video."""
    return all((frame_dir / f"frame_{index}.png").exists() for index in range(k_frames))


def find_cached_video(video_id: str, video_cache_dir: Path) -> Path | None:
    """Locate an already-downloaded mp4 in the cache or legacy temp folder."""
    legacy_temp_dir = Path(tempfile.gettempdir()) / "vtcf_videos"
    for directory in (video_cache_dir, legacy_temp_dir):
        candidate = directory / f"{video_id}.mp4"
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def process_videos(
    df: pd.DataFrame,
    audit_df: pd.DataFrame,
    config: dict[str, Any],
    k_frames: int,
    download_workers: int,
    logger: logging.Logger,
    video_cache_dir: Path,
    cookies_browser: str | None = None,
    cookies_file: Path | str | None = None,
    sleep_seconds: float = 0.0,
    delete_after_extract: bool = True,
) -> pd.DataFrame:
    """Download live videos, extract frames, optionally delete mp4s, compute TDS."""
    if delete_after_extract and download_workers > 1:
        logger.warning(
            "Forcing download-workers=1 because mp4s are deleted after frame extraction."
        )
        download_workers = 1

    detection_col, tactic_col = resolve_label_columns(df)
    frames_dir = Path(config["data"]["frames_dir"])
    frame_size = int(config["data"]["frame_size"])
    threshold = 27.0
    video_cache_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
        video_id = str(row["video_id"])
        audit_status = audit_df.loc[
            audit_df["video_id"] == video_id, "status"
        ].iloc[0] if (audit_df["video_id"] == video_id).any() else "unknown"

        title = row.get("title", "")
        label = normalize_detection_label(row.get(detection_col))
        tactic_label = row.get(tactic_col, "") if tactic_col else ""
        if pd.isna(tactic_label):
            tactic_label = ""

        frame_dir = frames_dir / video_id
        tds_score = float("nan")

        if audit_status == "live":
            if frames_already_extracted(frame_dir, k_frames):
                tds_score = compute_tds_score(frame_dir, k_frames)
                logger.info("Video %s skipped (frames exist), TDS=%.4f", video_id, tds_score)
            else:
                video_path = find_cached_video(video_id, video_cache_dir)
                if video_path is None:
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
                    _, success, message = download_video(
                        video_id,
                        video_cache_dir,
                        cookies_browser=cookies_browser,
                        cookies_file=cookies_file,
                    )
                    video_path = video_cache_dir / f"{video_id}.mp4"
                    if not success or not video_path.exists():
                        logger.warning("Video %s download failed: %s", video_id, message)
                        video_path = None
                    else:
                        logger.info("Video %s downloaded.", video_id)

                if video_path is not None and video_path.exists():
                    frame_dir = sample_and_save_frames(
                        video_path=video_path,
                        video_id=video_id,
                        frames_dir=frames_dir,
                        k_frames=k_frames,
                        frame_size=frame_size,
                        threshold=threshold,
                        logger=logger,
                    )
                    tds_score = compute_tds_score(frame_dir, k_frames)
                    logger.info("Video %s TDS=%.4f", video_id, tds_score)
                    if delete_after_extract:
                        try:
                            video_path.unlink(missing_ok=True)
                            logger.info("Deleted temp mp4 for %s", video_id)
                        except OSError as exc:
                            logger.warning(
                                "Could not delete %s: %s", video_path, exc
                            )

        results.append(
            {
                "video_id": video_id,
                "title": title,
                "label": label,
                "tactic_label": tactic_label,
                "tds_score": tds_score,
                "frame_dir": str(frame_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "audit_status": audit_status,
            }
        )

    return pd.DataFrame(results)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for ingestion pipeline overrides."""
    parser = argparse.ArgumentParser(description="VTCF data ingestion pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Process only the first 5 filtered rows",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Override number of frames to sample per video",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override audit thread count",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=1,
        help="Parallel download workers (default: 1 to avoid YouTube 429 rate limits)",
    )
    parser.add_argument(
        "--cookies-from-browser",
        type=str,
        default=None,
        help="Browser for yt-dlp cookies, e.g. chrome or edge (fully quit browser first)",
    )
    parser.add_argument(
        "--cookies-file",
        type=Path,
        default=None,
        help="Path to exported cookies.txt (preferred on Windows over --cookies-from-browser)",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=2.0,
        help="Delay between downloads when download-workers=1",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Reuse existing data/audit_log.csv and skip URL audit",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Override video download cache directory (default: config data.video_cache_dir)",
    )
    parser.add_argument(
        "--keep-videos",
        action="store_true",
        help="Keep mp4 files after frame extraction (uses much more disk space)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full VTCF ingestion pipeline."""
    args = parse_args()
    config = load_config(args.config)

    log_path = PROJECT_ROOT / "data" / "ingestion.log"
    logger = setup_logging(log_path)

    k_frames = args.k if args.k is not None else int(config["model"]["K_frames"])
    audit_workers = args.workers if args.workers is not None else 8
    download_workers = args.download_workers

    raw_parquet = Path(config["data"]["raw_parquet"])
    input_csv = Path(config["data"]["input_csv"])
    verified_csv = Path(config["data"]["verified_csv"])
    audit_log_path = PROJECT_ROOT / "data" / "audit_log.csv"

    if args.skip_audit and audit_log_path.exists() and input_csv.exists():
        logger.info("Skipping parquet filter/audit; reusing %s and %s", input_csv, audit_log_path)
        working_df = pd.read_csv(input_csv)
        audit_df = pd.read_csv(audit_log_path)
        if args.test_mode:
            working_df = working_df.head(5).copy()
            audit_df = audit_df[audit_df["video_id"].isin(working_df["video_id"])].copy()
            print("=== RUNNING IN TEST MODE: 5 videos only ===")
    else:
        logger.info("Loading raw parquet from %s", raw_parquet)
        raw_df = pd.read_parquet(raw_parquet)

        explore_dataset(raw_df)

        filtered_df = filter_human_annotated(raw_df, logger)
        input_csv.parent.mkdir(parents=True, exist_ok=True)
        filtered_df.to_csv(input_csv, index=False)
        logger.info("Saved filtered dataset to %s", input_csv)

        working_df = filtered_df.copy()
        if args.test_mode:
            working_df = working_df.head(5).copy()
            banner = "=== RUNNING IN TEST MODE: 5 videos only ==="
            print(banner)
            logger.info(banner)

        audit_df = audit_urls(
            df=working_df,
            audit_log_path=audit_log_path,
            max_workers=audit_workers,
            logger=logger,
            cookies_browser=args.cookies_from_browser,
            cookies_file=args.cookies_file,
        )

    if args.cookies_file:
        logger.info("Using cookies file: %s", args.cookies_file)
        print(f"Using cookies file: {args.cookies_file}")
    elif args.cookies_from_browser:
        logger.info("Using browser cookies from: %s", args.cookies_from_browser)
        print(f"Using cookies from browser: {args.cookies_from_browser}")

    video_cache_dir = (
        args.video_dir.expanduser()
        if args.video_dir
        else Path(config["data"].get("video_cache_dir") or resolve_video_cache_dir(config))
    )
    video_cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Video cache directory: %s", video_cache_dir)
    print(f"Video cache: {video_cache_dir}")
    if not args.keep_videos:
        print("Temp mp4s will be deleted after frame extraction (~2 GB total disk use).")

    verified_df = process_videos(
        df=working_df,
        audit_df=audit_df,
        config=config,
        k_frames=k_frames,
        download_workers=download_workers,
        logger=logger,
        video_cache_dir=video_cache_dir,
        cookies_browser=args.cookies_from_browser,
        cookies_file=args.cookies_file,
        sleep_seconds=args.sleep_seconds,
        delete_after_extract=not args.keep_videos,
    )

    verified_csv.parent.mkdir(parents=True, exist_ok=True)
    verified_df.to_csv(verified_csv, index=False)
    logger.info("Saved verified live videos to %s", verified_csv)
    print(f"Pipeline complete. Verified output: {verified_csv}")


if __name__ == "__main__":
    main()
