"""VTCF inference pipeline — reuses vtcf-research code without modification."""

from __future__ import annotations

import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import (
    CHECKPOINT_FULL,
    CHECKPOINT_TEXT_ONLY,
    CONFIG_PATH,
    TEMP_FRAMES_DIR,
    VTCF_RESEARCH_ROOT,
)

logger = logging.getLogger(__name__)

if str(VTCF_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(VTCF_RESEARCH_ROOT))

from data.custom_dataset import default_image_transform, resolve_project_path  # noqa: E402
from data.ingestion import download_video, fetch_video_metadata, sample_and_save_frames  # noqa: E402
from models.fusion_network import load_config  # noqa: E402
from scripts.evaluate import load_checkpoint_model  # noqa: E402
from scripts.predict import (  # noqa: E402
    SCENE_THRESHOLD,
    build_explanation,
    compute_alignment_scores,
    extract_video_id,
    fetch_title,
    load_pixel_values,
    resolve_cookies_file,
)

extract_video_id_from_url = extract_video_id

ProgressCallback = Callable[[str, str], None]


def _noop_progress(stage: str, message: str) -> None:
    del stage, message


def _copy_frames(source_dir: Path, dest_dir: Path, k_frames: int = 3) -> list[str]:
    """Copy frame PNGs to dest_dir and return relative URL paths."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    frame_urls: list[str] = []
    for index in range(k_frames):
        src = source_dir / f"frame_{index}.png"
        dst = dest_dir / f"frame_{index}.png"
        if not src.exists():
            raise FileNotFoundError(f"Missing frame: {src}")
        shutil.copy2(src, dst)
        frame_urls.append(f"/frames/{dest_dir.name}/frame_{index}.png")
    return frame_urls


def _run_text_only(
    title: str,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """BanglaBERT title-only baseline prediction."""
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_encoder"])
    encoding = tokenizer(
        title,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    model = load_checkpoint_model(CHECKPOINT_TEXT_ONLY, config, device, condition="text_only")
    outputs = model.forward_text_only(input_ids, attention_mask)
    logits = outputs["detection_logits"]
    probs = torch.softmax(logits, dim=-1)[0]
    pred_index = int(torch.argmax(probs).item())
    confidence = float(probs[pred_index].item()) * 100.0
    verdict = "CLICKBAIT" if pred_index == 1 else "GENUINE"
    return {
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "clickbait_probability": round(float(probs[1].item()) * 100.0, 2),
    }


@torch.no_grad()
def analyze_youtube_url(
    youtube_url: str,
    *,
    frames_root: Path = TEMP_FRAMES_DIR,
    include_text_only: bool = False,
    offline_frames_dir: Path | None = None,
    manual_title: str | None = None,
    on_progress: ProgressCallback = _noop_progress,
) -> dict[str, Any]:
    """
    Full VTCF pipeline: download → extract frames → fuse → verdict.

    If offline_frames_dir is set, skip download and use those frames (for hard-subset cache).
    """
    started = time.perf_counter()
    config = load_config(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k_frames = int(config["model"]["K_frames"])
    frame_size = int(config["data"]["frame_size"])

    video_id = extract_video_id(youtube_url)
    csv_path = resolve_project_path(config["data"]["verified_csv"])
    work_dir = frames_root / video_id
    download_dir = frames_root / "_downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    on_progress("downloading", "Fetching video stream from YouTube…")
    title = fetch_title(
        video_id=video_id,
        csv_path=csv_path,
        manual_title=manual_title,
        offline=offline_frames_dir is not None,
    )

    if offline_frames_dir is not None:
        on_progress("extracting", "Loading cached hook / context / delivery frames…")
        if not all((offline_frames_dir / f"frame_{i}.png").exists() for i in range(k_frames)):
            raise RuntimeError(f"Offline frames incomplete at {offline_frames_dir}")
        work_dir.mkdir(parents=True, exist_ok=True)
        for index in range(k_frames):
            shutil.copy2(
                offline_frames_dir / f"frame_{index}.png",
                work_dir / f"frame_{index}.png",
            )
        frames_dir = work_dir
    else:
        cookies_file = resolve_cookies_file()
        _, success, message = download_video(
            video_id=video_id,
            temp_dir=download_dir,
            cookies_file=cookies_file,
        )
        if not success:
            dataset_frames = resolve_project_path(config["data"]["frames_dir"]) / video_id
            if dataset_frames.exists() and all(
                (dataset_frames / f"frame_{i}.png").exists() for i in range(k_frames)
            ):
                logger.warning("Download failed (%s); using research cache.", message)
                work_dir.mkdir(parents=True, exist_ok=True)
                for index in range(k_frames):
                    shutil.copy2(
                        dataset_frames / f"frame_{index}.png",
                        work_dir / f"frame_{index}.png",
                    )
                frames_dir = work_dir
            else:
                raise RuntimeError(f"Video download failed: {message}")
        else:
            on_progress("detecting", "Detecting scene boundaries with PySceneDetect…")
            video_path = download_dir / f"{video_id}.mp4"
            if not video_path.exists():
                raise RuntimeError(f"Expected downloaded video at {video_path}")

            on_progress("extracting", "Extracting hook, context, and delivery frames…")
            sample_and_save_frames(
                video_path=video_path,
                video_id=video_id,
                frames_dir=frames_root,
                k_frames=k_frames,
                frame_size=frame_size,
                threshold=SCENE_THRESHOLD,
                logger=logger,
            )
            frames_dir = frames_root / video_id
            if video_path.exists():
                video_path.unlink(missing_ok=True)

    on_progress("inferring", "Running BanglaBERT + ViT fusion model…")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_encoder"])
    encoding = tokenizer(
        title,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    pixel_values = load_pixel_values(frames_dir=frames_dir, k_frames=k_frames).to(device)

    model = load_checkpoint_model(CHECKPOINT_FULL, config, device, condition="full")
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
    )

    hook_align, context_align, delivery_align = compute_alignment_scores(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        temporal_visual_matrix=outputs["temporal_visual_matrix"],
    )

    logits = outputs["detection_logits"]
    probs = torch.softmax(logits, dim=-1)[0]
    pred_index = int(torch.argmax(probs).item())
    confidence = float(probs[pred_index].item()) * 100.0
    verdict = "CLICKBAIT" if pred_index == 1 else "GENUINE"

    tds_tensor = outputs.get("tds_computed")
    tds = float(tds_tensor[0].item()) if tds_tensor is not None else float("nan")
    explanation = build_explanation(hook_align, delivery_align)

    frame_urls = [
        f"/frames/{video_id}/frame_{index}.png" for index in range(k_frames)
    ]

    elapsed = time.perf_counter() - started

    result: dict[str, Any] = {
        "video_id": video_id,
        "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
        "title": title,
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "tds_score": round(tds, 4),
        "explanation": explanation,
        "alignment_scores": {
            "hook": round(hook_align, 4),
            "context": round(context_align, 4),
            "delivery": round(delivery_align, 4),
        },
        "frame_urls": frame_urls,
        "processing_time_seconds": round(elapsed, 1),
    }

    if include_text_only:
        on_progress("text_only", "Running BanglaBERT title-only baseline…")
        result["text_only"] = _run_text_only(title, config, device)

    on_progress("done", "Analysis complete.")
    return result
