"""Live VTCF inference demo for a single YouTube URL."""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import torch
import torch.nn.functional as F
from colorama import Fore, Style, init as colorama_init
from PIL import Image
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.custom_dataset import default_image_transform, resolve_project_path
from data.ingestion import (
    build_youtube_url,
    download_video,
    fetch_video_metadata,
    sample_and_save_frames,
)
from models.fusion_network import VTCF, load_config
from models.interpretability import visualize_attention_storyboard
from scripts.evaluate import load_checkpoint_model

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "checkpoints" / "best_model_full.pt"
DEFAULT_COOKIES = PROJECT_ROOT / "data" / "youtube_cookies.txt"
SCENE_THRESHOLD = 27.0

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run live VTCF inference on a YouTube URL")
    parser.add_argument("--url", type=str, required=True, help="YouTube watch URL")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to trained VTCF checkpoint",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title override when yt-dlp metadata fetch fails",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use cached frames/title from the verified dataset instead of downloading",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Do not delete temporary download/extraction files",
    )
    return parser.parse_args()


def extract_video_id(url: str) -> str:
    """Extract a YouTube video ID from common URL formats."""
    parsed = urlparse(url.strip())
    if parsed.hostname in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.lstrip("/").split("/")[0]
        if video_id:
            return video_id

    if parsed.hostname in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        query_ids = parse_qs(parsed.query).get("v", [])
        if query_ids:
            return query_ids[0]
        match = re.match(r"^/(?:shorts|embed|live)/([^/?]+)", parsed.path)
        if match:
            return match.group(1)

    if re.fullmatch(r"[\w-]{11}", url.strip()):
        return url.strip()

    raise ValueError(f"Could not extract video_id from URL: {url}")


def lookup_title_from_csv(video_id: str, csv_path: Path) -> str | None:
    """Return a known title from the verified dataset CSV."""
    if not csv_path.exists():
        return None
    dataframe = pd.read_csv(csv_path)
    matches = dataframe[dataframe["video_id"].astype(str) == video_id]
    if matches.empty:
        return None
    title = matches.iloc[0].get("title")
    if pd.isna(title) or not str(title).strip():
        return None
    return str(title)


def resolve_cookies_file() -> Path | None:
    """Return cookies file path when present."""
    if DEFAULT_COOKIES.exists():
        return DEFAULT_COOKIES
    return None


def fetch_title(
    video_id: str,
    csv_path: Path,
    manual_title: str | None,
    offline: bool,
) -> str:
    """Resolve the video title from offline CSV, yt-dlp, or manual input."""
    if manual_title and manual_title.strip():
        return manual_title.strip()

    if offline:
        title = lookup_title_from_csv(video_id, csv_path)
        if title:
            return title
        raise RuntimeError(
            f"Offline mode enabled but no title found for {video_id} in {csv_path}"
        )

    title = lookup_title_from_csv(video_id, csv_path)
    if title:
        return title

    cookies_file = resolve_cookies_file()
    try:
        metadata = fetch_video_metadata(video_id, cookies_file=cookies_file)
        fetched_title = metadata.get("title")
        if fetched_title:
            return str(fetched_title)
    except Exception as exc:
        logger.warning("yt-dlp metadata fetch failed: %s", exc)

    try:
        entered = input("Enter video title manually: ").strip()
    except EOFError as exc:
        raise RuntimeError(
            "Could not fetch title automatically. Re-run with --title \"...\" or --offline."
        ) from exc

    if not entered:
        raise RuntimeError("A non-empty title is required for inference.")
    return entered


def prepare_frames(
    video_id: str,
    config: dict[str, Any],
    offline: bool,
    temp_root: Path,
) -> Path:
    """Download the video if needed and extract hook/context/delivery frames."""
    k_frames = int(config["model"]["K_frames"])
    frame_size = int(config["data"]["frame_size"])
    dataset_frames_dir = resolve_project_path(config["data"]["frames_dir"])
    output_dir = temp_root / video_id

    if offline:
        cached_dir = dataset_frames_dir / video_id
        if cached_dir.exists() and all(
            (cached_dir / f"frame_{index}.png").exists() for index in range(k_frames)
        ):
            return cached_dir
        raise RuntimeError(
            f"Offline mode enabled but cached frames not found at {cached_dir}"
        )

    cookies_file = resolve_cookies_file()
    _, success, message = download_video(
        video_id=video_id,
        temp_dir=temp_root,
        cookies_file=cookies_file,
    )
    if not success:
        cached_dir = dataset_frames_dir / video_id
        if cached_dir.exists():
            logger.warning("Download failed (%s); using cached frames.", message)
            return cached_dir
        raise RuntimeError(f"Video download failed: {message}")

    video_path = temp_root / f"{video_id}.mp4"
    if not video_path.exists():
        raise RuntimeError(f"Expected downloaded video at {video_path}")

    sample_and_save_frames(
        video_path=video_path,
        video_id=video_id,
        frames_dir=temp_root,
        k_frames=k_frames,
        frame_size=frame_size,
        threshold=SCENE_THRESHOLD,
        logger=logger,
    )
    return output_dir


def load_pixel_values(frames_dir: Path, k_frames: int) -> torch.Tensor:
    """Load K preprocessed frame tensors with shape [1, K, 3, 224, 224]."""
    transform = default_image_transform()
    pixel_values = torch.zeros(1, k_frames, 3, 224, 224, dtype=torch.float32)
    for index in range(k_frames):
        frame_path = frames_dir / f"frame_{index}.png"
        if not frame_path.exists():
            raise FileNotFoundError(f"Missing extracted frame: {frame_path}")
        image = Image.open(frame_path).convert("RGB")
        pixel_values[0, index] = transform(image)
    return pixel_values


def compute_alignment_scores(
    model: VTCF,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    temporal_visual_matrix: torch.Tensor,
) -> tuple[float, float, float]:
    """Compute cosine alignment between pooled text and each frame embedding."""
    text_features = model.text_encoder(input_ids, attention_mask)
    mask = attention_mask.unsqueeze(-1).float()
    text_repr = (text_features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    alignments: list[float] = []
    for frame_index in range(temporal_visual_matrix.shape[1]):
        frame_embedding = temporal_visual_matrix[:, frame_index, :]
        similarity = F.cosine_similarity(text_repr, frame_embedding, dim=-1)
        alignments.append(float(similarity.item()))

    while len(alignments) < 3:
        alignments.append(0.0)
    return alignments[0], alignments[1], alignments[2]


def build_explanation(hook_align: float, delivery_align: float) -> str:
    """Generate a short natural-language explanation."""
    hook_word = "relevant" if hook_align > 0.5 else "irrelevant"
    delivery_word = "maintains" if delivery_align > 0.4 else "diverges from"
    return (
        f"The video starts with {hook_word} visuals but {delivery_word} "
        "the headline promise by the delivery frame."
    )


@torch.no_grad()
def run_inference(
    config_path: Path,
    checkpoint_path: Path,
    url: str,
    manual_title: str | None = None,
    offline: bool = False,
    keep_temp: bool = False,
) -> dict[str, Any]:
    """Run the full live inference pipeline."""
    colorama_init(autoreset=True)
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_id = extract_video_id(url)
    csv_path = resolve_project_path(config["data"]["verified_csv"])
    temp_root = Path(tempfile.gettempdir()) / "vtcf_predict"
    temp_root.mkdir(parents=True, exist_ok=True)

    print("[INFO] Downloading Video Stream...")
    title = fetch_title(
        video_id=video_id,
        csv_path=csv_path,
        manual_title=manual_title,
        offline=offline,
    )

    print("[INFO] Extracting Frames: Hook (0%), Context (50%), Delivery (100%)")
    frames_dir = prepare_frames(
        video_id=video_id,
        config=config,
        offline=offline,
        temp_root=temp_root,
    )

    print(f'[INFO] Processing Headline: "{title}"')

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
    pixel_values = load_pixel_values(
        frames_dir=frames_dir,
        k_frames=int(config["model"]["K_frames"]),
    ).to(device)

    model = load_checkpoint_model(checkpoint_path, config, device, condition="full")
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
    verdict = "CLICKBAIT" if pred_index == 1 else "NOT CLICKBAIT"
    verdict_color = Fore.RED if pred_index == 1 else Fore.GREEN

    tds_tensor = outputs.get("tds_computed")
    tds = float(tds_tensor[0].item()) if tds_tensor is not None else float("nan")
    explanation = build_explanation(hook_align, delivery_align)

    print("\n=== VTCF INFERENCE RESULTS ===\n")
    print(f"- Text-to-Hook Alignment:     {hook_align:.4f}")
    print(f"- Text-to-Context Alignment:  {context_align:.4f}")
    print(f"- Text-to-Delivery Alignment: {delivery_align:.4f}\n")
    print(f"> Temporal Divergence Score (TDS): {tds:.4f} / 1.0\n")
    print(f"> Final Verdict: {verdict_color}{verdict}{Style.RESET_ALL}")
    print(f"  ({confidence:.1f}% Confidence)\n")
    print(f"> Explanation: {explanation}")

    storyboard_path = (
        PROJECT_ROOT / "outputs" / "visualizations" / f"predict_{video_id}.png"
    )
    visualize_attention_storyboard(
        video_id=video_id,
        frames_dir=frames_dir,
        attention_weights=outputs["attention_weights"][0],
        tds_score=tds,
        label=pred_index,
        prediction=pred_index,
        save_path=storyboard_path,
    )
    print(f"\nStoryboard saved -> {storyboard_path}")

    if not keep_temp and not offline:
        video_file = temp_root / f"{video_id}.mp4"
        frame_dir = temp_root / video_id
        if video_file.exists():
            video_file.unlink()
        if frame_dir.exists() and frame_dir != resolve_project_path(config["data"]["frames_dir"]):
            shutil.rmtree(frame_dir, ignore_errors=True)

    return {
        "video_id": video_id,
        "title": title,
        "verdict": verdict,
        "confidence": confidence,
        "tds": tds,
        "hook_align": hook_align,
        "context_align": context_align,
        "delivery_align": delivery_align,
        "storyboard_path": str(storyboard_path),
    }


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    run_inference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        url=args.url,
        manual_title=args.title,
        offline=args.offline,
        keep_temp=args.keep_temp,
    )


if __name__ == "__main__":
    main()
