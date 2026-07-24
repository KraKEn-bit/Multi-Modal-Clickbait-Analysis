"""Pre-compute cached demo examples for GET /examples."""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import (  # noqa: E402
    CACHED_EXAMPLES_DIR,
    HARD_SUBSET_CSV,
    MANIFEST_PATH,
    VTCF_RESEARCH_ROOT,
)
from pipeline import analyze_youtube_url  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Known-good demo videos (verified in prior inference test)
OBVIOUS_CLICKBAIT = {
    "video_id": "pYganyZsHYM",
    "category": "clickbait",
    "label": "Obvious clickbait",
    "description": "Sensational English headline on viral-style footage.",
}

OBVIOUS_GENUINE = {
    "video_id": "hcFpC8R6c24",
    "category": "genuine",
    "label": "Obvious genuine",
    "description": "Independent TV news bulletin — straightforward headline.",
}

# Hard cases from BanglaBERT failure set (frames exist in vtcf-research)
HARD_CASES = [
    {
        "video_id": "OoUO4vjgM4c",
        "category": "hard",
        "label": "Hard case — breaking news title, factual bulletin",
        "description": "Human label: genuine. BanglaBERT title-only flags clickbait; VTCF reads the newsroom visuals correctly.",
    },
    {
        "video_id": "DhESX8gA7wk",
        "category": "hard",
        "label": "Hard case — sensational wedding headline",
        "description": "Human label: clickbait. BanglaBERT title-only misses it; VTCF catches the mismatch from video frames.",
    },
]

BLOCKED_IDS = {"sSgSnAtoiTg", "PcTfIyPHNVg"}


def _load_hard_subset_row(video_id: str) -> dict | None:
    if not HARD_SUBSET_CSV.exists():
        return None
    df = pd.read_csv(HARD_SUBSET_CSV)
    matches = df[df["video_id"].astype(str) == video_id]
    if matches.empty:
        return None
    row = matches.iloc[0]
    true_label = "GENUINE" if int(row["label"]) == 0 else "CLICKBAIT"
    text_pred = "CLICKBAIT" if int(row["predicted_label"]) == 1 else "GENUINE"
    text_conf = round(float(row["confidence"]) * 100.0, 2)
    return {
        "true_label": true_label,
        "text_only_verdict": text_pred,
        "text_only_confidence": text_conf,
        "text_only_wrong": text_pred != true_label,
    }


def _offline_frames_dir(video_id: str) -> Path | None:
    frames_dir = VTCF_RESEARCH_ROOT / "data" / "extracted_frames" / video_id
    if all((frames_dir / f"frame_{i}.png").exists() for i in range(3)):
        return frames_dir
    return None


def cache_one(spec: dict) -> dict:
    video_id = spec["video_id"]
    if video_id in BLOCKED_IDS:
        raise RuntimeError(f"{video_id} is known to be blocked by yt-dlp")

    url = f"https://www.youtube.com/watch?v={video_id}"
    dest_dir = CACHED_EXAMPLES_DIR / video_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    offline = _offline_frames_dir(video_id)
    logger.info("Caching %s (offline=%s)", video_id, offline is not None)

    result = analyze_youtube_url(
        url,
        frames_root=CACHED_EXAMPLES_DIR,
        include_text_only=spec.get("category") == "hard",
        offline_frames_dir=offline,
    )

    hard_meta = _load_hard_subset_row(video_id)
    if hard_meta:
        result["ground_truth"] = hard_meta["true_label"]
        result["text_only"] = {
            "verdict": hard_meta["text_only_verdict"],
            "confidence": hard_meta["text_only_confidence"],
            "wrong": hard_meta["text_only_wrong"],
        }
        result["vtcf_rescued"] = (
            hard_meta["text_only_wrong"]
            and result["verdict"] == hard_meta["true_label"]
        )

    result["category"] = spec["category"]
    result["example_label"] = spec["label"]
    result["example_description"] = spec["description"]
    result["frame_urls"] = [
        f"/cached-frames/{video_id}/frame_{i}.png" for i in range(3)
    ]

    with (dest_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    return result


def main() -> None:
    CACHED_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    specs = [OBVIOUS_CLICKBAIT, OBVIOUS_GENUINE, *HARD_CASES]
    examples: list[dict] = []

    for spec in specs:
        try:
            examples.append(cache_one(spec))
        except Exception as exc:
            logger.error("Failed to cache %s: %s", spec["video_id"], exc)
            raise

    manifest = {
        "examples": examples,
        "research_note": (
            "Clickbait videos show LESS visual change on average (TDS ~0.38) than "
            "genuine videos (~0.64) — clickbait often reuses static footage rather "
            "than bait-and-switching content."
        ),
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    logger.info("Saved %d examples to %s", len(examples), MANIFEST_PATH)


if __name__ == "__main__":
    main()
