"""Refresh cached example confidence values to 2 decimal places."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import CACHED_EXAMPLES_DIR, HARD_SUBSET_CSV, MANIFEST_PATH, TEMP_FRAMES_DIR, VTCF_RESEARCH_ROOT
from pipeline import analyze_youtube_url


def offline_dir(video_id: str) -> Path | None:
    for base in (
        CACHED_EXAMPLES_DIR / video_id,
        VTCF_RESEARCH_ROOT / "data" / "extracted_frames" / video_id,
    ):
        if all((base / f"frame_{i}.png").exists() for i in range(3)):
            return base
    return None


def hard_text_confidence(video_id: str) -> float | None:
    if not HARD_SUBSET_CSV.exists():
        return None
    df = pd.read_csv(HARD_SUBSET_CSV)
    matches = df[df["video_id"].astype(str) == video_id]
    if matches.empty:
        return None
    return round(float(matches.iloc[0]["confidence"]) * 100.0, 2)


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    for example in manifest["examples"]:
        video_id = example["video_id"]
        offline = offline_dir(video_id)
        url = f"https://www.youtube.com/watch?v={video_id}"
        print(f"Refreshing {video_id} (offline={offline is not None})…")

        result = analyze_youtube_url(
            url,
            frames_root=TEMP_FRAMES_DIR,
            include_text_only=False,
            offline_frames_dir=offline,
            manual_title=example.get("title"),
        )
        example["confidence"] = result["confidence"]

        if example.get("category") == "hard" and example.get("text_only"):
            text_conf = hard_text_confidence(video_id)
            if text_conf is not None:
                example["text_only"]["confidence"] = text_conf

        result_path = CACHED_EXAMPLES_DIR / video_id / "result.json"
        if result_path.exists():
            stored = json.loads(result_path.read_text(encoding="utf-8"))
            stored["confidence"] = example["confidence"]
            if stored.get("text_only") and example.get("text_only"):
                stored["text_only"]["confidence"] = example["text_only"]["confidence"]
            result_path.write_text(
                json.dumps(stored, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        text_only_conf = example.get("text_only", {}).get("confidence", "n/a")
        print(f"  Full VTCF: {example['confidence']}% | text_only: {text_only_conf}")

    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Updated {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
