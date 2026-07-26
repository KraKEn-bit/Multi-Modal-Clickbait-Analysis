"""App backend configuration — paths to research repos (read-only reuse)."""

from __future__ import annotations

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = APP_ROOT / "backend"
VTCF_RESEARCH_ROOT = APP_ROOT.parent / "vtcf-research"

TEMP_FRAMES_DIR = BACKEND_ROOT / "temp_frames"
CACHED_EXAMPLES_DIR = BACKEND_ROOT / "cached_examples"
MANIFEST_PATH = CACHED_EXAMPLES_DIR / "manifest.json"
HARD_SUBSET_CSV = (
    VTCF_RESEARCH_ROOT
    / "data"
    / "baseline_banglabert_model"
    / "hard_subset_video_ids.csv"
)

CONFIG_PATH = VTCF_RESEARCH_ROOT / "config.yaml"
CHECKPOINT_FULL = VTCF_RESEARCH_ROOT / "outputs" / "checkpoints" / "best_model_full.pt"
CHECKPOINT_TEXT_ONLY = (
    VTCF_RESEARCH_ROOT / "outputs" / "checkpoints" / "best_model_text_only.pt"
)
COOKIES_FILE = VTCF_RESEARCH_ROOT / "data" / "youtube_cookies.txt"

ANALYZE_TIMEOUT_SECONDS = int(os.getenv("VTCF_ANALYZE_TIMEOUT", "300"))
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
