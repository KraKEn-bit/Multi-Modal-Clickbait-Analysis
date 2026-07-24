"""Shared config and path helpers for vtcf-finding2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load YAML config and resolve project-relative paths."""
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    for key in (
        "verified_csv",
        "frames_dir",
        "youtube_cookies",
        "spike_csv",
        "spike_transcripts_dir",
        "spike_audio_dir",
        "video_cache_dir",
        "transcripts_dir",
        "finding2_subset_csv",
        "finding2_verified_csv",
        "llm_call_log",
        "phase1_audio_dir",
    ):
        if key in config.get("data", {}):
            config["data"][key] = str(resolve_path(config["data"][key]))

    if "vtcf_research_root" in config:
        config["vtcf_research_root"] = str(resolve_path(config["vtcf_research_root"]))

    return config


def resolve_path(path: Path | str) -> Path:
    """Resolve a path relative to the finding-2 project root."""
    path = Path(path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def ensure_dirs(config: dict[str, Any]) -> None:
    """Create output and temp directories."""
    for key in (
        "spike_audio_dir",
        "spike_transcripts_dir",
        "video_cache_dir",
        "transcripts_dir",
        "phase1_audio_dir",
    ):
        resolve_path(config["data"][key]).mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "outputs" / "spike_results").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "outputs" / "logs").mkdir(parents=True, exist_ok=True)
