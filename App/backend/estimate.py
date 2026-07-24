"""Duration-aware processing time estimates (UI only — no pipeline changes)."""

from __future__ import annotations

import math
from typing import Any

# Calibrated against live runs: ~2 min video ≈ 50s, ~12.5 min video ≈ 226s
BASE_SECONDS_LOW = 15.0
BASE_SECONDS_HIGH = 20.0
PER_MINUTE_LOW = 15.0
PER_MINUTE_HIGH = 20.0


def estimate_processing_seconds(duration_seconds: float | None) -> tuple[int, int]:
    """Return (low, high) processing time bounds in seconds."""
    if not duration_seconds or duration_seconds <= 0:
        return 45, 60

    minutes = duration_seconds / 60.0
    low = BASE_SECONDS_LOW + PER_MINUTE_LOW * minutes
    high = BASE_SECONDS_HIGH + PER_MINUTE_HIGH * minutes
    return int(math.floor(low)), int(math.ceil(high))


def format_duration_label(duration_seconds: float | None) -> str:
    """Human-readable video length for the loading UI."""
    if not duration_seconds or duration_seconds <= 0:
        return "unknown length"

    total = int(round(duration_seconds))
    if total < 60:
        return f"{total} sec"

    minutes, seconds = divmod(total, 60)
    if seconds == 0:
        return f"{minutes} min"
    return f"{minutes} min {seconds} sec"


def format_estimate_label(seconds_low: int, seconds_high: int) -> str:
    """Human-readable estimated wait time."""
    if seconds_high < 90:
        return f"{seconds_low}–{seconds_high} sec"

    low_min = max(1, int(math.floor(seconds_low / 60)))
    high_min = max(low_min, int(math.ceil(seconds_high / 60)))
    if low_min == high_min:
        return f"~{low_min} min"
    return f"{low_min}–{high_min} min"


def build_estimate_payload(
    video_id: str,
    title: str | None,
    duration_seconds: float | None,
) -> dict[str, Any]:
    """Assemble estimate response fields for the API."""
    low, high = estimate_processing_seconds(duration_seconds)
    return {
        "video_id": video_id,
        "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
        "title": title or "",
        "duration_seconds": float(duration_seconds) if duration_seconds else None,
        "duration_label": format_duration_label(duration_seconds),
        "estimated_seconds_low": low,
        "estimated_seconds_high": high,
        "estimated_label": format_estimate_label(low, high),
    }
