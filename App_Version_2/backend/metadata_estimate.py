"""Fetch YouTube metadata for duration-aware UI estimates."""

from __future__ import annotations

import logging
import sys
from typing import Any

from config import VTCF_RESEARCH_ROOT
from estimate import build_estimate_payload

logger = logging.getLogger(__name__)

if str(VTCF_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(VTCF_RESEARCH_ROOT))

from data.ingestion import fetch_video_metadata  # noqa: E402
from scripts.predict import extract_video_id, resolve_cookies_file  # noqa: E402


def fetch_analyze_estimate(youtube_url: str) -> dict[str, Any]:
    """Return video duration and calibrated processing time bounds."""
    video_id = extract_video_id(youtube_url)
    cookies_file = resolve_cookies_file()

    duration_seconds: float | None = None
    title: str | None = None

    try:
        metadata = fetch_video_metadata(video_id, cookies_file=cookies_file)
        raw_duration = metadata.get("duration")
        if raw_duration is not None:
            duration_seconds = float(raw_duration)
        fetched_title = metadata.get("title")
        if fetched_title:
            title = str(fetched_title)
    except Exception as exc:
        logger.warning("Metadata fetch for estimate failed (%s): %s", video_id, exc)
        return build_estimate_payload(video_id, title=None, duration_seconds=None)

    return build_estimate_payload(video_id, title=title, duration_seconds=duration_seconds)
