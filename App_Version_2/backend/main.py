"""VTCF Hackathon Demo — FastAPI backend."""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import (
    ANALYZE_TIMEOUT_SECONDS,
    CACHED_EXAMPLES_DIR,
    CORS_ORIGINS,
    MANIFEST_PATH,
    TEMP_FRAMES_DIR,
)
from pipeline import analyze_youtube_url, extract_video_id_from_url
from metadata_estimate import fetch_analyze_estimate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VTCF Clickbait Detector",
    description="Visual-Temporal Contradiction Framework for Bangla YouTube",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
CACHED_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/frames", StaticFiles(directory=str(TEMP_FRAMES_DIR)), name="frames")
app.mount(
    "/cached-frames",
    StaticFiles(directory=str(CACHED_EXAMPLES_DIR)),
    name="cached-frames",
)

_executor = ThreadPoolExecutor(max_workers=1)


class AnalyzeRequest(BaseModel):
    youtube_url: str = Field(..., min_length=8, description="YouTube watch URL or video ID")


class AnalyzeResponse(BaseModel):
    video_id: str
    youtube_url: str
    title: str
    verdict: str
    confidence: float
    tds_score: float
    explanation: str
    alignment_scores: dict[str, float]
    frame_urls: list[str]
    processing_time_seconds: float


class AnalyzeEstimateResponse(BaseModel):
    video_id: str
    youtube_url: str
    title: str
    duration_seconds: float | None
    duration_label: str
    estimated_seconds_low: int
    estimated_seconds_high: int
    estimated_label: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/examples")
def get_examples() -> dict[str, Any]:
    """Return pre-computed example analyses for instant demo."""
    if not MANIFEST_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Cached examples not ready. Run: python cache_examples.py",
        )
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return manifest


@app.post("/analyze/estimate", response_model=AnalyzeEstimateResponse)
async def analyze_estimate(request: AnalyzeRequest) -> dict[str, Any]:
    """Fetch video duration via yt-dlp and return calibrated wait-time bounds."""
    try:
        extract_video_id_from_url(request.youtube_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(
                _executor,
                lambda: fetch_analyze_estimate(request.youtube_url),
            ),
            timeout=30,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="Could not fetch video metadata in time",
        ) from exc
    except Exception as exc:
        logger.exception("Estimate failed")
        raise HTTPException(status_code=500, detail=f"Estimate failed: {exc}") from exc


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    """Run full VTCF pipeline on a new YouTube URL."""
    try:
        video_id = extract_video_id_from_url(request.youtube_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                _executor,
                lambda: analyze_youtube_url(
                    request.youtube_url,
                    frames_root=TEMP_FRAMES_DIR,
                ),
            ),
            timeout=ANALYZE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=f"Analysis timed out after {ANALYZE_TIMEOUT_SECONDS}s",
        ) from exc
    except RuntimeError as exc:
        message = str(exc)
        if "download failed" in message.lower():
            raise HTTPException(status_code=502, detail=message) from exc
        raise HTTPException(status_code=500, detail=message) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Analysis failed for %s", video_id)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    return result


@app.on_event("startup")
def startup() -> None:
    logger.info("VTCF backend started. Temp frames: %s", TEMP_FRAMES_DIR)
    if MANIFEST_PATH.exists():
        logger.info("Cached examples manifest found at %s", MANIFEST_PATH)
