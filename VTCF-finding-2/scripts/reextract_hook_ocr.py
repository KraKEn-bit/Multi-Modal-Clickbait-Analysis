"""Re-extract hook OCR from YouTube thumbnails (no ASR/Gemini re-run)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._paths import DEFAULT_CONFIG_PATH, load_config, resolve_path
from scripts.extract_speech import download_youtube_thumbnail, extract_ocr_bn
from scripts.semantic_features import is_usable_hook_ocr, resolve_promise_text

logger = logging.getLogger(__name__)

PILOT_VIDEO_IDS = [
    "nRnVXkQ1RRs",
    "unDUZmRvEhI",
    "0qQTzPvw_ig",
    "5-1l3UcUtYI",
    "-SpHq_CW38Y",
    "Nimw7-mZTSw",
    "CAPNGvKji0A",
    "17M9XIMGApE",
    "NMZSw7qrEWY",
    "69FUjUnwIS0",
    "DHaQUpweOi8",
    "95vimI2OIUk",
    "v3jHYhg3vXQ",
    "8-YmrdtPzx0",
    "ppnbT5J1MWc",
    "34K1OflTVrg",
    "7p_OX_Dtmkw",
    "EU7eDuoYFhs",
    "stXBMYLqhWk",
    "m1semWG1_Zo",
]


def classify_ocr(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return "empty"
    if is_usable_hook_ocr(cleaned):
        return "usable"
    return "non_empty_unusable"


def compute_stats(transcripts_dir: Path) -> dict[str, int]:
    counts = {"empty": 0, "non_empty_unusable": 0, "usable": 0, "total": 0}
    for child in sorted(transcripts_dir.iterdir()):
        if not child.is_dir():
            continue
        ocr_path = child / "hook_ocr.txt"
        text = ocr_path.read_text(encoding="utf-8").strip() if ocr_path.exists() else ""
        bucket = classify_ocr(text)
        counts[bucket] += 1
        counts["total"] += 1
    return counts


def read_old_frame_ocr(
    video_id: str,
    frames_dir: Path,
    min_confidence: float,
) -> str:
    """Re-run OCR on legacy frame_0.png with confidence filter for fair comparison."""
    legacy = frames_dir / video_id / "frame_0.png"
    if not legacy.exists():
        return ""
    text, _, _ = extract_ocr_bn(legacy, min_confidence=min_confidence)
    return text


def process_video(
    video_id: str,
    *,
    thumbnails_dir: Path,
    transcripts_dir: Path,
    cookies_file: Path | None,
    min_confidence: float,
    write: bool,
) -> dict:
    transcript_dir = transcripts_dir / video_id
    before_path = transcript_dir / "hook_ocr.txt"
    before = before_path.read_text(encoding="utf-8").strip() if before_path.exists() else ""

    thumbnail_path = download_youtube_thumbnail(video_id, thumbnails_dir, cookies_file)
    after = ""
    if thumbnail_path is not None:
        after, _, _ = extract_ocr_bn(thumbnail_path, min_confidence=min_confidence)

    if write and transcript_dir.exists():
        if thumbnail_path is None:
            logger.warning("Skip write for %s: thumbnail unavailable", video_id)
        elif not after and before:
            logger.warning(
                "Skip write for %s: thumbnail OCR empty but prior hook_ocr exists",
                video_id,
            )
        else:
            before_path.write_text(after, encoding="utf-8")
            metadata_path = transcript_dir / "metadata.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            else:
                metadata = {"video_id": video_id}
            title = str(metadata.get("title", ""))
            _, hook_promise_source = resolve_promise_text(title, after)
            metadata["ocr_text_detected"] = bool(after)
            metadata["ocr_char_count"] = len(after)
            metadata["hook_ocr_usable"] = is_usable_hook_ocr(after)
            metadata["hook_promise_source"] = hook_promise_source
            metadata["ocr_source"] = "youtube_thumbnail"
            metadata["ocr_min_confidence"] = min_confidence
            metadata_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    return {
        "video_id": video_id,
        "before": before,
        "after": after,
        "before_class": classify_ocr(before),
        "after_class": classify_ocr(after),
        "thumbnail_path": str(thumbnail_path) if thumbnail_path else "",
    }


def format_comparison(rows: list[dict]) -> str:
    lines = [
        "Hook OCR Pilot — Before (stored frame_0 OCR) vs After (YouTube thumbnail OCR)",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        lines.extend(
            [
                "=" * 72,
                f"{index}. video_id: {row['video_id']}",
                f"   before_class: {row['before_class']} -> after_class: {row['after_class']}",
                f"   thumbnail: {row['thumbnail_path'] or '(missing)'}",
                "",
                "   BEFORE (stored hook_ocr.txt):",
                f"   {row['before'] or '(empty)'}",
                "",
                "   AFTER (thumbnail OCR, conf-filtered):",
                f"   {row['after'] or '(empty)'}",
                "",
            ]
        )
    return "\n".join(lines)


def list_video_ids(config: dict, limit: int | None = None) -> list[str]:
    transcripts_dir = resolve_path(config["data"]["transcripts_dir"])
    ids = sorted(
        child.name
        for child in transcripts_dir.iterdir()
        if child.is_dir()
    )
    if limit is not None:
        return ids[:limit]
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-extract hook OCR from YouTube thumbnails")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--pilot", action="store_true", help="Run fixed 20-video pilot")
    parser.add_argument("--all", action="store_true", help="Run all videos in finding2_subset.csv")
    parser.add_argument("--write", action="store_true", help="Overwrite hook_ocr.txt + metadata")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="EasyOCR confidence threshold (default: config ocr.min_confidence)",
    )
    parser.add_argument(
        "--comparison-out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "ocr_pilot_comparison.txt",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    config = load_config(args.config)
    ocr_cfg = config.get("ocr", {})
    min_confidence = float(
        args.min_confidence if args.min_confidence is not None else ocr_cfg.get("min_confidence", 0.4)
    )

    transcripts_dir = resolve_path(config["data"]["transcripts_dir"])
    thumbnails_dir = resolve_path(config["data"]["thumbnails_dir"])
    cookies_file = resolve_path(config["data"]["youtube_cookies"])
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    before_stats = compute_stats(transcripts_dir)
    print("BEFORE stats:", before_stats)

    if args.pilot:
        video_ids = PILOT_VIDEO_IDS
    elif args.all:
        video_ids = list_video_ids(config)
    else:
        raise SystemExit("Specify --pilot or --all")

    rows: list[dict] = []
    for index, video_id in enumerate(video_ids, start=1):
        logger.info("OCR %s/%s: %s", index, len(video_ids), video_id)
        rows.append(
            process_video(
                video_id,
                thumbnails_dir=thumbnails_dir,
                transcripts_dir=transcripts_dir,
                cookies_file=cookies_file if cookies_file.exists() else None,
                min_confidence=min_confidence,
                write=args.write,
            )
        )

    if args.pilot and not args.write:
        args.comparison_out.parent.mkdir(parents=True, exist_ok=True)
        args.comparison_out.write_text(format_comparison(rows), encoding="utf-8")
        print(f"Saved pilot comparison -> {args.comparison_out}")

    if args.write:
        after_stats = compute_stats(transcripts_dir)
        print("AFTER stats:", after_stats)
        improved_usable = after_stats["usable"] - before_stats["usable"]
        usable_rate = after_stats["usable"] / max(after_stats["total"], 1)
        print(
            f"Usable delta: {improved_usable:+d} | "
            f"usable rate: {usable_rate:.1%} ({after_stats['usable']}/{after_stats['total']})"
        )


if __name__ == "__main__":
    main()
