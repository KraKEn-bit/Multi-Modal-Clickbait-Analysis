"""Build 20% subset CSV and merge transcript/OCR artifacts into dataset rows."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._paths import DEFAULT_CONFIG_PATH, load_config, resolve_path

logger = logging.getLogger(__name__)


def map_detection_label(label: str) -> int:
    text = str(label).strip().lower().replace("-", "_")
    if "non_clickbait" in text or "not_clickbait" in text:
        return 0
    if "non" in text and "clickbait" in text:
        return 0
    if text == "clickbait" or "clickbait" in text:
        return 1
    return 0


def filter_usable(dataframe: pd.DataFrame, frames_dir: Path) -> pd.DataFrame:
    """Keep live rows with title, label, and all 3 hook/context/delivery frames."""
    filtered = dataframe[dataframe["audit_status"].astype(str).str.lower() == "live"].copy()
    filtered = filtered[filtered["title"].notna() & filtered["label"].notna()].copy()

    def has_frames(video_id: str) -> bool:
        frame_dir = frames_dir / str(video_id)
        return all((frame_dir / f"frame_{index}.png").exists() for index in range(3))

    mask = filtered["video_id"].astype(str).map(has_frames)
    return filtered[mask].reset_index(drop=True)


def build_subset_csv(
    config: dict,
    fraction: float,
    seed: int,
    output_path: Path,
) -> pd.DataFrame:
    """Sample stratified subset from vtcf-research verified CSV."""
    verified_csv = resolve_path(config["data"]["verified_csv"])
    frames_dir = resolve_path(config["data"]["frames_dir"])

    dataframe = pd.read_csv(verified_csv)
    dataframe = filter_usable(dataframe, frames_dir)
    dataframe["detection_label"] = dataframe["label"].map(map_detection_label)

    subset_size = max(int(len(dataframe) * fraction), 1)
    subset, _ = train_test_split(
        dataframe,
        train_size=subset_size,
        random_state=seed,
        stratify=dataframe["detection_label"],
    )
    subset = subset.sort_values("video_id").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output_path, index=False)
    logger.info("Wrote %s rows to %s", len(subset), output_path)
    return subset


def attach_transcripts(dataframe: pd.DataFrame, transcripts_dir: Path) -> pd.DataFrame:
    """Merge transcript.txt, ocr_hook.txt, metadata.json into dataset columns."""
    enriched = dataframe.copy()
    transcripts: list[str] = []
    ocr_texts: list[str] = []
    speech_coverages: list[float] = []
    ocr_successes: list[bool] = []
    has_transcript: list[bool] = []

    for video_id in enriched["video_id"].astype(str):
        video_dir = transcripts_dir / video_id
        transcript_path = video_dir / "transcript.txt"
        ocr_path = video_dir / "ocr_hook.txt"
        meta_path = video_dir / "metadata.json"

        transcript = (
            transcript_path.read_text(encoding="utf-8").strip()
            if transcript_path.exists()
            else ""
        )
        ocr_text = ocr_path.read_text(encoding="utf-8").strip() if ocr_path.exists() else ""
        coverage = 0.0
        ocr_ok = bool(ocr_text)
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            coverage = float(metadata.get("speech_coverage", 0.0))
            ocr_ok = bool(metadata.get("ocr_success", ocr_ok))

        transcripts.append(transcript)
        ocr_texts.append(ocr_text)
        speech_coverages.append(coverage)
        ocr_successes.append(ocr_ok)
        has_transcript.append(bool(transcript))

    enriched["transcript"] = transcripts
    enriched["ocr_hook"] = ocr_texts
    enriched["speech_coverage"] = speech_coverages
    enriched["ocr_success"] = ocr_successes
    enriched["has_transcript"] = has_transcript
    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and extend Finding-2 dataset CSV")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Input CSV to enrich (default: build 20%% subset from verified CSV)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "finding2_subset.csv",
    )
    parser.add_argument(
        "--build-subset-only",
        action="store_true",
        help="Only write stratified subset without transcript merge",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    config = load_config(args.config)
    transcripts_dir = resolve_path(config["data"]["transcripts_dir"])

    if args.input_csv:
        dataframe = pd.read_csv(args.input_csv)
    else:
        fraction = float(config["data"].get("subset_fraction", 0.20))
        seed = int(config["data"].get("subset_seed", 42))
        dataframe = build_subset_csv(
            config=config,
            fraction=fraction,
            seed=seed,
            output_path=args.output_csv,
        )
        if args.build_subset_only:
            return

    enriched = attach_transcripts(dataframe, transcripts_dir)
    enriched.to_csv(args.output_csv, index=False)
    logger.info(
        "Extended dataset saved to %s | has_transcript=%s/%s",
        args.output_csv,
        int(enriched["has_transcript"].sum()),
        len(enriched),
    )


if __name__ == "__main__":
    main()
