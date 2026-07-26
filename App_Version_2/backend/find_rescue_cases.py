"""Find hard-subset videos where VTCF rescues BanglaBERT failures."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

from config import HARD_SUBSET_CSV, VTCF_RESEARCH_ROOT  # noqa: E402
from pipeline import analyze_youtube_url  # noqa: E402

df = pd.read_csv(HARD_SUBSET_CSV)
frames_root = VTCF_RESEARCH_ROOT / "data" / "extracted_frames"

for _, row in df.iterrows():
    vid = str(row["video_id"])
    frame_dir = frames_root / vid
    if not all((frame_dir / f"frame_{i}.png").exists() for i in range(3)):
        continue
    true = "GENUINE" if int(row["label"]) == 0 else "CLICKBAIT"
    text_pred = "CLICKBAIT" if int(row["predicted_label"]) == 1 else "GENUINE"
    if text_pred == true:
        continue
    try:
        r = analyze_youtube_url(
            f"https://www.youtube.com/watch?v={vid}",
            frames_root=BACKEND / "_probe",
            offline_frames_dir=frame_dir,
        )
        rescued = r["verdict"] == true
        print(
            f"{vid} | true={true} | text={text_pred} | vtcf={r['verdict']} "
            f"({r['confidence']}%) | rescued={rescued}"
        )
    except Exception as exc:
        print(f"{vid} FAILED: {exc}")
