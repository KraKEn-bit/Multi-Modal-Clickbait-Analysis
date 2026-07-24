"""Check frame completion for specific CSV row indices."""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(PROJECT_ROOT / "data/input_videos.csv")
audit = pd.read_csv(PROJECT_ROOT / "data/audit_log.csv")
audit_map = dict(zip(audit.video_id.astype(str), audit.status))
frames_dir = PROJECT_ROOT / "data/extracted_frames"


def frame_status(vid: str) -> str:
    d = frames_dir / vid
    if not d.exists():
        return "no_frames"
    has = [(d / f"frame_{i}.png").exists() for i in range(3)]
    if all(has):
        return "complete"
    if any(has):
        return "partial"
    return "empty_dir"


start = int(sys.argv[1]) if len(sys.argv) > 1 else 1845
end = int(sys.argv[2]) if len(sys.argv) > 2 else 1860

print(f"Rows {start}-{end}:")
missing_live = []
for i in range(start, end + 1):
    vid = str(df.iloc[i]["video_id"])
    aud = audit_map.get(vid, "?")
    fr = frame_status(vid)
    mark = " *** GAP" if aud == "live" and fr != "complete" else ""
    print(f"  row {i}: {vid}  audit={aud}  frames={fr}{mark}")
    if aud == "live" and fr != "complete":
        missing_live.append((i, vid, fr))

print()
if missing_live:
    print(f"Live videos missing complete frames in range: {len(missing_live)}")
    for row, vid, fr in missing_live:
        print(f"  row {row}: {vid} ({fr})")
else:
    print("All live videos in this range have complete frames (or are dead).")
