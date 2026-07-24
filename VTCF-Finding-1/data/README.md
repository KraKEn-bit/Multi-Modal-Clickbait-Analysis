# Data directory

## Included
- `verified_live_videos.csv` — 10,000 annotated Bangla YouTube videos with audit status and TDS scores
- `baseline_banglabert_model/` — trained text-only ablation checkpoint (~8 MB)
- `sample_frames/` — 6 demo videos with hook/context/delivery frames (3 PNGs each)

## Not included (regenerate locally)
- `BaitBuster_raw.parquet` / `.xlsx` — source annotations (contact authors)
- `extracted_frames/` — full frame cache (~1.3 GB); run `data/ingestion.py`
- `input_videos.csv` — full ingestion queue
- YouTube cookies — never commit; use your own for ingestion
