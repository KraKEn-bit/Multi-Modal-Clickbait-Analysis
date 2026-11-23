# Data & checkpoints (not stored in this repo)

GitHub can't hold the large files. Download them separately and put them exactly where shown below.

| What | Size | Put it at | Needed for |
|---|---|---|---|
| Frame cache `vtcf_extracted_frames.zip` (8,047 videos × 3 PNG) | 1.4 GB | unzip to `VTCF-Finding-1/data/extracted_frames/<video_id>/frame_{0,1,2}.png` | Training/evaluating VTCF, XAI |
| VTCF checkpoints `best_model_full.pt`, `best_model_vision_only.pt`, `best_model_text_only.pt` | ~3 GB each | `VTCF-Finding-1/outputs/checkpoints/` | Explaining the *existing* VTCF (Track A) |
| Finding-2 checkpoints `finding2_{title_only,title_ocr,title_summary,title_ocr_summary}.pt` | ~2 GB each | `VTCF-finding-2/outputs/checkpoints/` | Explaining the *existing* SemanticVTCF |
| Raw BaitBuster corpus `BaitBuster_raw.parquet` (253k rows) | 176 MB | `VTCF-Finding-1/data/` | Adding more channels (Track B) |

Download links: **<add Google Drive / Kaggle links here>**

## Already in the repo
- `VTCF-Finding-1/data/verified_live_videos.csv`: 10,000 labelled videos, live/dead status, pixel TDS
- `VTCF-Finding-1/data/input_videos.csv`: the same 10,000 videos with **channel_name**, dates, stats and all 3 label sources
- `VTCF-Finding-1/data/baseline_banglabert_model/hard_subset_video_ids.csv`: the 33 hard cases
- `VTCF-finding-2/data/finding2_verified.csv`, `finding2_split.csv` (the persisted 1,272/159/160 split)
- `VTCF-finding-2/data/transcripts/<id>/`: `summary.txt`, `hook_ocr.txt`, `metadata.json` for 1,628 videos (raw ASR transcripts are not included)

## Files you must create yourself (never commit them)
- `VTCF-finding-2/.env`:
  ```
  GEMINI_API_KEY=your_key
  ```
- YouTube cookies (only needed to download videos): export your own to `VTCF-Finding-1/data/youtube_cookies.txt`

## Environment
- Python 3.11 or 3.12. Install with `pip install -r requirements.txt` in each finding folder.
- ffmpeg must be on your PATH.
- The VTCF split is not stored in a file. It's rebuilt deterministically: live rows with all 3 frames, sorted by `video_id`, stratified 80/10/10 split with seed 42 (6,437/805/805).
