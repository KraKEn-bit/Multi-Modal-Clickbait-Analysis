# VTCF — Bangla YouTube Clickbait Detector

**Hackathon demo** for the [Visual-Temporal Contradiction Framework (VTCF)](https://github.com/KraKEn-bit/Multi-Modal-Clickbait-Analysis): a multimodal system that flags Bangla YouTube clickbait by comparing what the **title promises** with what the **video actually shows** , not title text alone.

>  This is a real, working web app, not a mockup. Start the backend + frontend (5-minute setup below), open `http://localhost:3000`, and click the **Hard case : breaking news title, factual bulletin** example card. That one result, where a text-only model gets it wrong and VTCF gets it right, is the entire point of this project.

Part of: [Multi-Modal Clickbait Analysis](https://github.com/KraKEn-bit/Multi-Modal-Clickbait-Analysis)

---

## What problem does this solve?

Bangla YouTube clickbait often uses polished or sensational **headlines** while the **footage** tells a different story. Text-only classifiers, even strong ones like BanglaBERT can be fooled when the title alone reads as legitimate. Our own baseline model was **99.98% confident** on one such video, and completely wrong.

**VTCF fuses three signals to catch what text alone misses:**
1. **Title text** — BanglaBERT reads the headline's promise
2. **Visual frames** — a Vision Transformer (ViT) reads the hook, context, and delivery moments of the video
3. **Cross-modal fusion** — cross-attention compares the title against the visuals to catch contradictions

![Landing hero — detect clickbait hiding behind polished titles](docs/screenshots/SS-5.png)

---

## Demo walkthrough (recommended path for judges)

| Step | What to do | What you'll see |
|------|------------|-----------------|
| **1** | Open the landing page | Hero, then scroll to **Example videos** |
| **2** | Click **Hard case — breaking news title, factual bulletin** (`VTCF RESCUE`) | Full pipeline result in ~1 second — this example is pre-cached, no download needed |
| **3** | Scroll to **Judge Highlight — hard case** | Side-by-side comparison: the title-only baseline (BanglaBERT) gets this video **wrong** with 99.98% confidence; the full VTCF model gets it **right** with 99.96% confidence |
| **4** | Review the **Temporal frames** (Hook → Context → Delivery) | The actual visual evidence the model used to reach its verdict |
| **5** | (Optional) A second hard case, **sensational wedding headline**, is also on the Example videos grid — same pattern, opposite direction (BanglaBERT says genuine, VTCF correctly says clickbait) | A second, independent example of the same rescue effect |
| **6** | (Optional) Paste any YouTube URL and click **Analyze live** | Real download + inference on a video the model has never seen, ~45–90s depending on length. Requires local checkpoints/cookies — see Quick Start |

![Example videos grid + full VTCF verdict](docs/screenshots/SS-1.png)

![Judge Highlight — BanglaBERT wrong vs VTCF correct, with temporal frames](docs/screenshots/SS-2.png)

---

## Why text-only fails (the study behind this demo)

![Problem section — titles promise, videos don't always deliver](docs/screenshots/SS-3.png)

| Stat | Result | What it means |
|------|--------|----------------|
| **Full VTCF F1** | **99.63%** | On our full evaluation set of 8,047 human-labeled Bangla YouTube videos |
| **Hard-case rescue** | **100% vs 64%** | On the 33 hardest cases (where a well-written title fools text-only detection), VTCF's visual model rescues all of them. A comparable audio-transcript-based model only rescues 64% |
| **Text-only F1** | 90.4% | Reading the title alone — good on easy cases, blind to visual bait-and-switch |

---

## Temporal Divergence Score (TDS)

TDS (0–1) measures how much a video's **visuals change** across the hook, context, and delivery frames. It's shown alongside the frames and explanation as supporting evidence — **it is not a verdict by itself**.

**The counter-intuitive finding:** we expected clickbait to show a bigger visual shift (a classic bait-and-switch). The data said the opposite — clickbait videos average a **lower** TDS (~0.38) than genuine videos (~0.64), because clickbait often reuses the same static footage throughout rather than actually switching content.

![Interactive TDS score guide with hook / context / delivery timeline](docs/screenshots/SS-4.png)

---

## Quick start (run locally)

### Prerequisites

- Python 3.11+ · Node.js 18+ · [ffmpeg](https://ffmpeg.org/) on PATH
- Clone the **full parent repo** (not just `App/`):

```bash
git clone https://github.com/KraKEn-bit/Multi-Modal-Clickbait-Analysis.git
cd Multi-Modal-Clickbait-Analysis
```

**Link study code** (required once — the app imports `../vtcf-study`):

```powershell
# Windows
mklink /J vtcf-study VTCF-Finding-1
```

```bash
# macOS / Linux
ln -s VTCF-Finding-1 vtcf-study
```

**Model checkpoints (~2–3 GB, not included in this repo):** place trained weights at
`vtcf-study/outputs/checkpoints/best_model_full.pt`
(or train your own via [VTCF-Finding-1](../VTCF-Finding-1/readme.md)).
**The pre-cached example cards work fully without any checkpoints** — this is the fastest way to see real results.

### Terminal 1 — Backend

```powershell
cd App\backend
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### Terminal 2 — Frontend

```powershell
cd App\frontend
npm install
npm run dev
```

Open **http://localhost:3000**

Verify the API is running: `curl http://127.0.0.1:8000/health` → `{"status":"ok"}`

---

## Tech stack

| Layer | Stack |
|-------|-------|
| Text | BanglaBERT (`sagorsarker/bangla-bert-base`) |
| Vision | ViT (`google/vit-base-patch16-224`) |
| ML | PyTorch, Hugging Face Transformers |
| Video | yt-dlp, PySceneDetect, OpenCV, ffmpeg |
| Backend | FastAPI, Uvicorn |
| Frontend | Next.js 16, React 19, Tailwind CSS 4, Framer Motion |

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/examples` | 4 pre-cached demo videos, load instantly |
| `POST` | `/analyze` | `{ "youtube_url": "..." }` — runs the full live pipeline |

---

## Cached demo videos

| Video | Label | Why it matters |
|-------|-------|----------------|
| `pYganyZsHYM` | Obvious clickbait | Sensational English headline, viral-style footage |
| `hcFpC8R6c24` | Obvious genuine | Independent TV news bulletin, straightforward headline |
| `OoUO4vjgM4c` | Hard case — **VTCF rescue** | BanglaBERT (title-only) → clickbait, wrong. VTCF → genuine, correct. Real newsroom visuals the text model couldn't see |
| `DhESX8gA7wk` | Hard case — **VTCF rescue** | BanglaBERT (title-only) → genuine, wrong. VTCF → clickbait, correct. Frame content contradicts the title |

---

## Honest limitations

**Finding-1 (this demo, what's actually running live):**
- Model checkpoints and the full 8,047-frame cache aren't shipped in this repo (multi-GB scale) — download or retrain separately
- Live YouTube analysis may fail without valid browser cookies (YouTube rate-limits/blocks some automated downloads)
- The model samples 3 frames per video (hook, context, delivery), not full-video understanding
- Trained specifically on Bangla YouTube news-style content — may not generalize to other genres or languages

**Finding-2 (a second, audio-based model — referenced in our study stats, but not running live in this app):**
We also built a separate model that transcribes a video's speech and compares an AI-generated summary against the title, instead of watching the frames. It's the source of the "64%" figure above. We did **not** wire this into the live app — it depends on an external AI summarization API with daily rate limits, and in our own testing it only rescued 64% of hard cases vs. 100% for the visual approach, so the product is built around the stronger, faster, more reliable visual model. Additionally:
- This model was trained on only 1,591 of the full 8,047 videos, due to those API rate limits
- Thumbnail OCR text was sparse and usable on roughly 38% of videos even after a pipeline fix

---

## Project layout
