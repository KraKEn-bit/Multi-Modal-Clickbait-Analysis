# Modeling Temporal and Semantic Contradictions in Bangla YouTube Clickbait

> **Anonymous submission** · Conference manuscript under review  
> **Code release:** full source, configs, and reproduction scripts will be published upon paper acceptance. Until then, this repository documents the research contribution without releasing implementation details.

---

## Overview

Clickbait detection is well studied for text headlines and static thumbnail–article pairs. This work asks a different question: **does a video’s own progression over time give away a lie**, and how does a **visual-timeline signal** compare to a **speech-based** one for Bangla—a low-resource language?

We present two matched-architecture systems:

| System | Promise | Delivery | Divergence signal |
|--------|---------|----------|-------------------|
| **VTCF** (Visual-Temporal Contradiction Framework) | Title (BanglaBERT) | Three frames: hook → context → delivery (ViT) | **TDS** — Temporal Divergence Score |
| **SemanticVTCF** | Title (+ optional thumbnail OCR) | LLM summary of ASR transcript | **SDS** — Semantic Divergence Score |

Both share the same cross-attention fusion template and contrastive-margin loss family; they differ only in what counts as *delivery*. That lets us attribute performance gaps to the evidence modality, not to architectural asymmetry.

---

## Key Results (at a glance)

| Finding | Result |
|---------|--------|
| Full VTCF / Vision-only on general test ($n{=}805$) | **F1 = 0.9963** |
| Text-only baseline (BanglaBERT titles) | F1 = 0.9851 |
| Hard-subset rescue — VTCF (evaluable $n{=}29$) | **100%** (29/29) |
| Hard-subset rescue — SemanticVTCF (evaluable $n{=}25$) | **64%** |
| Counter-intuitive diagnostic | Clickbait mean TDS **0.383** < genuine **0.644** (Mann–Whitney $U$, $p \approx 0$) |

**Takeaway:** on the hardest, most deceptive cases, a video’s visual timeline is substantially more reliable than an LLM-summarized transcript of its speech. The gap is structural: speech summarization washes out production-style cues (banners, B-roll editing, channel branding) that visual modeling reads directly.

---

## Motivation

Existing multimodal clickbait work typically treats video as a **single static artifact** (a thumbnail). Audio/transcript signals appear only as one ingredient in ensembles. Prior work has not:

1. Modeled a video’s **temporal visual arc** as a contradiction signal  
2. Built a dedicated **promise–delivery** architecture for speech in a low-resource language  
3. Compared these axes under **matched hard-case conditions**

This paper does all three for Bangla YouTube, using BaitBuster-Bangla human labels.

---

## Dataset

Built on **BaitBuster-Bangla** (253,070 entries, 58 channels), restricted to **10,000 human-annotated** Clickbait / Not-Clickbait videos so both axes share the same ground truth.

| Pipeline stage | VTCF | SemanticVTCF |
|----------------|------|--------------|
| Human-labelled pool | 10,000 | 10,000 |
| Live on YouTube | 8,054 / 10,000 | 8,054 / 10,000 |
| Usable after modality extraction | **8,047** (99.9% of live) | **1,591** (19.8% of live) |
| Train / Val / Test | 6,437 / 805 / 805 | 1,272 / 159 / 160 |
| Hard-subset evaluable (of 33) | 29 | 25 |

- **VTCF frames:** hook (opening), context (midpoint), delivery (closing) via content-aware scene detection; resized to $224{\times}224$.  
- **SemanticVTCF:** ASR (Bengali-fine-tuned Whisper) → Gemini summary; optional EasyOCR on thumbnails with confidence filtering. ASR cost limits SemanticVTCF to a stratified ~20% subset.  
- **Hard subset:** 33 videos where a standalone text-only BanglaBERT fails with confidence $>0.85$. Coverage drops where frames or speech are missing.

Splits are seeded once and **persisted to a fixed file** so composition cannot silently drift across runs.

---

## Method (conceptual)

### VTCF
- **Title encoder:** BanglaBERT → $T \in \mathbb{R}^{L \times 768}$  
- **Vision:** ViT-B/16 on $K{=}3$ frames → temporal matrix $V \in \mathbb{R}^{K \times 768}$  
- **Temporal position embeddings** so hook vs. delivery roles are distinguishable  
- **Cross-modal attention:** text queries visual keys/values  
- **TDS** (diagnostic + training signal):

$$
\mathrm{TDS} = 1 - \cos(v_{\mathrm{hook}}, v_{\mathrm{delivery}})
$$

### SemanticVTCF
Same fusion template; delivery is a Gemini summary of the transcript. Ablations:

| ID | Inputs |
|----|--------|
| **A** | Title only |
| **B** | Title + OCR |
| **C** | Title + summary |
| **D** | Title + OCR + summary (full) |

### Contrastive divergence loss
Adapted from prior Romanian clickbait work (margin-based cosine dissimilarity). Instantiated twice:

- VTCF: $(p,d) = (v_{\mathrm{hook}}, v_{\mathrm{delivery}})$ → $\mathcal{L}_{\mathrm{TDS}}$  
- SemanticVTCF: pooled promise vs. summary → $\mathcal{L}_{\mathrm{SDS}}$  

Full objective: $\mathcal{L} = \alpha\,\mathcal{L}_{\mathrm{det}} + \beta\,\mathcal{L}_{\mathrm{div}}$ with $\alpha{=}1.0$, $\beta{=}0.3$, margin $m{=}0.5$.

> **Attribution note:** we adapt an existing contrastive-margin family; the contribution is the **modality/language adaptation** and the **TDS/SDS construction**, not invention of the margin loss itself.

---

## Main Results

### General and hard splits

| Model | Split ($n$) | F1 | Acc. |
|-------|-------------|-----|------|
| *VTCF* | | | |
| Text-Only (title) | General (805) | 0.9851 | — |
| Vision-Only (frames) | General (805) | 0.9963 | — |
| **Full VTCF** | General (805) | **0.9963** | — |
| Text-Only | Hard (33) | 0.0000 | — |
| **Full VTCF** | Hard (29) | **1.0000** | — |
| *SemanticVTCF (A–D)* | | | |
| A — title only | General (160) | 0.9937 | 0.9938 |
| B — title+OCR | General (160) | 1.0000 | 1.0000 |
| C — title+summary | General (160) | 1.0000 | 1.0000 |
| D — full | General (160) | 0.9937 | 0.9938 |
| *(cf.)* VTCF | General (160) | 0.9875 | 0.9875 |
| A / C / D (tied) | Hard (25) | 0.6305 | — |
| *(cf.)* Full VTCF | Hard (29) | **1.0000** | — |

On the general split, titles alone nearly saturate accuracy (McNemar: full VTCF vs. text-only $p \approx 0$; vs. vision-only $p{=}1.0$). The **hard subset** is the primary evidence that visual evidence adds real capability.

### OCR channel (negative result)

| Category (of 1,600) | Before | After |
|---------------------|--------|-------|
| Empty (no text) | 877 | 889 |
| Unusable / garbage | 586 | 101 |
| Usable Bangla text | 137 (8.6%) | **610 (38.1%)** |

Switching to true thumbnails + confidence filter ($\geq 0.4$) raised usable OCR by **4.4×**, but OCR still adds **no measurable benefit** beyond title+summary. Treated as a genuine negative finding, not merely an extraction bug.

### Why SemanticVTCF underperforms on hard cases

Manual audit of 9 videos that text-only and SemanticVTCF both miss (all rescued by VTCF):

| Pattern | Mechanism | Count |
|---------|-----------|-------|
| **False clickbait** | Sensational title + calm factual report; Gemini summary reads as neutral wire-copy → spurious “mismatch” | 4/9 |
| **False non-clickbait** | Aggressive visual packaging (banners, B-roll, overlays) + flat voiceover; summary looks clean → deception missed | 5/9 |

This is a **structural ceiling** of speech-only summarization: the decisive evidence lives on screen, not in words.

### Counter-intuitive TDS finding

Pre-training diagnostics show **lower** hook–delivery divergence for clickbait than for genuine videos (means 0.383 vs. 0.644). Interpretation: low-effort clickbait often stays visually static; genuine news shows more scene variety across a real timeline—opposite of the classical “bait-and-switch” story. The same directional pattern appears for semantic divergence.

---

## Verification & robustness

Headline numbers near F1 = 1.0 warrant scrutiny. Checks performed:

1. **Leakage screening** — zero video-ID overlap across train/val/test for both pipelines.  
2. **Title-only isolation** — ablation A (F1 = 0.9937) nearly matches full SemanticVTCF; only 1/160 predictions flipped across ablations A–D (general-split title saturation).  
3. **Manual reading** — title / OCR / summary inspected for 10 videos; summaries are genuine descriptions, not paraphrased titles. Example: `nRnVXkQ1RRs` (office-anniversary) — summary correctly overrides a superficially alarming but celebratory title.

### Engineering issues caught before final reporting

| Problem | Diagnosis / fix | Outcome |
|---------|-----------------|---------|
| ASR wrong language | Benchmarked 3 ASR models on 50-video pilot; Bengali-tuned Whisper | 50/50 pilot videos readable |
| Text-only baseline silently used image pipeline | Isolated true text-only path; retrained | F1 corrected 0.34 → 0.9851 |
| Hard-subset eval found only 4/33 | Expanded ID search from test-only to all splits | Coverage 29/33 |
| Split drift (unpersisted) | Persist exact split to a fixed file | Run-to-run reproducibility |
| OCR mostly garbage (8.6% usable) | True thumbnail + confidence filter | 38.1% usable (4.4×) |
| Checkpoint load failure (library version change) | Key-remapping compatibility utility | Checkpoint restored; no retrain |

None of these affect the **post-fix** numbers reported above.

---

## Limitations (please read)

- **Scale imbalance:** SemanticVTCF trains on ~1,591 videos vs. VTCF’s ~8,047; compare axes with this gap in mind.  
- **Hard subset size:** 29 (VTCF) / 25 (SemanticVTCF) of 33 candidates; the 100% vs. 64% pattern held across two independent manual audits, but $n$ is modest.  
- **General-split saturation:** SemanticVTCF ablation differences rest on a single flipped prediction out of 160—not statistically robust alone. Hard subset is primary evidence.  
- **Summarizer dependence:** SDS inherits Gemini’s output style; another summarizer might retain more production-adjacent cues, but our structural account suggests that alone would not close the full ~36-point hard-case gap.  
- **Method attribution:** TDS/SDS losses adapt an existing contrastive-margin family; novelty is adaptation + construction for video/speech in Bangla, not the margin mechanism itself.  
- **Three-frame sampling:** denser or learned temporal sampling is left for future work.

---

## What is / is not in this repository (pre-acceptance)

| Included (documentation) | Deferred until acceptance |
|--------------------------|---------------------------|
| Paper-facing description of methods & results | Training / inference source code |
| High-level architecture narrative | Model checkpoints & configs |
| Tables and verified numbers from the camera-ready draft | Raw frame / audio / transcript corpora |
| Limitation and verification notes | End-to-end reproduction scripts |

> **Code and full artifacts will be released publicly when the paper is accepted**, under a license announced with that release. Until then, please cite the preprint/camera-ready version of the paper rather than reverse-engineering unpublished code.

---

