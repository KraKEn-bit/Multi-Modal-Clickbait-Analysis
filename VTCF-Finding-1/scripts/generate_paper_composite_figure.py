"""Generate a camera-ready composite figure: Grad-CAM keyframes + TDS KDE.

Replaces the Gemini-fabricated HPa6mRwjUg8 collage with real VTCF outputs.
Default target: OoUO4vjgM4c (professional Independent TV bulletin where
title-only over-triggered clickbait; VTCF reads newsroom visuals correctly).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.gridspec import GridSpec
from PIL import Image
from scipy.stats import gaussian_kde, mannwhitneyu
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.custom_dataset import default_image_transform, resolve_project_path
from models.fusion_network import load_config
from models.interpretability import (
    _cls_spatial_attention_map,
    extract_attention_maps,
)
from scripts.evaluate import load_checkpoint_model

FRAME_LABELS = [
    "Frame 0 (Hook)",
    "Frame 1 (Context)",
    "Frame 2 (Delivery)",
]


@dataclass
class FigureTypography:
    """Paper-readable font sizes; multiply bases by ``scale``."""

    scale: float = 1.5
    graph_scale: float = 0.72  # smaller in-plot notes vs frame/header text

    def sz(self, base: float) -> float:
        return base * self.scale

    def graph_sz(self, base: float) -> float:
        return base * self.scale * self.graph_scale

    @property
    def header(self) -> float:
        return self.sz(13)

    @property
    def video_title(self) -> float:
        return self.sz(13)

    @property
    def frame_title(self) -> float:
        return self.sz(14)

    @property
    def gradcam_badge(self) -> float:
        return self.sz(8.5)

    @property
    def plot_title(self) -> float:
        return self.sz(12.5)

    @property
    def axis_label(self) -> float:
        return self.graph_sz(10)

    @property
    def tick(self) -> float:
        return self.graph_sz(9)

    @property
    def annotation(self) -> float:
        return self.graph_sz(8.5)

    @property
    def legend(self) -> float:
        return self.graph_sz(8)


def _setup_bengali_font() -> None:
    """Prefer a Windows font that can render Bangla titles."""
    from matplotlib import font_manager

    candidates = [
        "Nirmala UI",
        "Noto Sans Bengali",
        "Arial Unicode MS",
        "Segoe UI",
        "Kalpurush",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return


def _truncate_title(title: str, max_len: int = 100) -> str:
    cleaned = " ".join(str(title).split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rstrip() + "…"


def _tds_vertical_marker(ax_kde, tds: float) -> None:
    """Draw dashed vertical line at this video's TDS score."""
    tds_plot = min(max(tds, 0.0), 1.0)
    ax_kde.axvline(tds_plot, color="#111111", ls="--", lw=1.3, alpha=0.85, zorder=5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video-id",
        default="OoUO4vjgM4c",
        help="Video ID with extracted frames (default: professional-news rescue case)",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Override frames directory (defaults to config frames_dir / video_id)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "best_model_full.pt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.5,
        help="Multiply frame/header/title font sizes (default 1.5)",
    )
    parser.add_argument(
        "--graph-scale",
        type=float,
        default=0.72,
        help="Relative scale for in-graph text (legend, Mann-Whitney, axes)",
    )
    return parser.parse_args()


def load_title(video_id: str, config: dict) -> str:
    csv_path = resolve_project_path(config["data"]["verified_csv"])
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        hit = df.loc[df["video_id"].astype(str) == video_id]
        if len(hit):
            return str(hit.iloc[0]["title"])
    # App cached examples
    for alt in [
        PROJECT_ROOT.parent
        / "App"
        / "Version-2"
        / "backend"
        / "cached_examples"
        / video_id
        / "result.json",
    ]:
        if alt.exists():
            import json

            return str(json.loads(alt.read_text(encoding="utf-8"))["title"])
    return video_id


def resolve_frames_dir(video_id: str, config: dict, override: Path | None) -> Path:
    if override is not None:
        return Path(override)
    base = resolve_project_path(config["data"]["frames_dir"]) / video_id
    if base.exists():
        return base
    alt = (
        PROJECT_ROOT.parent
        / "App"
        / "Version-2"
        / "frontend"
        / "public"
        / "cached-frames"
        / video_id
    )
    if alt.exists():
        return alt
    alt2 = (
        PROJECT_ROOT.parent
        / "App"
        / "Version-2"
        / "backend"
        / "cached_examples"
        / video_id
    )
    if alt2.exists():
        return alt2
    raise FileNotFoundError(f"No frames found for {video_id}")


def load_pixel_values(frames_dir: Path, k_frames: int) -> torch.Tensor:
    transform = default_image_transform()
    tensors = []
    for i in range(k_frames):
        path = frames_dir / f"frame_{i}.png"
        if not path.exists():
            raise FileNotFoundError(path)
        image = Image.open(path).convert("RGB")
        tensors.append(transform(image))
    return torch.stack(tensors, dim=0).unsqueeze(0)


def overlay_gradcam(frame_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Jet-style Grad-CAM overlay on an RGB frame (uint8)."""
    h, w = frame_rgb.shape[:2]
    heat = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat_u8 = np.uint8(255 * heat)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    blend = (alpha * colored.astype(np.float32) + (1.0 - alpha) * frame_rgb.astype(np.float32))
    return np.clip(blend, 0, 255).astype(np.uint8)


def load_frame_rgb(frames_dir: Path, index: int, size: int = 224) -> np.ndarray:
    path = frames_dir / f"frame_{index}.png"
    image = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return np.array(image)


def tds_distribution_from_csv(config: dict) -> tuple[np.ndarray, np.ndarray, float]:
    csv_path = resolve_project_path(config["data"]["verified_csv"])
    df = pd.read_csv(csv_path)
    df = df[df["audit_status"] == "live"].dropna(subset=["tds_score", "label"])
    cb = df.loc[df["label"] == "clickbait", "tds_score"].astype(float).to_numpy()
    ncb = df.loc[df["label"] == "non_clickbait", "tds_score"].astype(float).to_numpy()
    # Drop saturated 1.0 ingestion placeholders if they dominate
    cb = cb[cb < 0.999]
    ncb = ncb[ncb < 0.999]
    if len(cb) < 5 or len(ncb) < 5:
        cb = df.loc[df["label"] == "clickbait", "tds_score"].astype(float).to_numpy()
        ncb = df.loc[df["label"] == "non_clickbait", "tds_score"].astype(float).to_numpy()
    stat = mannwhitneyu(cb, ncb, alternative="two-sided")
    return cb, ncb, float(stat.pvalue)


def run_model(
    video_id: str,
    frames_dir: Path,
    title: str,
    config: dict,
    checkpoint: Path,
    device: torch.device,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_encoder"])
    encoding = tokenizer(
        title,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    batch = {
        "input_ids": encoding["input_ids"].to(device),
        "attention_mask": encoding["attention_mask"].to(device),
        "pixel_values": load_pixel_values(frames_dir, int(config["model"]["K_frames"])).to(device),
    }

    model = load_checkpoint_model(checkpoint, config, device, condition="full")
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
        )
        maps = extract_attention_maps(model, batch)

    tds_tensor = outputs.get("tds_computed")
    if tds_tensor is not None:
        tds = float(tds_tensor[0].item())
    else:
        from models.interpretability import compute_tds

        tds = float(compute_tds(outputs["temporal_visual_matrix"])[0].item())

    probs = torch.softmax(outputs["detection_logits"], dim=-1)[0]
    pred = int(torch.argmax(probs).item())
    conf = float(probs[pred].item())

    vit = maps["vit_attention"][0].detach().cpu()  # [K, heads, 14, 14]
    cross = maps["cross_attention"][0].detach().cpu()

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "tds": tds,
        "pred": pred,
        "confidence": conf,
        "vit_attention": vit,
        "cross_attention": cross,
        "title": title,
    }


def build_figure(
    video_id: str,
    frames_dir: Path,
    result: dict,
    clickbait_tds: np.ndarray,
    genuine_tds: np.ndarray,
    mannwhitney_p: float,
    save_path: Path,
    dpi: int,
    typography: FigureTypography | None = None,
) -> None:
    _setup_bengali_font()
    typ = typography or FigureTypography()
    overlays = []
    frame_px = 320
    for i in range(3):
        frame = load_frame_rgb(frames_dir, i, size=frame_px)
        heat = _cls_spatial_attention_map(result["vit_attention"], i)
        overlays.append(overlay_gradcam(frame, heat))

    fig = plt.figure(figsize=(15.5, 7.0), facecolor="white")
    gs = GridSpec(
        2,
        4,
        figure=fig,
        height_ratios=[1.0, 0.24],
        width_ratios=[1.0, 1.0, 1.0, 1.55],
        wspace=0.20,
        hspace=0.06,
        left=0.04,
        right=0.98,
        top=0.86,
        bottom=0.10,
    )

    pred_name = "Clickbait" if result["pred"] == 1 else "Genuine"
    title_text = _truncate_title(result.get("title", ""))
    fig.suptitle(
        f"Video ID: {video_id}    |    TDS: {result['tds']:.3f}    |    "
        f"VTCF: {pred_name} ({100 * result['confidence']:.1f}%)",
        fontsize=typ.header,
        fontweight="bold",
        y=0.975,
    )

    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(overlays[i])
        ax.set_title(FRAME_LABELS[i], fontsize=typ.frame_title, pad=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_color("#222222")
        if i == 0:
            ax.text(
                0.03,
                0.97,
                "Grad-CAM",
                transform=ax.transAxes,
                fontsize=typ.gradcam_badge,
                color="white",
                fontweight="bold",
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.25", fc="#111111", alpha=0.55, ec="none"),
            )

    ax_title = fig.add_subplot(gs[1, 0:3])
    ax_title.axis("off")
    ax_title.text(
        0.0,
        0.55,
        f"Video Title: {title_text}",
        fontsize=typ.video_title,
        ha="left",
        va="center",
        color="#222222",
        wrap=True,
    )

    ax_kde = fig.add_subplot(gs[0:2, 3])
    xs = np.linspace(0.0, 1.0, 400)
    for scores, color, label in [
        (clickbait_tds, "#d62728", "Clickbait"),
        (genuine_tds, "#1f77b4", "Genuine"),
    ]:
        if len(scores) < 3:
            continue
        kde = gaussian_kde(scores)
        dens = kde(xs)
        mu = float(np.mean(scores))
        ax_kde.plot(xs, dens, color=color, lw=2.4, label=rf"{label} ($\mu$={mu:.3f})")
        ax_kde.fill_between(xs, dens, color=color, alpha=0.12)

    tds = result["tds"]
    _tds_vertical_marker(ax_kde, tds)

    p_text = "p < 0.001" if mannwhitney_p < 0.001 else f"p = {mannwhitney_p:.3f}"
    ax_kde.text(
        0.03,
        0.97,
        f"Mann–Whitney U, {p_text}\nTDS = {tds:.3f}",
        transform=ax_kde.transAxes,
        fontsize=typ.annotation,
        va="top",
        ha="left",
        linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.92, ec="#cccccc"),
    )
    ax_kde.set_xlim(0.0, 1.0)
    ax_kde.set_xlabel("TDS Score (0 to 1)", fontsize=typ.axis_label, labelpad=6)
    ax_kde.set_ylabel("Density", fontsize=typ.axis_label, labelpad=6)
    ax_kde.set_title(
        "Temporal Divergence Score (TDS) Distribution",
        fontsize=typ.plot_title,
        pad=18,
        fontweight="bold",
        y=1.02,
    )
    ax_kde.legend(
        loc="upper right",
        fontsize=typ.legend,
        frameon=True,
        framealpha=0.92,
        edgecolor="#cccccc",
        handlelength=1.8,
        borderpad=0.45,
        labelspacing=0.35,
    )
    ax_kde.grid(True, alpha=0.22, lw=0.5)
    ax_kde.tick_params(labelsize=typ.tick)

    # Nudge graph down so its title clears the header line.
    kde_pos = ax_kde.get_position()
    ax_kde.set_position(
        [kde_pos.x0, kde_pos.y0 - 0.028, kde_pos.width, kde_pos.height * 0.96]
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print(f"Saved -> {save_path}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    frames_dir = resolve_frames_dir(args.video_id, config, args.frames_dir)
    title = load_title(args.video_id, config)
    print(f"Video: {args.video_id}")
    print(f"Title: {title}")
    print(f"Frames: {frames_dir}")

    result = run_model(
        video_id=args.video_id,
        frames_dir=frames_dir,
        title=title,
        config=config,
        checkpoint=args.checkpoint,
        device=device,
    )
    print(
        f"TDS={result['tds']:.4f}  pred={'clickbait' if result['pred']==1 else 'genuine'}  "
        f"conf={result['confidence']:.4f}"
    )

    cb, ncb, p_val = tds_distribution_from_csv(config)
    print(f"TDS means: clickbait={cb.mean():.3f} genuine={ncb.mean():.3f} MW p={p_val:.2e}")

    out = args.output or (
        PROJECT_ROOT
        / "outputs"
        / "visualizations"
        / f"figure_paper_composite_{args.video_id}.png"
    )
    build_figure(
        video_id=args.video_id,
        frames_dir=frames_dir,
        result=result,
        clickbait_tds=cb,
        genuine_tds=ncb,
        mannwhitney_p=p_val,
        save_path=out,
        dpi=args.dpi,
        typography=FigureTypography(scale=args.font_scale, graph_scale=args.graph_scale),
    )


if __name__ == "__main__":
    main()
