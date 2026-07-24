"""Generate paper figures 4–6 from existing logs and hardcoded results."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VIS_DIR = PROJECT_ROOT / "outputs" / "visualizations"

TRAINING_LOGS = {
    "Text-Only": {
        "path": PROJECT_ROOT / "outputs" / "logs" / "training_text_only.csv",
        "color": "#3498db",
    },
    "Vision-Only": {
        "path": PROJECT_ROOT / "outputs" / "logs" / "kaggle" / "training_vision_only.csv",
        "color": "#e74c3c",
    },
    "Full VTCF": {
        "path": PROJECT_ROOT / "outputs" / "logs" / "kaggle" / "training_full.csv",
        "color": "#2ecc71",
    },
}

TEXT_ONLY_CEILING = 0.9851


def _ensure_output_dir() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)


def _read_training_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Training log not found: {path}")
    dataframe = pd.read_csv(path)
    required = {"epoch", "val_f1"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return dataframe.sort_values("epoch")


def generate_figure_4_training_curves() -> Path:
    """Plot validation F1 curves for all ablation conditions."""
    figure, axis = plt.subplots(figsize=(8, 5))

    best_epoch_max = 10.0

    for label, spec in TRAINING_LOGS.items():
        dataframe = _read_training_log(spec["path"])
        epochs = dataframe["epoch"].astype(float)
        val_f1 = dataframe["val_f1"].astype(float)
        best_epoch_max = max(best_epoch_max, float(epochs.max()))
        axis.plot(
            epochs,
            val_f1,
            color=spec["color"],
            linestyle="-",
            linewidth=2,
            label=label,
        )

        best_index = val_f1.idxmax()
        best_epoch = float(epochs.loc[best_index])
        best_f1 = float(val_f1.loc[best_index])
        axis.plot(
            best_epoch,
            best_f1,
            marker="*",
            markersize=14,
            color=spec["color"],
            markeredgecolor="black",
            markeredgewidth=0.6,
            zorder=5,
        )

    axis.axhline(
        TEXT_ONLY_CEILING,
        color="#3498db",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
    )
    axis.text(
        best_epoch_max * 0.72,
        TEXT_ONLY_CEILING + 0.0008,
        "Text-Only ceiling",
        color="#3498db",
        fontsize=9,
        va="bottom",
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation F1 (Detection)")
    axis.set_title("VTCF Training Curves — All Ablation Conditions")
    axis.grid(True, linestyle=":", alpha=0.6)
    axis.legend(loc="upper right")
    axis.set_xlim(left=1)
    axis.set_ylim(bottom=0.96, top=1.001)

    output_path = VIS_DIR / "figure_4_training_curves.png"
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def generate_figure_5_ablation_chart() -> Path:
    """Plot test-set F1 comparison across ablation conditions."""
    models = [
        "Text-Only\n(BanglaBERT)",
        "Vision-Only\n(ViT)",
        "Full VTCF",
        "VTCF\nHard Subset",
    ]
    f1_scores = [0.9851, 0.9963, 0.9963, 1.0000]
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(models, f1_scores, color=colors, edgecolor="black", linewidth=0.8)

    for bar, score in zip(bars, f1_scores):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0005,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    axis.text(
        bars[0].get_x() + bars[0].get_width() / 2,
        bars[0].get_height() + 0.0035,
        "* p≈0 vs Full",
        ha="center",
        va="bottom",
        fontsize=10,
        color="red",
        fontweight="bold",
    )

    axis.set_ylabel("Test F1 (Detection)")
    axis.set_title("VTCF Ablation Study — Test Set Performance")
    axis.set_ylim(0.96, 1.005)
    axis.grid(axis="y", linestyle=":", alpha=0.5)

    output_path = VIS_DIR / "figure_5_ablation_chart.png"
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _draw_box(
    axis: plt.Axes,
    xy: tuple[float, float],
    text: str,
    facecolor: str,
    width: float = 0.12,
    height: float = 0.08,
    fontsize: int = 9,
) -> FancyBboxPatch:
    """Draw a rounded box centered at xy."""
    x, y = xy
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.3",
        linewidth=1.2,
        edgecolor="black",
        facecolor=facecolor,
        transform=axis.transAxes,
        zorder=2,
    )
    axis.add_patch(box)
    axis.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        transform=axis.transAxes,
        zorder=3,
    )
    return box


def _draw_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    dashed: bool = False,
    color: str = "black",
) -> None:
    """Draw an arrow between two axes-fraction coordinates."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.2,
        linestyle="--" if dashed else "-",
        color=color,
        transform=axis.transAxes,
        zorder=1,
    )
    axis.add_patch(arrow)


def generate_figure_6_architecture() -> Path:
    """Draw the VTCF architecture diagram using matplotlib patches."""
    figure, axis = plt.subplots(figsize=(14, 8))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    axis.set_xlim(0, 1)
    axis.set_ylim(-0.15, 0.85)
    axis.axis("off")
    axis.set_title(
        "VTCF Architecture — Visual-Temporal Contradiction Framework",
        fontsize=13,
        pad=16,
    )

    orange = "#fdebd0"
    blue = "#d6eaf8"
    green = "#d5f5e3"
    purple = "#e8daef"

    # Visual path
    _draw_box(axis, (0.05, 0.7), "YouTube\nVideo ID", orange, width=0.11, height=0.07)
    _draw_arrow(axis, (0.05, 0.66), (0.05, 0.59))
    _draw_box(axis, (0.05, 0.55), "yt-dlp +\nPySceneDetect", orange, width=0.12, height=0.07)
    _draw_arrow(axis, (0.05, 0.51), (0.08, 0.44))

    _draw_box(axis, (0.02, 0.38), "Hook\nframe_0\n224×224", orange, width=0.08, height=0.09)
    _draw_box(axis, (0.08, 0.38), "Context\nframe_1\n224×224", orange, width=0.08, height=0.09)
    _draw_box(axis, (0.14, 0.38), "Delivery\nframe_2\n224×224", orange, width=0.08, height=0.09)
    _draw_arrow(axis, (0.08, 0.33), (0.08, 0.26))

    _draw_box(
        axis,
        (0.08, 0.2),
        "ViT-B/16\n(×K independent\nforward passes)",
        orange,
        width=0.14,
        height=0.09,
    )
    _draw_arrow(axis, (0.08, 0.15), (0.08, 0.11))
    _draw_box(
        axis,
        (0.08, 0.05),
        "K×768\nTemporal Visual\nMatrix",
        orange,
        width=0.13,
        height=0.08,
    )

    # Text path
    _draw_box(axis, (0.55, 0.7), "Bangla\nVideo Title", blue, width=0.11, height=0.07)
    _draw_arrow(axis, (0.55, 0.66), (0.55, 0.57))
    _draw_box(
        axis,
        (0.55, 0.5),
        "BanglaBERT\n(sagorsarker/\nbangla-bert-base)",
        blue,
        width=0.14,
        height=0.09,
    )
    _draw_arrow(axis, (0.55, 0.45), (0.55, 0.36))
    _draw_box(
        axis,
        (0.55, 0.3),
        "T×768\nToken Sequence",
        blue,
        width=0.12,
        height=0.07,
    )

    # Fusion
    _draw_box(
        axis,
        (0.35, 0.15),
        "Cross-Modal\nAttention Fusion\nQ=Text K=V=Visual",
        green,
        width=0.16,
        height=0.09,
    )
    _draw_arrow(axis, (0.14, 0.05), (0.27, 0.15))
    _draw_arrow(axis, (0.49, 0.3), (0.43, 0.18))
    _draw_arrow(axis, (0.35, 0.10), (0.35, 0.07))

    _draw_box(
        axis,
        (0.35, 0.02),
        "768-dim\nFused Vector",
        green,
        width=0.11,
        height=0.06,
    )
    _draw_arrow(axis, (0.41, 0.02), (0.54, 0.02))
    _draw_box(
        axis,
        (0.6, 0.02),
        "Detection Head\n↓\nClickbait /\nNot Clickbait",
        green,
        width=0.12,
        height=0.08,
    )

    # TDS
    _draw_box(
        axis,
        (0.08, -0.1),
        "TDS =\n1 - cos(hook, delivery)\nContrastive Loss",
        purple,
        width=0.16,
        height=0.08,
    )
    _draw_arrow(axis, (0.08, 0.01), (0.08, -0.06), dashed=True, color="#8e44ad")

    output_path = VIS_DIR / "figure_6_architecture.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def main() -> None:
    """Generate all paper figures and print save locations."""
    _ensure_output_dir()

    outputs = [
        generate_figure_4_training_curves(),
        generate_figure_5_ablation_chart(),
        generate_figure_6_architecture(),
    ]

    print("All figures saved:")
    for path in outputs:
        status = "OK" if path.exists() else "MISSING"
        print(f"  [{status}] {path}")


if __name__ == "__main__":
    main()
