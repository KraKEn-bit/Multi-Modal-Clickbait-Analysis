"""XAI and interpretability utilities for the VTCF framework."""

from __future__ import annotations

import logging
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NUM_VIT_PATCHES = 196
VIT_GRID_SIZE = 14


def compute_tds(frame_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute Temporal Divergence Score between hook and delivery frame embeddings.

    Args:
        frame_embeddings: Tensor of shape [B, K, 768].

    Returns:
        Tensor of shape [B] containing TDS values.
    """
    hook_embedding = F.normalize(frame_embeddings[:, 0, :], p=2, dim=-1)
    delivery_embedding = F.normalize(frame_embeddings[:, -1, :], p=2, dim=-1)
    cosine_similarity = torch.clamp(
        (hook_embedding * delivery_embedding).sum(dim=-1),
        min=-1.0,
        max=1.0,
    )
    return 1.0 - cosine_similarity


def _compute_vit_attention_probs(
    attention_module: torch.nn.Module,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Manually compute ViT self-attention probabilities for hook-based XAI."""
    query_layer = attention_module.transpose_for_scores(
        attention_module.query(hidden_states)
    )
    key_layer = attention_module.transpose_for_scores(
        attention_module.key(hidden_states)
    )
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(attention_module.attention_head_size)
    return torch.softmax(attention_scores, dim=-1)


@contextmanager
def _attention_capture_context(model: torch.nn.Module) -> Iterator[dict[str, torch.Tensor | None]]:
    """
    Register temporary hooks to capture ViT self-attention and cross-attention weights.

    Yields:
        Mutable dictionary populated during the wrapped forward pass.
    """
    captured: dict[str, torch.Tensor | None] = {
        "vit_attention": None,
        "cross_attention": None,
    }
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    vit_self_attention = model.visual_encoder.vit.encoder.layer[-1].attention.attention

    def vit_attention_hook(
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        hidden_states = inputs[0]
        captured["vit_attention"] = _compute_vit_attention_probs(
            module,
            hidden_states,
        ).detach()

    hooks.append(vit_self_attention.register_forward_hook(vit_attention_hook))

    def cross_attention_hook(
        _module: torch.nn.Module,
        _inputs: tuple[Any, ...],
        output: tuple[torch.Tensor, torch.Tensor | None],
    ) -> None:
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            weights = output[1].detach()
            if weights.dim() == 4:
                weights = weights.mean(dim=1)
            captured["cross_attention"] = weights

    hooks.append(
        model.fusion.cross_attention.register_forward_hook(cross_attention_hook)
    )

    try:
        yield captured
    finally:
        for hook in hooks:
            hook.remove()


def extract_attention_maps(
    model: torch.nn.Module,
    dataloader_batch: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """
    Capture ViT self-attention and cross-modal attention for one batch.

    Args:
        model: Trained VTCF model.
        dataloader_batch: Batch dictionary from the VTCF DataLoader.

    Returns:
        Dictionary with ViT and cross-attention tensors.
    """
    model.eval()
    pixel_values = dataloader_batch["pixel_values"]
    batch_size, num_frames = pixel_values.shape[:2]

    with torch.no_grad():
        with _attention_capture_context(model) as captured:
            model(
                input_ids=dataloader_batch["input_ids"],
                attention_mask=dataloader_batch["attention_mask"],
                pixel_values=pixel_values,
            )

    vit_attention = captured["vit_attention"]
    cross_attention = captured["cross_attention"]

    if vit_attention is None:
        raise RuntimeError("ViT attention weights were not captured from the forward hook.")

    if cross_attention is None:
        raise RuntimeError("Cross-attention weights were not captured from the forward hook.")

    num_heads = vit_attention.shape[1]
    cls_to_patches = vit_attention[:, :, 0, 1 : 1 + NUM_VIT_PATCHES]
    cls_to_patches = cls_to_patches.reshape(
        batch_size,
        num_frames,
        num_heads,
        VIT_GRID_SIZE,
        VIT_GRID_SIZE,
    )

    return {
        "vit_attention": cls_to_patches,
        "cross_attention": cross_attention,
    }


def _cls_spatial_attention_map(
    vit_attention: torch.Tensor,
    frame_index: int,
) -> np.ndarray:
    """
    Build a 224x224 spatial heatmap from ViT CLS-to-patch attention.

    Args:
        vit_attention: Tensor shaped [K, num_heads, grid, grid] or
            [num_heads, grid, grid] for one sample.
        frame_index: Frame index within the K-frame sequence.

    Returns:
        Upsampled attention map as a numpy array of shape [224, 224].
    """
    if vit_attention.dim() == 4:
        frame_attention = vit_attention[frame_index]
    elif vit_attention.dim() == 3:
        frame_attention = vit_attention
    else:
        raise ValueError(
            f"Expected vit_attention with 3 or 4 dims, got shape {tuple(vit_attention.shape)}"
        )

    cls_to_patch = frame_attention.mean(dim=0)
    grid = cls_to_patch.detach().cpu().float().numpy()
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)

    heatmap = cv2.resize(grid, (224, 224), interpolation=cv2.INTER_CUBIC)
    return heatmap


def _load_frame_image(frames_dir: Path | str, video_id: str, frame_index: int) -> np.ndarray:
    """Load one RGB frame as a numpy array."""
    frames_dir = Path(frames_dir)
    direct_path = frames_dir / f"frame_{frame_index}.png"
    nested_path = frames_dir / video_id / f"frame_{frame_index}.png"
    frame_path = direct_path if direct_path.exists() else nested_path
    if not frame_path.exists():
        logger.warning("Missing frame %s; using blank placeholder.", frame_path)
        return np.zeros((224, 224, 3), dtype=np.uint8)

    image = Image.open(frame_path).convert("RGB")
    return np.array(image.resize((224, 224)))


def visualize_temporal_attention(
    frames_dir: Path | str,
    video_id: str,
    vit_attention: torch.Tensor,
    cross_attention: torch.Tensor,
    tds_score: float,
    save_path: Path | str,
) -> None:
    """
    Visualize per-frame ViT spatial attention and cross-modal frame importance.

    Args:
        frames_dir: Directory containing extracted frame PNGs.
        video_id: Video identifier subdirectory name.
        vit_attention: ViT attention for one sample, shaped [K, num_heads, 14, 14]
            or [num_heads, 14, 14] when K=1.
        cross_attention: Cross-attention weights shaped [T, K] or [K].
        tds_score: Temporal Divergence Score for title coloring.
        save_path: Output PNG path.
    """
    if vit_attention.dim() == 3:
        num_frames = 1
        vit_attention = vit_attention.unsqueeze(0)
    else:
        num_frames = vit_attention.shape[0]

    if cross_attention.dim() == 2:
        cross_magnitudes = cross_attention.abs().mean(dim=0).detach().cpu().numpy()
    else:
        cross_magnitudes = cross_attention.abs().detach().cpu().numpy()

    cross_magnitudes = cross_magnitudes[:num_frames]

    figure, axes = plt.subplots(
        1,
        num_frames + 1,
        figsize=(4 * (num_frames + 1), 4),
        constrained_layout=True,
    )
    if num_frames + 1 == 1:
        axes = [axes]

    title_color = "red" if tds_score > 0.5 else "green"
    figure.suptitle(
        f"Video {video_id} | TDS: {tds_score:.3f}",
        fontsize=14,
        color=title_color,
        weight="bold",
    )

    for frame_index in range(num_frames):
        frame = _load_frame_image(frames_dir, video_id, frame_index)
        heatmap = _cls_spatial_attention_map(vit_attention, frame_index)

        axis = axes[frame_index]
        axis.imshow(frame)
        axis.imshow(heatmap, cmap="jet", alpha=0.5)
        axis.set_title(f"Frame {frame_index}")
        axis.axis("off")

    bar_axis = axes[-1]
    frame_labels = [f"Frame {index}" for index in range(len(cross_magnitudes))]
    bar_axis.bar(frame_labels, cross_magnitudes, color="steelblue")
    bar_axis.set_title("Cross-Attention Magnitude")
    bar_axis.set_ylabel("Mean |attention|")
    bar_axis.tick_params(axis="x", rotation=20)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved temporal attention visualization to %s", save_path)


def _normalize_label_name(label: Any) -> str:
    """Normalize label strings for summary statistics."""
    if pd.isna(label):
        return "unknown"

    text = str(label).strip().lower().replace("-", "_")
    if "non" in text and "clickbait" in text:
        return "non_clickbait"
    if "not" in text and "clickbait" in text:
        return "non_clickbait"
    if "clickbait" in text:
        return "clickbait"
    return text


def _assign_tds_bucket(tds_score: float) -> str:
    """Map a TDS score to a categorical bucket."""
    if tds_score > 0.7:
        return "high"
    if tds_score >= 0.3:
        return "medium"
    return "low"


def run_tds_analysis_from_csv(
    verified_csv_path: Path | str,
    output_dir: Path | str | None = None,
    label_column: str = "label",
    tds_column: str = "tds_score",
) -> pd.DataFrame:
    """
    Analyze precomputed TDS scores from verified_live_videos.csv.

    Filters rows with valid labels and TDS, prints class means, runs Mann-Whitney U,
    and saves a histogram to outputs/visualizations/tds_distribution.png.
    """
    from scipy import stats

    verified_csv_path = Path(verified_csv_path)
    output_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "outputs" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(verified_csv_path)
    if tds_column not in dataframe.columns:
        raise ValueError(f"Column '{tds_column}' not found in {verified_csv_path}")

    working = dataframe.copy()
    working[label_column] = working[label_column].map(_normalize_label_name)
    working = working[working[label_column].isin(["clickbait", "non_clickbait"])]
    working = working[working[tds_column].notna()]

    if working.empty:
        logger.warning("No rows with valid labels and TDS scores.")
        return working

    working["tds_bucket"] = working[tds_column].astype(float).map(_assign_tds_bucket)

    print("\n=== TDS Summary by Class (CSV) ===")
    for label_name, group in working.groupby(label_column):
        print(f"{label_name:15s} mean_tds={group[tds_column].mean():.4f} n={len(group)}")

    clickbait_scores = working.loc[working[label_column] == "clickbait", tds_column].astype(float)
    non_clickbait_scores = working.loc[
        working[label_column] == "non_clickbait", tds_column
    ].astype(float)

    if len(clickbait_scores) > 0 and len(non_clickbait_scores) > 0:
        statistic, p_value = stats.mannwhitneyu(
            clickbait_scores,
            non_clickbait_scores,
            alternative="two-sided",
        )
        direction = "clickbait" if clickbait_scores.mean() > non_clickbait_scores.mean() else "non_clickbait"
        print(
            f"\nMann-Whitney U (two-sided): U={statistic:.1f}, p={p_value:.2e} "
            f"| higher mean: {direction}"
        )
    else:
        p_value = float("nan")
        print("\nMann-Whitney U skipped: missing class in one group.")

    histogram_path = output_dir / "tds_distribution.png"
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(
        non_clickbait_scores,
        bins=40,
        alpha=0.6,
        label=f"non_clickbait (n={len(non_clickbait_scores)})",
        color="green",
    )
    axis.hist(
        clickbait_scores,
        bins=40,
        alpha=0.6,
        label=f"clickbait (n={len(clickbait_scores)})",
        color="red",
    )
    axis.set_xlabel("TDS score")
    axis.set_ylabel("Count")
    axis.set_title("Temporal Divergence Score by Class")
    if not np.isnan(p_value):
        axis.text(
            0.98,
            0.95,
            f"Mann-Whitney p={p_value:.2e}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=10,
        )
    axis.legend()
    figure.tight_layout()
    figure.savefig(histogram_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved TDS histogram to %s", histogram_path)

    return working


def run_tds_analysis(
    verified_csv_path: Path | str,
    embeddings_dict: dict[str, torch.Tensor],
    predictions_dict: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Compute dataset-level TDS statistics from stored frame embeddings.

    Args:
        verified_csv_path: Path to verified_live_videos.csv.
        embeddings_dict: Mapping of video_id to embeddings shaped [K, 768].
        predictions_dict: Optional mapping of video_id to model predictions.

    Returns:
        DataFrame with per-video TDS analysis columns.
    """
    dataframe = pd.read_csv(verified_csv_path)
    predictions_dict = predictions_dict or {}

    rows: list[dict[str, Any]] = []
    for video_id, embedding in embeddings_dict.items():
        if video_id not in set(dataframe["video_id"].astype(str)):
            continue

        if embedding.dim() == 2:
            embedding = embedding.unsqueeze(0)

        tds_score = float(compute_tds(embedding)[0].item())
        metadata = dataframe[dataframe["video_id"].astype(str) == str(video_id)].iloc[0]

        rows.append(
            {
                "video_id": str(video_id),
                "tds_score": tds_score,
                "label": metadata.get("label", ""),
                "prediction": predictions_dict.get(str(video_id), np.nan),
                "tds_bucket": _assign_tds_bucket(tds_score),
            }
        )

    results = pd.DataFrame(rows)
    if results.empty:
        logger.warning("No overlapping video IDs found for TDS analysis.")
        return results

    print("\n=== TDS Summary by Class ===")
    for label_name, group in results.groupby(results["label"].map(_normalize_label_name)):
        print(f"{label_name:15s} mean_tds={group['tds_score'].mean():.4f} n={len(group)}")

    return results


def ambiguous_text_subset(
    predictions_df: pd.DataFrame,
    threshold: float = 0.55,
) -> pd.DataFrame:
    """
    Extract samples where a text-only model was near-random.

    Args:
        predictions_df: DataFrame with columns [video_id, text_only_confidence, label].
        threshold: Upper bound for the ambiguous confidence band.

    Returns:
        Filtered DataFrame containing ambiguous samples.
    """
    lower_bound = 1.0 - threshold
    upper_bound = threshold
    ambiguous = predictions_df[
        predictions_df["text_only_confidence"].between(lower_bound, upper_bound, inclusive="both")
    ].copy()

    total = len(predictions_df)
    subset_count = len(ambiguous)
    percentage = (subset_count / total * 100.0) if total else 0.0
    print(f"Ambiguous subset: {subset_count} samples ({percentage:.1f}% of total)")
    return ambiguous


def _frame_attention_weights(attention_weights: torch.Tensor) -> np.ndarray:
    """Reduce cross-modal attention to per-frame magnitudes."""
    if attention_weights.dim() == 2:
        magnitudes = attention_weights.abs().mean(dim=0)
    elif attention_weights.dim() == 1:
        magnitudes = attention_weights.abs()
    else:
        raise ValueError(
            f"Unsupported attention_weights shape: {tuple(attention_weights.shape)}"
        )
    values = magnitudes.detach().cpu().float().numpy()
    if values.size == 0:
        return values
    total = values.sum()
    if total > 0:
        return values / total
    return values


def _format_detection_label(label: Any) -> str:
    """Map numeric or string labels to display names."""
    if pd.isna(label):
        return "Unknown"
    if isinstance(label, (int, float, np.integer, np.floating)):
        return "Clickbait" if int(label) == 1 else "Not Clickbait"
    text = str(label).strip().lower().replace("-", "_")
    if "non" in text and "clickbait" in text:
        return "Not Clickbait"
    if "clickbait" in text:
        return "Clickbait"
    return str(label)


def visualize_attention_storyboard(
    video_id: str,
    frames_dir: Path | str,
    attention_weights: torch.Tensor,
    tds_score: float,
    label: Any,
    prediction: Any,
    save_path: Path | str,
) -> None:
    """
    Create a four-panel storyboard: hook, context, delivery frames + attention bars.

    Intended as Figure 3 in the paper.
    """
    frames_dir = Path(frames_dir)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    frame_titles = ["Hook (t=0%)", "Context (t=50%)", "Delivery (t=100%)"]
    bar_colors = ["#FF6B6B", "#FFE66D", "#4ECDC4"]
    bar_labels = ["Hook", "Context", "Delivery"]

    frame_weights = _frame_attention_weights(attention_weights)
    if frame_weights.size < 3:
        padded = np.zeros(3, dtype=np.float32)
        padded[: frame_weights.size] = frame_weights
        frame_weights = padded

    figure, axes = plt.subplots(1, 4, figsize=(15, 4))

    for index in range(3):
        axis = axes[index]
        frame = _load_frame_image(frames_dir, video_id, index)
        axis.imshow(frame)
        axis.set_title(frame_titles[index])
        axis.axis("off")

    bar_axis = axes[3]
    bar_axis.barh(bar_labels, frame_weights[:3], color=bar_colors)
    bar_axis.set_xlim(0, max(float(frame_weights[:3].max()) * 1.2, 0.05))
    bar_axis.set_xlabel("Attention weight")
    bar_axis.set_title("Frame importance")

    label_text = _format_detection_label(label)
    prediction_text = _format_detection_label(prediction)
    title_color = "red" if label_text == "Clickbait" else "green"
    figure.suptitle(
        f"Video: {video_id} | TDS: {tds_score:.3f} | "
        f"True: {label_text} | Pred: {prediction_text}",
        fontsize=12,
        color=title_color,
        weight="bold",
    )
    figure.tight_layout()
    figure.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved attention storyboard to %s", save_path)


def generate_paper_results_table(
    text_only_f1: float,
    vision_only_f1: float,
    full_f1: float,
    hard_subset_f1: float,
    text_only_acc: float,
    vision_only_acc: float,
    full_acc: float,
    mcnemar_p_full_vs_text: float,
    mcnemar_p_full_vs_vision: float,
    clickbait_mean_tds: float | None = None,
    non_clickbait_mean_tds: float | None = None,
    mann_whitney_p: float | None = None,
    output_path: Path | str | None = None,
) -> str:
    """Build, print, and optionally save the paper ablation results table."""
    def _p_text(value: float) -> str:
        if np.isnan(value):
            return "nan"
        if value < 0.001:
            return "p≈0 ✅"
        return f"p={value:.3f}"

    text_p = _p_text(mcnemar_p_full_vs_text)
    vision_p = "p=1.0" if mcnemar_p_full_vs_vision >= 0.999 else f"p={mcnemar_p_full_vs_vision:.3f}"

    table = f"""
┌─────────────────────────────────────────────────────────────┐
│                  VTCF ABLATION RESULTS                      │
├──────────────────────┬──────────┬────────┬──────────────────┤
│ Model                │ Accuracy │ F1     │ McNemar p vs Full│
├──────────────────────┼──────────┼────────┼──────────────────┤
│ Text-Only (BERT)     │ {text_only_acc:7.3f} │ {text_only_f1:6.3f} │ {text_p:>16} │
│ Vision-Only (ViT)    │ {vision_only_acc:7.3f} │ {vision_only_f1:6.3f} │ {vision_p:>16} │
│ Full VTCF            │ {full_acc:7.3f} │ {full_f1:6.3f} │       —          │
│ VTCF Hard Subset     │    —     │ {hard_subset_f1:6.3f} │       —          │
└──────────────────────┴──────────┴────────┴──────────────────┘

TDS Analysis:
Clickbait mean TDS:     {clickbait_mean_tds if clickbait_mean_tds is not None else float('nan'):.3f}
Non-clickbait mean TDS: {non_clickbait_mean_tds if non_clickbait_mean_tds is not None else float('nan'):.3f}
Mann-Whitney p ≈ {mann_whitney_p if mann_whitney_p is not None else float('nan'):.0e} ✅
""".strip()

    table_ascii = f"""
+-------------------------------------------------------------+
|                  VTCF ABLATION RESULTS                      |
+----------------------+----------+--------+------------------+
| Model                | Accuracy | F1     | McNemar p vs Full|
+----------------------+----------+--------+------------------+
| Text-Only (BERT)     | {text_only_acc:7.3f} | {text_only_f1:6.3f} | {text_p:>16} |
| Vision-Only (ViT)    | {vision_only_acc:7.3f} | {vision_only_f1:6.3f} | {vision_p:>16} |
| Full VTCF            | {full_acc:7.3f} | {full_f1:6.3f} |       -          |
| VTCF Hard Subset     |    -     | {hard_subset_f1:6.3f} |       -          |
+-------------------------------------------------------------+

TDS Analysis:
Clickbait mean TDS:     {clickbait_mean_tds if clickbait_mean_tds is not None else float('nan'):.3f}
Non-clickbait mean TDS: {non_clickbait_mean_tds if non_clickbait_mean_tds is not None else float('nan'):.3f}
Mann-Whitney p ~ {mann_whitney_p if mann_whitney_p is not None else float('nan'):.0e}
""".strip()

    rendered = table if os.name != "nt" else table_ascii
    print("\n" + rendered + "\n")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        logger.info("Saved paper results table to %s", output_path)

    return rendered


def _build_dummy_vit_attention(
    batch_size: int,
    num_frames: int,
    num_heads: int = 8,
) -> torch.Tensor:
    """Create synthetic ViT patch attention maps for demo purposes."""
    attention = torch.rand(batch_size, num_frames, num_heads, VIT_GRID_SIZE, VIT_GRID_SIZE)
    return attention / attention.sum(dim=(-2, -1), keepdim=True)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    torch.manual_seed(42)
    batch_size = 2
    num_frames = 3
    hidden_dim = 768
    sequence_length = 16

    print("=== Function 1: compute_tds ===")
    dummy_embeddings = torch.randn(batch_size, num_frames, hidden_dim)
    tds_scores = compute_tds(dummy_embeddings)
    print(f"input shape:  {tuple(dummy_embeddings.shape)}")
    print(f"tds shape:    {tuple(tds_scores.shape)}")
    print(f"tds values:   {tds_scores.tolist()}")

    print("\n=== Function 2: extract_attention_maps ===")
    try:
        from models.fusion_network import VTCF, _build_dummy_config

        model = VTCF(_build_dummy_config())
        model.eval()

        dummy_batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, sequence_length)),
            "attention_mask": torch.ones(batch_size, sequence_length, dtype=torch.long),
            "pixel_values": torch.randn(batch_size, num_frames, 3, 224, 224),
        }

        attention_maps = extract_attention_maps(model, dummy_batch)
        print(f"vit_attention shape:   {tuple(attention_maps['vit_attention'].shape)}")
        print(f"cross_attention shape: {tuple(attention_maps['cross_attention'].shape)}")
    except Exception as exc:
        logger.warning("Skipping live model hook demo: %s", exc)
        attention_maps = {
            "vit_attention": _build_dummy_vit_attention(1, num_frames),
            "cross_attention": torch.softmax(torch.randn(sequence_length, num_frames), dim=-1),
        }
        print("Using synthetic attention maps for remaining demos.")

    print("\n=== Function 3: visualize_temporal_attention ===")
    demo_video_id = "HPa6mRwjUg8"
    demo_frames_dir = PROJECT_ROOT / "data" / "extracted_frames"
    demo_save_path = PROJECT_ROOT / "outputs" / "visualizations" / f"{demo_video_id}_attention.png"

    sample_vit = attention_maps["vit_attention"][0]
    sample_cross = attention_maps["cross_attention"][0]
    visualize_temporal_attention(
        frames_dir=demo_frames_dir,
        video_id=demo_video_id,
        vit_attention=sample_vit,
        cross_attention=sample_cross,
        tds_score=float(tds_scores[0].item()),
        save_path=demo_save_path,
    )
    print(f"saved visualization: {demo_save_path}")

    print("\n=== Function 4: run_tds_analysis_from_csv ===")
    verified_csv = PROJECT_ROOT / "data" / "verified_live_videos.csv"
    if verified_csv.exists():
        csv_tds_df = run_tds_analysis_from_csv(verified_csv_path=verified_csv)
        print(f"CSV-backed rows analyzed: {len(csv_tds_df)}")
    else:
        print("verified_live_videos.csv not found; skipping CSV-backed TDS analysis.")

    print("\n=== Function 5: run_tds_analysis (embeddings) ===")
    predictions_df = pd.DataFrame(
        {
            "video_id": [f"vid_{index}" for index in range(10)],
            "text_only_confidence": [0.10, 0.48, 0.50, 0.52, 0.90, 0.55, 0.44, 0.75, 0.51, 0.20],
            "label": ["clickbait", "clickbait", "non_clickbait", "clickbait"] * 2 + ["clickbait", "clickbait"],
        }
    )
    ambiguous_df = ambiguous_text_subset(predictions_df, threshold=0.55)
    print(ambiguous_df.to_string(index=False))
