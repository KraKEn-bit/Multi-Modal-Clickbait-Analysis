"""VTCF multimodal fusion architecture and multi-task loss."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from transformers import AutoModel, ViTModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path | str) -> dict[str, Any]:
    """Load YAML configuration from disk."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class TemporalVisualEncoder(nn.Module):
    """Encode K temporal frames with a shared ViT backbone."""

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.hidden_size = self.vit.config.hidden_size

        if gradient_checkpointing:
            self.enable_gradient_checkpointing()

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing on the ViT backbone for memory savings."""
        if hasattr(self.vit, "gradient_checkpointing_enable"):
            self.vit.gradient_checkpointing_enable()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode independent frames into a temporal visual matrix.

        Args:
            pixel_values: Tensor of shape [B, K, 3, 224, 224].

        Returns:
            Tensor of shape [B, K, hidden_size].
        """
        batch_size, num_frames = pixel_values.shape[:2]
        flat_pixels = pixel_values.reshape(
            batch_size * num_frames,
            *pixel_values.shape[2:],
        )

        outputs = self.vit(pixel_values=flat_pixels)
        cls_tokens = outputs.last_hidden_state[:, 0, :]
        return cls_tokens.reshape(batch_size, num_frames, self.hidden_size)


class TextEncoder(nn.Module):
    """Encode text headlines into token-level hidden states."""

    def __init__(self, model_name: str = "sagorsarker/bangla-bert-base") -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return full token representations for cross-attention queries.

        Args:
            input_ids: Token ids of shape [B, T].
            attention_mask: Attention mask of shape [B, T].

        Returns:
            Tensor of shape [B, T, hidden_size].
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state


class CrossModalAttentionFusion(nn.Module):
    """Fuse text token queries with temporal visual keys/values."""

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_features: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply text-to-vision cross-attention and pool over text tokens.

        Args:
            text_features: Tensor of shape [B, T, hidden_dim].
            visual_features: Tensor of shape [B, K, hidden_dim].

        Returns:
            fused_output: Tensor of shape [B, hidden_dim].
            attention_weights: Tensor of shape [B, T, K].
        """
        attended, attention_weights = self.cross_attention(
            query=text_features,
            key=visual_features,
            value=visual_features,
            need_weights=True,
            average_attn_weights=True,
        )

        if attention_weights.dim() == 4:
            attention_weights = attention_weights.mean(dim=1)

        fused = attended.mean(dim=1)
        fused = self.dropout(self.layer_norm(fused))
        return fused, attention_weights


class DetectionHead(nn.Module):
    """Binary clickbait detection head."""

    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return detection logits of shape [B, 2]."""
        return self.network(features)


def compute_tds_from_embeddings(frame_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute Temporal Divergence Score from per-frame CLS embeddings.

    Args:
        frame_embeddings: Tensor of shape [B, K, 768].

    Returns:
        Tensor of shape [B] with TDS = 1 - cosine_similarity(hook, delivery).
    """
    hook_embedding = F.normalize(frame_embeddings[:, 0, :], p=2, dim=-1)
    delivery_embedding = F.normalize(frame_embeddings[:, -1, :], p=2, dim=-1)
    cosine_similarity = torch.clamp(
        (hook_embedding * delivery_embedding).sum(dim=-1),
        min=-1.0,
        max=1.0,
    )
    return 1.0 - cosine_similarity


class TDSContrastiveLoss(nn.Module):
    """Supervise temporal contradiction between hook and delivery frame embeddings."""

    def __init__(self, margin: float = 0.5) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        hook_embedding: torch.Tensor,
        delivery_embedding: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Push TDS high for clickbait and low for non-clickbait samples.

        Args:
            hook_embedding: Tensor [B, 768].
            delivery_embedding: Tensor [B, 768].
            labels: Binary detection labels [B], 1=clickbait.

        Returns:
            Scalar contrastive loss.
        """
        hook_norm = F.normalize(hook_embedding, p=2, dim=-1)
        delivery_norm = F.normalize(delivery_embedding, p=2, dim=-1)
        cosine_similarity = torch.clamp(
            (hook_norm * delivery_norm).sum(dim=-1),
            min=-1.0,
            max=1.0,
        )
        tds = 1.0 - cosine_similarity
        labels = labels.float()

        clickbait_loss = labels * (1.0 - tds)
        non_clickbait_loss = (1.0 - labels) * torch.clamp(self.margin - tds, min=0.0)
        return (clickbait_loss + non_clickbait_loss).mean()


class AttributionHead(nn.Module):
    """Multi-label clickbait tactic attribution head."""

    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 4),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return attribution logits of shape [B, 4]."""
        return self.network(features)


class VTCF(nn.Module):
    """Visual-Temporal Contradiction Framework multimodal classifier."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]

        self.visual_encoder = TemporalVisualEncoder(
            model_name=model_cfg["vision_encoder"],
            gradient_checkpointing=model_cfg.get("gradient_checkpointing", False),
        )
        self.text_encoder = TextEncoder(model_name=model_cfg["text_encoder"])
        self.fusion = CrossModalAttentionFusion(
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_heads=int(model_cfg["num_attention_heads"]),
            dropout=float(model_cfg["dropout"]),
        )
        self.detection_head = DetectionHead(
            hidden_dim=int(model_cfg["hidden_dim"]),
            dropout=float(model_cfg["dropout"]),
        )
        self.attribution_head = AttributionHead(
            hidden_dim=int(model_cfg["hidden_dim"]),
            dropout=float(model_cfg["dropout"]),
        )

    @classmethod
    def from_config(cls, config_path: Path | str) -> "VTCF":
        """Instantiate the model from a YAML config file."""
        return cls(load_config(config_path))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Run the full VTCF forward pass.

        Args:
            input_ids: Token ids of shape [B, T].
            attention_mask: Attention mask of shape [B, T].
            pixel_values: Frame tensor of shape [B, K, 3, 224, 224].

        Returns:
            Dictionary containing logits, attention weights, and visual features.
        """
        temporal_visual_matrix = self.visual_encoder(pixel_values)
        text_features = self.text_encoder(input_ids, attention_mask)
        fused_features, attention_weights = self.fusion(
            text_features,
            temporal_visual_matrix,
        )

        return {
            "detection_logits": self.detection_head(fused_features),
            "attribution_logits": self.attribution_head(fused_features),
            "attention_weights": attention_weights,
            "temporal_visual_matrix": temporal_visual_matrix,
            "hook_embedding": temporal_visual_matrix[:, 0, :],
            "delivery_embedding": temporal_visual_matrix[:, -1, :],
            "tds_computed": compute_tds_from_embeddings(temporal_visual_matrix),
        }

    def forward_text_only(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        """
        Text-only ablation path: BERT mean-pool → detection head.

        Skips ViT, fusion, TDS, and attribution. Non-detection outputs are None.
        """
        text_features = self.text_encoder(input_ids, attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled_text = (text_features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        detection_logits = self.detection_head(pooled_text)

        return {
            "detection_logits": detection_logits,
            "attribution_logits": None,
            "attention_weights": None,
            "temporal_visual_matrix": None,
            "hook_embedding": None,
            "delivery_embedding": None,
            "tds_computed": None,
        }

    def configure_text_only_training(self) -> None:
        """Freeze vision/fusion/attribution; train BanglaBERT + detection head only."""
        for parameter in self.visual_encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.fusion.parameters():
            parameter.requires_grad = False
        for parameter in self.attribution_head.parameters():
            parameter.requires_grad = False
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = True
        for parameter in self.detection_head.parameters():
            parameter.requires_grad = True

    def freeze_backbones(self) -> None:
        """Freeze ViT and BanglaBERT backbone parameters."""
        for parameter in self.visual_encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_backbones(self) -> None:
        """Unfreeze ViT and BanglaBERT backbone parameters."""
        for parameter in self.visual_encoder.parameters():
            parameter.requires_grad = True
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = True

    def backbone_parameters(self) -> list[nn.Parameter]:
        """Return parameters belonging to pretrained encoders."""
        return list(self.visual_encoder.parameters()) + list(self.text_encoder.parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        """Return parameters belonging to fusion and task heads."""
        modules = [self.fusion, self.detection_head, self.attribution_head]
        params: list[nn.Parameter] = []
        for module in modules:
            params.extend(list(module.parameters()))
        return params

    def parameter_count(self) -> dict[str, int]:
        """Print and return trainable vs frozen parameter counts."""
        trainable = 0
        frozen = 0

        for parameter in self.parameters():
            num_params = parameter.numel()
            if parameter.requires_grad:
                trainable += num_params
            else:
                frozen += num_params

        total = trainable + frozen
        print(f"VTCF parameters | trainable: {trainable:,} | frozen: {frozen:,} | total: {total:,}")
        return {
            "trainable": trainable,
            "frozen": frozen,
            "total": total,
        }


class VTCFLoss(nn.Module):
    """Combined detection, TDS contrastive, and optional attribution loss."""

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        alpha: float = 1.0,
        beta: float = 0.3,
        gamma: float = 0.1,
        margin: float = 0.5,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.detection_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.contrastive_loss_fn = TDSContrastiveLoss(margin=margin)
        self.attribution_loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        detection_labels: torch.Tensor,
        attribution_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute detection + contrastive (+ optional attribution) losses.

        Returns:
            Dictionary with total_loss, detection_loss, contrastive_loss,
            and attribution_loss scalars.
        """
        detection_loss = self.detection_loss_fn(
            outputs["detection_logits"],
            detection_labels,
        )
        contrastive_loss = self.contrastive_loss_fn(
            outputs["hook_embedding"],
            outputs["delivery_embedding"],
            detection_labels,
        )

        total_loss = self.alpha * detection_loss + self.beta * contrastive_loss
        attribution_loss = torch.zeros((), device=detection_loss.device)

        if attribution_labels is not None and self.gamma > 0:
            attribution_loss = self.attribution_loss_fn(
                outputs["attribution_logits"],
                attribution_labels,
            )
            total_loss = total_loss + self.gamma * attribution_loss

        return {
            "total_loss": total_loss,
            "detection_loss": detection_loss,
            "contrastive_loss": contrastive_loss,
            "attribution_loss": attribution_loss,
        }


def _print_output_shapes(outputs: dict[str, torch.Tensor]) -> None:
    """Print tensor shapes from a model forward pass."""
    print("\n=== VTCF Forward Pass Shapes ===")
    for key, value in outputs.items():
        print(f"{key:25s} shape={tuple(value.shape)} dtype={value.dtype}")


def _build_dummy_config() -> dict[str, Any]:
    """Return a minimal config dictionary for smoke testing."""
    return {
        "model": {
            "text_encoder": "sagorsarker/bangla-bert-base",
            "vision_encoder": "google/vit-base-patch16-224",
            "hidden_dim": 768,
            "num_attention_heads": 8,
            "dropout": 0.1,
            "K_frames": 3,
            "gradient_checkpointing": False,
        }
    }


if __name__ == "__main__":
    torch.manual_seed(42)

    config = _build_dummy_config()
    model = VTCF(config)
    model.eval()

    batch_size = 2
    num_frames = 3
    sequence_length = 64

    dummy_input_ids = torch.randint(0, 1000, (batch_size, sequence_length))
    dummy_attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.long)
    dummy_pixel_values = torch.randn(batch_size, num_frames, 3, 224, 224)

    with torch.no_grad():
        outputs = model(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            pixel_values=dummy_pixel_values,
        )

    _print_output_shapes(outputs)
    model.parameter_count()

    dummy_detection_labels = torch.tensor([1, 0], dtype=torch.long)
    dummy_attribution_labels = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32)
    loss_fn = VTCFLoss(class_weights=class_weights)
    loss_dict = loss_fn(outputs, dummy_detection_labels, dummy_attribution_labels)
    print("\n=== VTCFLoss ===")
    for key, value in loss_dict.items():
        print(f"{key:20s} {value.item():.4f}")

    contrastive_only = TDSContrastiveLoss(margin=0.5)
    contrastive_value = contrastive_only(
        outputs["hook_embedding"],
        outputs["delivery_embedding"],
        dummy_detection_labels,
    )
    print(f"TDSContrastiveLoss (standalone): {contrastive_value.item():.4f}")
    print(f"tds_computed mean: {outputs['tds_computed'].mean().item():.4f}")
