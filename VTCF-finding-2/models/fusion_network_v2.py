"""Finding-2 fusion network: Title + hook OCR + LLM summary (delivery stream).

Phase 0 generates raw_transcript separately for audit; the trainable model uses
title, hook OCR, and summary — not raw ASR text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path | str) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class TextEncoder(nn.Module):
    """Shared BanglaBERT encoder for title, hook OCR, and delivery summary."""

    def __init__(self, model_name: str = "sagorsarker/bangla-bert-base") -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


class TextStreamFusion(nn.Module):
    """Fuse title, hook OCR, and delivery summary via gating + cross-attention."""

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.title_to_ocr = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ocr_to_summary = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _masked_mean(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        return (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(
        self,
        title_features: torch.Tensor,
        title_mask: torch.Tensor,
        ocr_features: torch.Tensor,
        ocr_mask: torch.Tensor,
        summary_features: torch.Tensor,
        summary_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        title_repr = self._masked_mean(title_features, title_mask)
        ocr_repr = self._masked_mean(ocr_features, ocr_mask)
        summary_repr = self._masked_mean(summary_features, summary_mask)

        ocr_attended, ocr_weights = self.title_to_ocr(
            query=title_features,
            key=ocr_features,
            value=ocr_features,
            need_weights=True,
            average_attn_weights=True,
        )
        context_attended, summary_weights = self.ocr_to_summary(
            query=ocr_attended,
            key=summary_features,
            value=summary_features,
            need_weights=True,
            average_attn_weights=True,
        )

        attended_repr = self._masked_mean(context_attended, title_mask)
        gate_weights = torch.softmax(
            self.gate(torch.cat([title_repr, ocr_repr, summary_repr], dim=-1)),
            dim=-1,
        )

        fused = (
            gate_weights[:, 0:1] * title_repr
            + gate_weights[:, 1:2] * ocr_repr
            + gate_weights[:, 2:3] * summary_repr
            + attended_repr
        ) / 2.0
        fused = self.dropout(self.layer_norm(fused))

        aux = {
            "gate_weights": gate_weights,
            "ocr_attention": ocr_weights,
            "summary_attention": summary_weights,
        }
        return fused, aux


class DetectionHead(nn.Module):
    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class SemanticVTCF(nn.Module):
    """Finding-2: Title + hook OCR + LLM summary semantic fusion."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        self.text_encoder = TextEncoder(model_name=model_cfg["text_encoder"])
        self.fusion = TextStreamFusion(
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_heads=int(model_cfg["num_attention_heads"]),
            dropout=float(model_cfg["dropout"]),
        )
        self.detection_head = DetectionHead(
            hidden_dim=int(model_cfg["hidden_dim"]),
            dropout=float(model_cfg["dropout"]),
        )

    @classmethod
    def from_config(cls, config_path: Path | str) -> "SemanticVTCF":
        return cls(load_config(config_path))

    def forward(
        self,
        title_input_ids: torch.Tensor,
        title_attention_mask: torch.Tensor,
        ocr_input_ids: torch.Tensor,
        ocr_attention_mask: torch.Tensor,
        summary_input_ids: torch.Tensor,
        summary_attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        title_features = self.text_encoder(title_input_ids, title_attention_mask)
        ocr_features = self.text_encoder(ocr_input_ids, ocr_attention_mask)
        summary_features = self.text_encoder(summary_input_ids, summary_attention_mask)
        fused, aux = self.fusion(
            title_features,
            title_attention_mask,
            ocr_features,
            ocr_attention_mask,
            summary_features,
            summary_attention_mask,
        )
        return {"detection_logits": self.detection_head(fused), "fused_features": fused, **aux}

    def forward_title_only(
        self,
        title_input_ids: torch.Tensor,
        title_attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        title_features = self.text_encoder(title_input_ids, title_attention_mask)
        mask = title_attention_mask.unsqueeze(-1).float()
        pooled = (title_features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return {"detection_logits": self.detection_head(pooled)}


class SemanticVTCFLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, labels)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = SemanticVTCF(load_config(DEFAULT_CONFIG_PATH))
    model.eval()
    b = 2
    out = model(
        torch.randint(0, 1000, (b, 64)),
        torch.ones(b, 64, dtype=torch.long),
        torch.randint(0, 1000, (b, 32)),
        torch.ones(b, 32, dtype=torch.long),
        torch.randint(0, 1000, (b, 128)),
        torch.ones(b, 128, dtype=torch.long),
    )
    print("SemanticVTCF OK:", out["detection_logits"].shape)
