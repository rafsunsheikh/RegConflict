"""DeBERTa-v3-base cross-encoder with two classification heads.

Architecture:
  [CLS] regime_A [SEP] regime_B [SEP]
                    │
              DeBERTa-v3-base
                    │
        last_hidden_state[CLS_index]   (the [CLS] embedding, 768-dim)
                    │
        ┌───────────┴───────────┐
        │                       │
    binary head            typology head
   (Linear → 2 logits)    (Linear → 4 logits)

Both heads are trained jointly:
    loss = binary_loss + typology_weight × masked_typology_loss

The typology loss is masked via PyTorch's ignore_index=-100 sentinel — on
non-conflict examples the typology label is -100, and CrossEntropyLoss
contributes zero gradient for that example to the typology head.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from src.baselines.fine_tuned.data import CONFLICT_TYPES, IGNORE_INDEX

N_BINARY_CLASSES = 2
N_TYPOLOGY_CLASSES = len(CONFLICT_TYPES)


@dataclass
class ModelOutput:
    binary_logits: torch.Tensor      # (B, 2)
    typology_logits: torch.Tensor    # (B, 4)
    binary_loss: torch.Tensor | None
    typology_loss: torch.Tensor | None
    total_loss: torch.Tensor | None


class CrossEncoderConflictDetector(nn.Module):
    """DeBERTa-v3-base + two classification heads.

    Args:
        model_name: HuggingFace model id (default microsoft/deberta-v3-base).
        binary_class_weights: optional 2-tensor (w_neg, w_pos) for weighted CE.
            Default ~6.6 on positive (inverse train frequency).
        typology_weight: scalar multiplier on the typology loss in the total.
        dropout: dropout applied between [CLS] embedding and each head.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        binary_class_weights: tuple[float, float] | None = (1.0, 6.6),
        typology_weight: float = 1.0,
        dropout: float = 0.1,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        # Gradient checkpointing trades ~30% compute for ~40% activation memory —
        # essential on MPS where the shared pool is heavily contended.
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        hidden = self.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.binary_head = nn.Linear(hidden, N_BINARY_CLASSES)
        self.typology_head = nn.Linear(hidden, N_TYPOLOGY_CLASSES)

        if binary_class_weights is not None:
            self.register_buffer(
                "binary_class_weights",
                torch.tensor(binary_class_weights, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.binary_class_weights = None

        self.typology_weight = float(typology_weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        binary_label: torch.Tensor | None = None,
        typology_label: torch.Tensor | None = None,
    ) -> ModelOutput:
        # DeBERTa-v3 uses relative-position bias; token_type_ids is accepted
        # but the model effectively ignores it. We still pass it for analysis.
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = outputs.last_hidden_state[:, 0, :]  # [CLS] embedding
        cls = self.dropout(cls)

        binary_logits = self.binary_head(cls)
        typology_logits = self.typology_head(cls)

        binary_loss = typology_loss = total_loss = None
        if binary_label is not None:
            binary_loss_fn = nn.CrossEntropyLoss(
                weight=self.binary_class_weights if self.binary_class_weights is not None else None
            )
            binary_loss = binary_loss_fn(binary_logits, binary_label)

        if typology_label is not None:
            # ignore_index=-100 (matches IGNORE_INDEX in data.py); non-conflict
            # records carry typology_label=-100 and contribute zero gradient.
            typology_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            typology_loss = typology_loss_fn(typology_logits, typology_label)

        if binary_loss is not None:
            total_loss = binary_loss
            if typology_loss is not None and not torch.isnan(typology_loss):
                total_loss = total_loss + self.typology_weight * typology_loss

        return ModelOutput(
            binary_logits=binary_logits,
            typology_logits=typology_logits,
            binary_loss=binary_loss,
            typology_loss=typology_loss,
            total_loss=total_loss,
        )
