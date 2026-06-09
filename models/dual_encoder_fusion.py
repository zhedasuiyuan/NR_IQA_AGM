"""Dual-encoder cross-attention fusion for NR-IQA.

A second (auxiliary) vision encoder -- e.g. DINOv2/DINOv3 -- is fused into the
SigLIP2 pipeline at the token level. The SigLIP **trunk** (post-final-LN
``last_hidden_state``) is the *query*; the auxiliary encoder's tokens are the
*key/value*. The attended result is added back to the trunk through a zero-init
``out_proj`` residual, so at step 0 the fused output equals the trunk -- i.e.
``native_pool(fused)`` is bit-identical to the vanilla ``get_image_features``
path and the MLP head's input dim is unchanged.

This differs from :mod:`models.multi_layer_fusion` (which fuses several layers
of *one* backbone) in two ways:

  * the key/value come from a **different encoder**, so ``q_dim`` (SigLIP) and
    ``kv_dim`` (aux) generally differ -- ``k_proj``/``v_proj`` map ``kv_dim ->
    q_dim`` so attention runs in SigLIP's space and the residual lands on the
    SigLIP trunk;
  * the aux encoder runs at its own resolution / token count -- cross-attention
    needs no token-count alignment between the two encoders.

The aux encoder is treated as a frozen feature extractor (see ``train.py``); the
only trainable parameters here are the cross-attention block.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .multi_layer_fusion import _unwrap_backbone


def extract_trunk(model: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    """Return the SigLIP ``[B, N, q_dim]`` trunk (post-final-LN ``last_hidden_state``).

    This is the query stream and the same tensor the native pooler is trained on,
    so pooling it reproduces ``get_image_features`` (keeps the dual path step-0
    identical when the fusion residual is zero).
    """
    backbone = _unwrap_backbone(model)
    if hasattr(backbone, "vision_model"):
        out = backbone.vision_model(pixel_values=pixel_values)
    else:
        out = backbone(pixel_values=pixel_values)
    return out.last_hidden_state


def extract_aux_tokens(model: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    """Return the auxiliary encoder's ``[B, M, kv_dim]`` token sequence.

    Uses ``last_hidden_state`` directly (DINOv2/v3 prepend a CLS + register
    tokens; we keep them all as key/value context). Handles both two-tower
    models (``.vision_model``) and plain vision encoders.
    """
    backbone = _unwrap_backbone(model)
    if hasattr(backbone, "vision_model"):
        out = backbone.vision_model(pixel_values=pixel_values)
    else:
        out = backbone(pixel_values=pixel_values)
    return out.last_hidden_state


def aux_hidden_size(model: nn.Module) -> int:
    backbone = _unwrap_backbone(model)
    cfg = backbone.config
    cfg = cfg.vision_config if hasattr(cfg, "vision_config") else cfg
    return cfg.hidden_size


class CrossEncoderFusion(nn.Module):
    """SigLIP trunk (query) attends to a second encoder's tokens (key/value).

    ``out = trunk + drop(out_proj(out_norm(attn(q=trunk, kv=aux))))``.

    With ``gate_init=0.0`` the ``out_proj`` is zero-initialised so the block is a
    step-0 no-op (output == trunk); a small positive ``gate_init`` warm-starts the
    residual (weights ~ ``normal(std=gate_init / sqrt(q_dim))``) so it is not born
    switched off.
    """

    def __init__(self, q_dim: int, kv_dim: int, num_heads: int = 8,
                 dropout: float = 0.0, gate_init: float = 0.0):
        super().__init__()
        if q_dim % num_heads != 0:
            raise ValueError(
                f"query hidden_size {q_dim} is not divisible by aux_num_heads {num_heads}."
            )
        self.q_norm = nn.LayerNorm(q_dim)
        self.kv_norm = nn.LayerNorm(kv_dim)
        self.q_proj = nn.Linear(q_dim, q_dim)
        self.k_proj = nn.Linear(kv_dim, q_dim)
        self.v_proj = nn.Linear(kv_dim, q_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=q_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.out_norm = nn.LayerNorm(q_dim)
        self.out_proj = nn.Linear(q_dim, q_dim)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.out_proj.bias)
        if gate_init == 0.0:
            nn.init.zeros_(self.out_proj.weight)
        else:
            nn.init.normal_(self.out_proj.weight, std=gate_init * q_dim ** -0.5)

    def forward(self, trunk: torch.Tensor, aux_tokens: torch.Tensor) -> torch.Tensor:
        kv = self.kv_norm(aux_tokens.to(trunk.dtype))         # [B, M, kv_dim]
        q = self.q_proj(self.q_norm(trunk))                   # [B, N, q_dim]
        attn_out, _ = self.attn(query=q, key=self.k_proj(kv), value=self.v_proj(kv))
        fused = self.out_proj(self.out_norm(attn_out))
        return trunk + self.drop(fused).to(trunk.dtype)


class DualEncoderFusion(nn.Module):
    """Container for :class:`CrossEncoderFusion` with self-describing save/load.

    ``forward(trunk, aux_tokens)`` returns the fused ``[B, N, q_dim]`` tokens;
    callers pool with :func:`models.multi_layer_fusion.native_pool` (in SigLIP's
    space, so the head input dim is unchanged). The aux ``model_id`` travels with
    the checkpoint config so eval can rebuild the matching aux encoder.
    """

    def __init__(self, *, q_dim: int, kv_dim: int, aux_model_id: str,
                 num_heads: int = 8, dropout: float = 0.0, gate_init: float = 0.0):
        super().__init__()
        self.aux_model_id = aux_model_id
        self.fuser = CrossEncoderFusion(
            q_dim=q_dim, kv_dim=kv_dim, num_heads=num_heads,
            dropout=dropout, gate_init=gate_init,
        )
        self.config = dict(
            q_dim=q_dim, kv_dim=kv_dim, aux_model_id=aux_model_id,
            num_heads=num_heads, dropout=dropout, gate_init=gate_init,
        )

    def forward(self, trunk: torch.Tensor, aux_tokens: torch.Tensor) -> torch.Tensor:
        return self.fuser(trunk, aux_tokens)

    def save(self, path: str) -> None:
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location=None) -> "DualEncoderFusion":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        module = cls(**blob["config"])
        module.load_state_dict(blob["state_dict"])
        return module
