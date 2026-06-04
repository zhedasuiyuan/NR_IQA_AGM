"""Multi-layer feature fusion for NR-IQA backbones.

Ports the MLS (multi-layer sum) and adaptive layer-weighting ideas from
VisualQuality-R1 to this repo's pooled-vector pipeline. Unlike the VLM case,
fusion here happens at the **token level**: we tap several hidden layers of the
vision encoder, combine them across layers while keeping the ``[B, N, D]``
token shape, and then let the backbone's **native pooler** (SigLIP2's learned
multi-head attention pooling head, or DINO's mean-over-tokens) run unchanged on
the fused token sequence.

Two consequences of this design:

  * The MLP head's input dim is unchanged -- fusion preserves the hidden size,
    and the same pooler produces the same-shaped vector.
  * With the adaptive zero-init gate, the fused trunk equals ``last_hidden_state``
    at step 0, so ``pool(fused)`` is *bit-identical* to the vanilla
    ``get_image_features`` path. Adaptive fusion is therefore strictly additive
    at initialisation; training decides whether the extra layers help.

Supported backbones: plain-ViT encoders that preserve the token count through
every block (SigLIP2, DINOv2, DINOv3). Hierarchical backbones (Swin-style
downsampling) would misalign tokens across layers and are rejected.

Fusion variants (see the adaptive_layer_fusion design notes in VisualQuality-R1):

  * ``mls``      -- RAE-V2-faithful: parameter-free RMSNorm per layer, mean
                    across layers, *hard-replace* the trunk. NOT step-0 identical.
  * ``adaptive`` -- ``trunk + g * sum_l w_l * LN_l(F_l)`` with a learnable
                    zero-init gate ``g`` and per-layer LayerNorms. Weights ``w_l``
                    come from ``adaptive_conditioning``:
                      - ``uniform`` : fixed 1/L (the step-0-identical uniform
                                      baseline; learnable LNs + gate, fixed weights).
                      - ``static``  : a learned [L] vector (~ELMo scalar mix).
                      - ``image``   : MLP(mean-pooled trunk) -> [L], image-conditioned.
                    ``adaptive_norm`` is ``softmax`` (sums to 1) or ``sigmoid``
                    (independent gates).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


_VALID_CONDITIONING = ("uniform", "static", "image")
_VALID_NORM = ("softmax", "sigmoid")


# ---------------------------------------------------------------------------
# Backbone-aware helpers (stateless): extraction + native pooling
# ---------------------------------------------------------------------------
def _unwrap_backbone(model: nn.Module) -> nn.Module:
    """Peel DDP / PEFT wrappers to reach the HF backbone.

    LoRA injects its adapters *in place* on the target Linear submodules, so
    reaching the base model via ``get_base_model()`` still routes through the
    LoRA-wrapped layers. (DPT prompt-tuning injects virtual tokens in the
    PeftModel forward and would be bypassed here -- multi-layer fusion is
    intended for LoRA / full fine-tuning, not DPT.)
    """
    m = model
    if hasattr(m, "module"):  # DDP / accelerate
        m = m.module
    if hasattr(m, "get_base_model"):  # PEFT
        m = m.get_base_model()
    return m


def _vision_config(backbone) -> object:
    cfg = backbone.config
    return cfg.vision_config if hasattr(cfg, "vision_config") else cfg


def backbone_hidden_size(model: nn.Module) -> int:
    return _vision_config(_unwrap_backbone(model)).hidden_size


def backbone_num_hidden_layers(model: nn.Module) -> int:
    return _vision_config(_unwrap_backbone(model)).num_hidden_layers


def resolve_fusion_layers(num_hidden_layers: int, stride: int) -> List[int]:
    """Right-anchored stride selection over ``hidden_states`` indices.

    ``output_hidden_states=True`` returns ``num_hidden_layers + 1`` tensors:
    index 0 is the raw patch embedding, indices ``1..H`` are the per-block
    outputs. We select block outputs every ``stride`` layers counting *down*
    from the last block, so the final block (the trunk's source) is always
    included and ``stride=1`` selects every block. Index 0 (embeddings) is
    never selected.
    """
    if stride < 1:
        raise ValueError(f"fusion stride must be >= 1, got {stride}")
    last = num_hidden_layers  # hidden_states[H] == last block output
    idxs = sorted(set(range(last, 0, -stride)))
    return idxs


def extract_token_features(
    model: nn.Module, pixel_values: torch.Tensor, layer_indices: List[int]
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Run the vision encoder and return ``(layer_features, trunk)``.

    ``layer_features`` are the ``[B, N, D]`` hidden states at ``layer_indices``;
    ``trunk`` is the post-final-LayerNorm ``last_hidden_state`` (what the native
    pooler is trained on). Keeping the trunk as the post-LN output is what makes
    the adaptive path step-0 identical to ``get_image_features``.
    """
    backbone = _unwrap_backbone(model)
    if hasattr(backbone, "vision_model"):  # SigLIP-style two-tower
        out = backbone.vision_model(pixel_values=pixel_values, output_hidden_states=True)
    else:  # DINOv2 / DINOv3: the model *is* the vision encoder
        out = backbone(pixel_values, output_hidden_states=True)

    hidden_states = out.hidden_states
    n0 = hidden_states[0].shape[1]
    if any(h.shape[1] != n0 for h in hidden_states):
        raise ValueError(
            "Token count varies across layers -- multi-layer token fusion needs "
            "a plain-ViT backbone that preserves the sequence length (SigLIP2, "
            "DINOv2, DINOv3). Hierarchical/downsampling backbones are unsupported."
        )
    feats = [hidden_states[i] for i in layer_indices]
    return feats, out.last_hidden_state


def native_pool(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """Pool ``[B, N, D]`` tokens to ``[B, D]`` using the backbone's own pooler.

    SigLIP2: the learned multi-head attention pooling head. DINO and anything
    without a head: mean over tokens (matches the repo's existing DINO path).
    """
    backbone = _unwrap_backbone(model)
    vm = getattr(backbone, "vision_model", None)
    if vm is not None and getattr(vm, "use_head", False) and getattr(vm, "head", None) is not None:
        return vm.head(tokens)
    return tokens.mean(dim=1)


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Parameter-free RMSNorm over the last dim (RAE-V2 MLS normalisation)."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


# ---------------------------------------------------------------------------
# Fusion modules (operate on batched token tensors [B, N, D])
# ---------------------------------------------------------------------------
class TokenMLSFusion(nn.Module):
    """RAE-V2-faithful multi-layer sum: per-layer RMSNorm, mean across layers,
    hard-replace the trunk. Parameter-free; NOT step-0 identical."""

    def forward(self, layer_features: List[torch.Tensor], trunk: torch.Tensor) -> torch.Tensor:
        normed = torch.stack([_rms_norm(f) for f in layer_features], dim=0)  # [L, B, N, D]
        return normed.mean(dim=0).to(trunk.dtype)


class TokenAdaptiveFusion(nn.Module):
    """Learned weighted-residual aggregation at the token level.

    ``out = trunk + g * sum_l w_l * LN_l(F_l)`` over ``num_layers`` tapped
    layers. ``g`` is a zero-init scalar gate (step 0 == trunk). Weights come
    from ``conditioning``; see module docstring.
    """

    def __init__(
        self,
        num_layers: int,
        dim: int,
        conditioning: str = "static",
        norm: str = "softmax",
        dropout: float = 0.0,
    ):
        super().__init__()
        if conditioning not in _VALID_CONDITIONING:
            raise ValueError(f"conditioning='{conditioning}' invalid; expected {_VALID_CONDITIONING}")
        if norm not in _VALID_NORM:
            raise ValueError(f"norm='{norm}' invalid; expected {_VALID_NORM}")

        self.num_layers = num_layers
        self.conditioning = conditioning
        self.norm = norm

        self.layer_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.gate = nn.Parameter(torch.zeros(1))

        if conditioning == "uniform":
            self.register_buffer("uniform_w", torch.full((num_layers,), 1.0 / num_layers))
            self.alpha = None
            self.weight_net = None
        elif conditioning == "static":
            self.alpha = nn.Parameter(torch.zeros(num_layers))
            self.weight_net = None
        else:  # image
            hidden = max(dim // 4, 128)
            layers: List[nn.Module] = [nn.Linear(dim, hidden), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden, num_layers))
            self.weight_net = nn.Sequential(*layers)
            self.alpha = None

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=-1) if self.norm == "softmax" else torch.sigmoid(x)

    def forward(self, layer_features: List[torch.Tensor], trunk: torch.Tensor) -> torch.Tensor:
        if len(layer_features) != self.num_layers:
            raise ValueError(f"expected {self.num_layers} layers, got {len(layer_features)}")

        # [B, N, L, D]: stack a layer axis just before the feature axis.
        normed = torch.stack(
            [ln(f) for ln, f in zip(self.layer_norms, layer_features)], dim=2
        )

        if self.conditioning == "uniform":
            w = self.uniform_w.to(normed.dtype).view(1, 1, -1, 1)  # [1,1,L,1]
            weighted = (w * normed).sum(dim=2)
        elif self.conditioning == "static":
            w = self._normalise(self.alpha).to(normed.dtype).view(1, 1, -1, 1)
            weighted = (w * normed).sum(dim=2)
        else:  # image: per-image weights from the mean-pooled trunk
            pooled = trunk.mean(dim=1)                       # [B, D]
            w = self._normalise(self.weight_net(pooled))     # [B, L]
            w = w.to(normed.dtype).view(w.shape[0], 1, w.shape[1], 1)  # [B,1,L,1]
            weighted = (w * normed).sum(dim=2)

        return trunk + self.gate.to(trunk.dtype) * weighted.to(trunk.dtype)


# ---------------------------------------------------------------------------
# Container: holds the fusion module + the resolved layer indices, with
# self-describing save/load so eval can rebuild without CLI flags.
# ---------------------------------------------------------------------------
class MultiLayerFusion(nn.Module):
    """Wraps a token-level fusion module together with the tapped layer indices.

    ``forward(layer_features, trunk)`` returns the fused ``[B, N, D]`` tokens;
    callers pool with :func:`native_pool`. Use :meth:`from_backbone` to build
    from a model + stride, and :meth:`save` / :meth:`load` to round-trip a
    checkpoint (the config travels with the weights).
    """

    def __init__(
        self,
        *,
        fusion_type: str,
        layer_indices: List[int],
        hidden_size: int,
        adaptive_conditioning: str = "static",
        adaptive_norm: str = "softmax",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.layer_indices = list(layer_indices)
        L = len(self.layer_indices)

        if fusion_type == "mls":
            self.fuser: nn.Module = TokenMLSFusion()
        elif fusion_type == "adaptive":
            self.fuser = TokenAdaptiveFusion(
                num_layers=L, dim=hidden_size,
                conditioning=adaptive_conditioning, norm=adaptive_norm, dropout=dropout,
            )
        else:
            raise ValueError(f"fusion_type='{fusion_type}' invalid; expected 'mls' or 'adaptive'")

        # Self-describing config for checkpoint round-trips.
        self.config = dict(
            fusion_type=fusion_type,
            layer_indices=self.layer_indices,
            hidden_size=hidden_size,
            adaptive_conditioning=adaptive_conditioning,
            adaptive_norm=adaptive_norm,
            dropout=dropout,
        )

    @classmethod
    def from_backbone(cls, model: nn.Module, *, fusion_type: str, stride: int = 4,
                      layer_indices: Optional[List[int]] = None, **kwargs) -> "MultiLayerFusion":
        """Build from a backbone. Layers are chosen either by ``stride`` (right-anchored
        stride selection) or, when ``layer_indices`` is given, from those explicit
        ``hidden_states`` block indices (1..num_hidden_layers; 0 is the patch embedding)."""
        num_layers = backbone_num_hidden_layers(model)
        if layer_indices is None:
            layer_indices = resolve_fusion_layers(num_layers, stride)
        else:
            layer_indices = sorted(set(int(i) for i in layer_indices))
            bad = [i for i in layer_indices if not (1 <= i <= num_layers)]
            if bad:
                raise ValueError(
                    f"fusion layer indices {bad} out of range; valid block indices are "
                    f"1..{num_layers} (index 0 is the patch embedding and is excluded)."
                )
        return cls(
            fusion_type=fusion_type,
            layer_indices=layer_indices,
            hidden_size=backbone_hidden_size(model),
            **kwargs,
        )

    def forward(self, layer_features: List[torch.Tensor], trunk: torch.Tensor) -> torch.Tensor:
        return self.fuser(layer_features, trunk)

    def save(self, path: str) -> None:
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location=None) -> "MultiLayerFusion":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        module = cls(**blob["config"])
        module.load_state_dict(blob["state_dict"])
        return module
