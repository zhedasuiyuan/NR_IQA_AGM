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


def resolve_layer_selection(
    num_hidden_layers: int,
    *,
    stride: int = 4,
    layer_indices: Optional[List[int]] = None,
    first_n: Optional[int] = None,
    last_n: Optional[int] = None,
) -> List[int]:
    """Resolve which ``hidden_states`` block indices (1..H) to tap.

    At most one *override* may be set (they are mutually exclusive); otherwise
    the right-anchored ``stride`` selection is used:

      * ``layer_indices`` -- explicit block indices.
      * ``first_n``       -- the first N blocks  : ``1 .. N``.
      * ``last_n``        -- the last  N blocks  : ``H-N+1 .. H`` (H is the trunk source).

    Index 0 (the patch embedding) is never selectable.
    """
    overrides = {k: v for k, v in (("layer_indices", layer_indices),
                                   ("first_n", first_n), ("last_n", last_n))
                 if v is not None}
    if len(overrides) > 1:
        raise ValueError(
            f"Specify at most one of layer_indices/first_n/last_n; got {sorted(overrides)}."
        )

    if layer_indices is not None:
        idxs = sorted(set(int(i) for i in layer_indices))
    elif first_n is not None:
        if not (1 <= first_n <= num_hidden_layers):
            raise ValueError(f"first_n must be in 1..{num_hidden_layers}, got {first_n}.")
        idxs = list(range(1, first_n + 1))
    elif last_n is not None:
        if not (1 <= last_n <= num_hidden_layers):
            raise ValueError(f"last_n must be in 1..{num_hidden_layers}, got {last_n}.")
        idxs = list(range(num_hidden_layers - last_n + 1, num_hidden_layers + 1))
    else:
        idxs = resolve_fusion_layers(num_hidden_layers, stride)

    bad = [i for i in idxs if not (1 <= i <= num_hidden_layers)]
    if bad:
        raise ValueError(
            f"fusion layer indices {bad} out of range; valid block indices are "
            f"1..{num_hidden_layers} (index 0 is the patch embedding and is excluded)."
        )
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


class TokenSoftMLSFusion(nn.Module):
    """Soft (residual) multi-layer sum: ``out = trunk + alpha * mean_l RMSNorm(F_l)``.

    Unlike :class:`TokenMLSFusion` (which *hard-replaces* the trunk with the
    multi-scale mean), this **keeps the pooler-aligned trunk** and adds the mean
    as a learnable-strength residual -- aiming to retain MLS's cross-dataset gain
    without the within-dataset cost of discarding the trunk.

    The mean is parameter-free, so the only learnable parameter is the scalar
    ``gate`` (=alpha), initialised to ``gate_init`` (0.0 -> step-0 identical).
    Because nothing learnable is gated *behind* ``gate``, its gradient
    (``<dL/dout, MLS>``) is not starved: it grows iff the multi-scale mean
    reduces the loss. A 'safe' fusion that learns exactly how much to use.
    """

    def __init__(self, gate_init: float = 0.0):
        super().__init__()
        self.gate = nn.Parameter(torch.full((1,), gate_init))

    def forward(self, layer_features: List[torch.Tensor], trunk: torch.Tensor) -> torch.Tensor:
        mls = torch.stack([_rms_norm(f) for f in layer_features], dim=0).mean(dim=0)
        return trunk + self.gate.to(trunk.dtype) * mls.to(trunk.dtype)


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
        gate_init: float = 0.0,
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
        # gate_init=0.0 -> step-0 identical (residual off); a small positive value
        # (e.g. 0.1) warm-starts the residual so the gate and the per-layer
        # weights/LNs receive gradient instead of staying switched off.
        self.gate = nn.Parameter(torch.full((1,), gate_init))

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


class TokenCrossAttentionFusion(nn.Module):
    """Cross-attention multi-layer fusion (ported from VisualQuality-R1's
    ``CrossAttentionFusion`` in ``dual_encoder/fusion_block.py``).

    The trunk (post-LN ``last_hidden_state``) is the **query**; the tapped
    layers' tokens, concatenated along the sequence axis, are the **key/value**.
    The attended result is added back to the trunk through an ``out_proj``
    residual. With ``gate_init=0.0`` the ``out_proj`` is zero-initialised, so at
    step 0 the fused output equals the trunk (identical to
    ``get_image_features``); a small positive ``gate_init`` warm-starts the
    residual (its weights are initialised with std ``gate_init / sqrt(dim)``) so
    the block is not born switched off.

    Memory note: the key/value length is ``num_layers * N`` tokens, so attention
    is ``O(N * num_layers * N)``. Reduce the number of tapped layers
    (``--fusion_stride`` / ``--fusion_first_n`` / ``--fusion_last_n``) if memory
    is tight.
    """

    def __init__(self, num_layers: int, dim: int, num_heads: int = 8,
                 dropout: float = 0.0, gate_init: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"hidden_size {dim} is not divisible by fusion_num_heads {num_heads}."
            )
        self.num_layers = num_layers
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        # Residual projection. gate_init=0 -> zero-init -> step-0 no-op; >0 ->
        # warm-start the residual at relative magnitude ~gate_init.
        nn.init.zeros_(self.out_proj.bias)
        if gate_init == 0.0:
            nn.init.zeros_(self.out_proj.weight)
        else:
            nn.init.normal_(self.out_proj.weight, std=gate_init * dim ** -0.5)

    def forward(self, layer_features: List[torch.Tensor], trunk: torch.Tensor,
                query: Optional[torch.Tensor] = None) -> torch.Tensor:
        if len(layer_features) != self.num_layers:
            raise ValueError(f"expected {self.num_layers} layers, got {len(layer_features)}")
        kv = self.kv_norm(torch.cat(layer_features, dim=1))   # [B, L*N, D]
        # Query source: the trunk (final layer) by default, or an intermediate
        # layer's tokens when given -- only changes WHERE attention looks; the
        # result is still residual-added to the trunk (native_pool expects it,
        # and zero-init out_proj keeps step-0 identity regardless of query).
        q_src = trunk if query is None else query
        q = self.q_proj(self.q_norm(q_src))                   # [B, N, D]
        attn_out, _ = self.attn(query=q, key=self.k_proj(kv), value=self.v_proj(kv))
        fused = self.out_proj(self.out_norm(attn_out))
        return trunk + self.drop(fused).to(trunk.dtype)


class TokenALFusion(nn.Module):
    """Attentive Layer Fusion (ALF, arXiv:2601.09322) -- a *different* aggregator
    from :class:`TokenCrossAttentionFusion`, added for a faithful comparison.

    A single learned query cross-attends over per-layer **summary** tokens and
    outputs the pooled feature **directly** (it replaces the backbone pooler, so
    ``MultiLayerFusion.returns_pooled`` is True and ``native_pool`` is skipped).

    How it differs from this repo's cross-attention fusion:
      * query = one learned prototype vector, not the trunk's ``N`` tokens;
      * key/value = one summary token *per layer* (``L`` tokens), not the full
        ``L*N`` spatial tokens;
      * output = a single ``[B, D]`` vector to the head, not a token-level
        residual added back to the trunk (so it is NOT step-0 identical).

    Per-layer summary tokens: ``use_cls=True`` uses ALF's faithful **CLS + AP**
    (token 0 + mean), i.e. 2 tokens/layer -- for CLS-bearing backbones (CLIP,
    DINO). ``use_cls=False`` uses **AP only** (1 token/layer) for SigLIP2, which
    has no CLS token. No pre-attention norm, no residual, per the paper --
    ``nn.MultiheadAttention``'s own ``out_proj`` is ALF's Wout.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0,
                 use_cls: bool = False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"hidden_size {dim} is not divisible by fusion_num_heads {num_heads}."
            )
        self.use_cls = use_cls
        self.query = nn.Parameter(torch.randn(1, 1, dim) * dim ** -0.5)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )

    def forward(self, layer_features: List[torch.Tensor], trunk: torch.Tensor) -> torch.Tensor:
        parts = []
        for f in layer_features:
            ap = f.mean(dim=1, keepdim=True)                 # [B, 1, D] average-pool
            parts.append(torch.cat([f[:, :1], ap], dim=1) if self.use_cls else ap)
        summ = torch.cat(parts, dim=1)                       # [B, (2 or 1)*L, D]
        q = self.query.expand(summ.shape[0], -1, -1).to(summ.dtype)  # [B, 1, D]
        pooled, _ = self.attn(q, summ, summ)                 # [B, 1, D]
        return pooled[:, 0]                                  # [B, D] -> straight to head


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
        fusion_num_heads: int = 8,
        fusion_gate_init: float = 0.0,
        dropout: float = 0.0,
        alf_use_cls: bool = False,
        fusion_query_layer: Optional[int] = None,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.layer_indices = list(layer_indices)
        L = len(self.layer_indices)

        # Optional query source for cross_attention: use a tapped intermediate
        # layer as the attention query instead of the final-layer trunk.
        self.fusion_query_layer = fusion_query_layer
        if fusion_query_layer is not None:
            if fusion_type != "cross_attention":
                raise ValueError("fusion_query_layer is only supported with "
                                 "fusion_type='cross_attention'.")
            if fusion_query_layer not in self.layer_indices:
                raise ValueError(
                    f"fusion_query_layer={fusion_query_layer} must be one of the tapped "
                    f"layers {self.layer_indices}; add it via --fusion_layers."
                )
            self._query_pos = self.layer_indices.index(fusion_query_layer)
        else:
            self._query_pos = None

        if fusion_type == "mls":
            self.fuser: nn.Module = TokenMLSFusion()
        elif fusion_type == "soft_mls":
            self.fuser = TokenSoftMLSFusion(gate_init=fusion_gate_init)
        elif fusion_type == "adaptive":
            self.fuser = TokenAdaptiveFusion(
                num_layers=L, dim=hidden_size,
                conditioning=adaptive_conditioning, norm=adaptive_norm, dropout=dropout,
                gate_init=fusion_gate_init,
            )
        elif fusion_type == "cross_attention":
            self.fuser = TokenCrossAttentionFusion(
                num_layers=L, dim=hidden_size,
                num_heads=fusion_num_heads, dropout=dropout, gate_init=fusion_gate_init,
            )
        elif fusion_type == "alf":
            self.fuser = TokenALFusion(
                dim=hidden_size, num_heads=fusion_num_heads, dropout=dropout,
                use_cls=alf_use_cls,
            )
        else:
            raise ValueError(
                f"fusion_type='{fusion_type}' invalid; expected "
                "'mls', 'soft_mls', 'adaptive', 'cross_attention', or 'alf'"
            )

        # ALF outputs the pooled [B, D] vector directly (replaces native_pool);
        # every other fuser returns [B, N, D] tokens that native_pool then pools.
        self.returns_pooled = fusion_type == "alf"

        # Self-describing config for checkpoint round-trips.
        self.config = dict(
            fusion_type=fusion_type,
            layer_indices=self.layer_indices,
            hidden_size=hidden_size,
            adaptive_conditioning=adaptive_conditioning,
            adaptive_norm=adaptive_norm,
            fusion_num_heads=fusion_num_heads,
            fusion_gate_init=fusion_gate_init,
            dropout=dropout,
            alf_use_cls=alf_use_cls,
            fusion_query_layer=fusion_query_layer,
        )

    @classmethod
    def from_backbone(cls, model: nn.Module, *, fusion_type: str, stride: int = 4,
                      layer_indices: Optional[List[int]] = None,
                      first_n: Optional[int] = None, last_n: Optional[int] = None,
                      **kwargs) -> "MultiLayerFusion":
        """Build from a backbone. Layers are chosen by ``stride`` (right-anchored), or
        by one of the mutually-exclusive overrides ``layer_indices`` / ``first_n`` /
        ``last_n`` over the ``hidden_states`` block indices (1..num_hidden_layers;
        index 0 is the patch embedding)."""
        layer_indices = resolve_layer_selection(
            backbone_num_hidden_layers(model),
            stride=stride, layer_indices=layer_indices, first_n=first_n, last_n=last_n,
        )
        return cls(
            fusion_type=fusion_type,
            layer_indices=layer_indices,
            hidden_size=backbone_hidden_size(model),
            **kwargs,
        )

    def forward(self, layer_features: List[torch.Tensor], trunk: torch.Tensor) -> torch.Tensor:
        if self._query_pos is not None:  # cross_attention with an intermediate query
            return self.fuser(layer_features, trunk, query=layer_features[self._query_pos])
        return self.fuser(layer_features, trunk)

    def save(self, path: str) -> None:
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location=None) -> "MultiLayerFusion":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        module = cls(**blob["config"])
        module.load_state_dict(blob["state_dict"])
        return module
