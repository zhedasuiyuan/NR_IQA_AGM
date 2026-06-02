# Multi-Layer Feature Fusion (MLS + Adaptive Weighting)

Ports the **MLS (multi-layer sum)** and **adaptive layer-weighting** ideas from
VisualQuality-R1 into this repo's pooled-vector NR-IQA pipeline. Works across
plain-ViT backbones — tested design targets **SigLIP2, DINOv2, DINOv3**.

## Motivation

The default pipeline is `backbone → one pooled feature vector → MLP head → score`.
The pooled vector comes only from the **last** layer:

- **SigLIP2**: a learned multi-head attention pooling head (MAP) — a learnable
  probe query cross-attends over all patch tokens, then LayerNorm + MLP. *Not*
  mean, *not* CLS.
- **DINOv2/v3** (via the wrapper fallback): `last_hidden_state.mean(dim=1)`.

Multi-layer fusion lets the head also see **intermediate** layers, which carry
low-level distortion cues (blur, noise, blocking) that the final semantic layer
tends to wash out.

## Design

Fusion happens at the **token level**, *before* pooling:

```
tap layers ─▶ fuse across layers (keep [B, N, D]) ─▶ native pooler ─▶ MLP head
```

Key decisions and why:

1. **Fuse tokens, then pool with the backbone's *native* pooler.** Intermediate
   layers have no pooling head of their own; instead of inventing per-layer
   poolers, we fuse at the token level so the shape stays `[B, N, D]` and the
   existing trained pooler (SigLIP2 MAP / DINO mean) does all the pooling. The
   MLP head's input dim is therefore **unchanged**.
2. **Trunk = post-LayerNorm `last_hidden_state`.** For SigLIP2,
   `last_hidden_state = post_layernorm(hidden_states[-1])`, and the MAP head was
   trained on that. Using it as the residual base is what makes the adaptive
   path step-0 identical.
3. **Zero-init gate ⇒ step-0 identity.** `out = trunk + g · (…)` with `g=0` at
   init means `pool(fused) == get_image_features` bit-for-bit. Adaptive fusion
   is strictly additive at initialisation; training decides if extra layers help.
4. **No `instruction` conditioning.** There is no text prompt in this repo, so
   the VQ-R1 `instruction` mode is dropped; `uniform`/`static`/`image` remain.

### The last block appears in both terms (by design)

The adaptive equation `out = F_last + g · Σ_l w_l · LN_l(F_l)` writes `F_last`
once, but two distinct tensors are involved, and the last block contributes to
**both**:

```
out = last_hidden_state(post-LN)  +  g · Σ_l w_l · LN_l(F_l)
            ▲ residual base                ▲ Σ runs over the tapped layers,
                                             which (right-anchored stride) always
                                             include the last block as
                                             hidden_states[H] — its PRE-final-LN
                                             output, carrying its own learnable LN_H
```

So:

- **Residual base** = the **post-**final-LayerNorm `last_hidden_state` (what the
  pooler is trained on → step-0 identity).
- **In-sum last-block term** = `hidden_states[H]`, the **pre-**final-LayerNorm
  output, with its own learnable `LN_H`.

These differ only by the encoder's final LayerNorm, so the last layer is
represented in both places via two distinct tensors. At step 0 (`g=0`) this has
no effect; after training the aggregate can additionally re-weight a normalized
copy of the last block on top of the trunk. This is intentional and left as-is.

**Difference from VQ-R1:** the VQ-R1 pre-merger path *deduped* the last index out
of the context list and fed the trunk as the single last entry, so there the
trunk appeared once in the sum and was the *same* tensor as the residual base.
This port does **not** dedupe — the right-anchored stride keeps the last block in
the sum (as its pre-LN tensor) while the residual base is the post-LN trunk.

### Token-count requirement

Token-level fusion is position-wise (CLS↔CLS, patch *j*↔patch *j*), so it needs
a backbone that **preserves the sequence length through every block** — true for
SigLIP2, DINOv2, DINOv3. Hierarchical/downsampling backbones (Swin-style) would
misalign and are rejected with a clear error.

## Fusion variants

| `--fusion_type` | Formula | Norm | Step-0 identical? | Lineage |
|---|---|---|---|---|
| `none` | vanilla `get_image_features` | — | n/a (baseline) | current repo |
| `mls` | `mean_l RMSNorm(F_l)` (hard-replace trunk) | param-free RMSNorm | **No** (replaces) | RAE-V2 prior work |
| `adaptive` | `trunk + g · Σ_l w_l · LN_l(F_l)` | learnable per-layer LN + gate | **Yes** (`g=0` at init) | this work |

For `adaptive`, the per-layer weights `w_l` come from `--adaptive_conditioning`:

- **`uniform`** — fixed `1/L` (not learned). Same architecture (learnable LNs +
  gate) with weights held uniform → the step-0-identical *uniform baseline*.
- **`static`** — a learned `[L]` vector, image-independent (≈ ELMo scalar mix).
- **`image`** — `MLP(mean-pooled trunk) → [L]`, image-conditioned (per-image weights).

`--adaptive_norm` is `softmax` (interpretable, sums to 1) or `sigmoid`
(independent gates).

### `mls_gated` was intentionally dropped

`adaptive --conditioning uniform` already provides a step-0-identical uniform
baseline, making a separate `mls_gated` mode a near-duplicate. The two differ
only in (a) residual-add vs. convex blend and (b) learnable LayerNorm vs.
param-free RMSNorm — not worth a third mode. Plain `mls` is kept precisely
because it's a *different lineage* (faithful RAE-V2 baseline for comparison).

The natural ablation reads: `mls` (prior work) vs `adaptive-uniform` (same arch,
weights fixed) vs `adaptive-static`/`image` (weights learned) — each adjacent
pair isolates one factor.

## Layer selection (stride)

`--fusion_stride N` taps **every Nth block, right-anchored on the last block**,
resolved per-backbone from its depth:

- `stride=1` → all blocks
- `stride=N` → `[…, depth]` counting down by `N` (always includes the last block)
- huge stride → just the last block

The raw patch-embedding output (`hidden_states[0]`) is never selected. Example:
SigLIP2-so400m has 27 layers, so `stride=6` → hidden-state indices
`[3, 9, 15, 21, 27]`.

## Files

| File | Change |
|---|---|
| `models/multi_layer_fusion.py` | **New.** Backbone helpers (`extract_token_features`, `native_pool`, `resolve_fusion_layers`), `TokenMLSFusion`, `TokenAdaptiveFusion`, and the `MultiLayerFusion` container with `from_backbone` / `save` / `load`. |
| `models/__init__.py` | Export `MultiLayerFusion`, `extract_token_features`, `native_pool`. |
| `models/wrappers.py` | `SIGLIPWithMLP` takes an optional `fusion`; routes through extract→fuse→pool when set. |
| `train.py` | CLI flags; build fusion from backbone; add params to optimizer; route training forward; `accelerate.prepare`; save `fusion.pt` at every checkpoint site + resume state; pass fusion to `evaluate()`. |
| `eval.py`, `eval_checkpoint.py` | Auto-load `fusion.pt` if present and pass to the wrapper. `eval_all.py` inherits this via `run_eval`. |

`fusion.pt` is self-describing (stores config + weights), so eval rebuilds the
module without needing any CLI flags.

## Usage

```bash
# Adaptive, image-independent learned weights (≈ ELMo scalar mix), tap every 4th layer
python train.py --dataset KonIQ_10K \
    --fusion_type adaptive --adaptive_conditioning static --fusion_stride 4

# Image-conditioned weights, sigmoid gates, tap all layers
python train.py --dataset CLIVE \
    --fusion_type adaptive --adaptive_conditioning image --adaptive_norm sigmoid --fusion_stride 1

# RAE-V2 MLS baseline (prior-work comparison)
python train.py --dataset KonIQ_10K --fusion_type mls --fusion_stride 4

# Step-0-identical uniform baseline (ablate: does learning the weights help?)
python train.py --dataset KonIQ_10K --fusion_type adaptive --adaptive_conditioning uniform

# A different backbone (set mlp_input_dim to that backbone's hidden size)
python train.py --dataset KonIQ_10K --model_id facebook/dinov2-base \
    --mlp_input_dim 768 --fusion_type adaptive --adaptive_conditioning static

# Eval: fusion.pt in the checkpoint dir is picked up automatically
python eval_checkpoint.py --checkpoint best_checkpoints/<run_dir> --dataset KonIQ_10K
```

CLI flags (all on `train.py`):

| Flag | Choices | Default | Meaning |
|---|---|---|---|
| `--fusion_type` | `none`, `mls`, `adaptive` | `none` | Fusion strategy (`none` = unchanged baseline) |
| `--fusion_stride` | int ≥ 1 | `4` | Tap every Nth block (right-anchored; `1` = all) |
| `--adaptive_conditioning` | `uniform`, `static`, `image` | `static` | How adaptive weights are produced |
| `--adaptive_norm` | `softmax`, `sigmoid` | `softmax` | Adaptive weight normalisation |

## Limitations / notes

- **Step-0 identity** holds for `adaptive` only; `mls` hard-replaces the trunk.
- **PEFT**: intended for **LoRA** (injected in-place, still routed through) or
  **full FT**. It **bypasses DPT** prompt-tuning (virtual tokens live in the
  PeftModel forward).
- **Multi-GPU**: follows the repo's existing `.module`-style backbone access;
  fusion params are not separately DDP-synced. Fine for the single-GPU/LoRA
  setup; revisit if going multi-GPU.
- `mlp_input_dim` is unchanged by fusion (hidden size preserved, same pooler);
  still set it per backbone (1152 SigLIP2-so400m, 768 DINOv2-base, etc.).

## Verification status

Logic verified against mocked SigLIP-/DINO-like backbones (stride resolution,
step-0 identity across all conditioning×norm combos, gradient flow, per-image
weights, MLS, save/load round-trip, both pooler paths, ragged-token guard). A
real end-to-end forward (`--dry_run` on one dataset) is the recommended first
smoke test on the GPU env to confirm the actual hidden-states/head shapes line up.
