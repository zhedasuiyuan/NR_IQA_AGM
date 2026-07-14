# Layer-Wise Probing of SigLIP for NR-IQA (+ Aggregation Design Plan)

This document covers two coupled pieces of an in-progress paper:

- **Part 1 — a probing study** (`probe_layers.py`): *where* in the SigLIP2
  encoder does NR-IQA-relevant information live, and does the optimal depth
  depend on distortion type?
- **Part 2 — a feature-aggregation method** motivated by Part 1's findings:
  cross-attention / adaptive aggregation over the layers the probe flags.

The two parts are deliberately coupled: the probing insight is meant to
*predict and motivate* the aggregation design, so the method is "the thing the
probe told us to build," not an unmotivated attention block.

## Research framing

The default pipeline pools **only the last layer** (`get_image_features`;
SigLIP2's MAP head). Hypothesis: the contrastive/sigmoid pretraining objective
rewards invariance to low-level appearance, so the top layers *discard exactly
the low-level signal* (blur, noise, compression, banding) that NR-IQA needs.
If so, intermediate layers should probe *better* for IQA, and the pooled-output
default is systematically suboptimal.

### Relation to Perception Encoder (PE)

PE ([arXiv:2504.13181](https://arxiv.org/abs/2504.13181), *"The best visual
embeddings are not at the output of the network"*) established the
best-layer-is-intermediate phenomenon **for high-level semantic tasks**
(detection ~L40, tracking ~L32), attributing it to an attention
**locality→globality** transition, and *solves* it by **retraining/alignment**
(PE-Lang, PE-Spatial) that lifts good features to the output. PE also uses a
**learned attention probe** and notes it *reshapes the layerwise curve*.

We do **not** claim the general phenomenon — PE owns that. Our distinct niche:

| Axis | PE | This work |
|---|---|---|
| Task regime | High-level semantic | **NR-IQA / low-level quality** |
| Resolution | Per *task* | Per **distortion type** (KADID/TID) |
| Mechanism | Attention locality→globality | **Contrastive low-level invariance** |
| Remedy | Retrain encoder to lift features | **Frozen-backbone aggregation** (+ optional IQA finetune) |

So Part 1 is *characterization + mechanism in an unstudied (low-level) regime*,
not a discovery of the phenomenon. Cite PE as motivation, not a competitor.

## Part 1 — the probing study (`probe_layers.py`)

### What it measures

For every vision block `1..H`, pool its `[B, N, D]` tokens to a vector, fit a
**frozen-backbone** probe to predict MOS, and report test **SRCC/PLCC**. The
backbone is never updated — this measures the *information already present* at
each depth.

### Probes (three, side by side)

| Probe (`pooling`) | Pooling → head | Reads | Role |
|---|---|---|---|
| `mean` | mean over tokens → **Ridge** | *linearly decodable* signal | conservative scientific instrument; where signal **lives** |
| `native` | SigLIP2 MAP head → Ridge | fixed attention pool + linear | what the **deployed** pooler recovers; biased toward the final layer (the head was trained there) |
| `attention` | **learned** single-query attention pool + linear, trained per layer | accessible signal under a strong reader | PE-style probe; can **reshape** the curve vs `mean` |

- **Ridge, not bare OLS** — a linear probe *is* a linear fit; Ridge is the
  regularized form. With D=1152 collinear features and small per-distortion
  slices, unregularized OLS overfits and gives noisy/misleading SRCC. `α` is
  picked on the val split per layer; `α→0` recovers plain linear regression.
- **Learned attention probe** — tokens are re-extracted on the fly under
  `no_grad` (frozen backbone), so nothing large is cached (token caching would
  be hundreds of GB). All target-layer heads **share one backbone forward per
  batch**, so probing *all* layers costs the same backbone compute as probing a
  few — only the small heads multiply. Each head early-stops on its own val SRCC.

### Datasets

Uses the repo's `build_splits` (same partitions as training). Synthetic sets are
split **by reference image** (no content leakage). Distortion-type breakdown is
available where the type is encoded in the filename:

| Dataset | Type source | # types | Notes |
|---|---|---|---|
| `KADID10K` | `Ixx_TT_LL.png` → `TT` | 25 | KADID's **7 official** super-categories |
| `TID2013` | `ixx_TT_L.bmp` → `TT` | 24 | grouped by distortion **family** (editable; less canonical than KADID) |
| `KonIQ_10K`, `SPAQ`, `CLIVE` | authentic | — | overall curve only (no synthetic type labels) |

`TID2013` was added to `dataset.py` (standard official layout:
`distorted_images/` + `mos_with_names.txt`, MOS `/9`, higher=better) and
registered in `_DATASET_CTORS` / `_REFERENCE_GROUPED` and `configs/default.py`.

### Outputs (`probe_out/`)

- `{dataset}_layerwise.csv` — per-layer SRCC/PLCC for every probe.
- `{dataset}_by_group.csv` / `{dataset}_by_type.csv` — per-distortion SRCC ×
  layer × probe (`--breakdown {group,type,both}`, default both). Shared schema
  `pooling, layer, label, group, srcc, n`; type rows carry their family tag.
- `{dataset}_layerwise.png` — the three curves overlaid (the key figure).
- `{dataset}_heatmap_{group,type}.png` — SRCC over (distortion × layer),
  mean-pool. **Distortion-dependent peaks here are the novelty hinge vs PE.**

### Usage

```bash
# fast smoke test end-to-end (caps images, few epochs)
python probe_layers.py --dataset KonIQ_10K --max_images 300 --attention --attn_epochs 3

# full run: linear (mean+native) + learned attention probe on ALL layers, both breakdowns
python probe_layers.py --dataset KADID10K --attention --batch_size 8
python probe_layers.py --dataset TID2013  --attention --batch_size 8

# restrict the attention probe (memory/time), or pin layers
python probe_layers.py --dataset KADID10K --attention --attn_topk 3
python probe_layers.py --dataset KADID10K --attention --attn_layers "8,16,20,27"
```

Key flags: `--attention` (add the learned probe), `--attn_topk 0` = all layers
(default), `--attn_epochs/--attn_lr/--attn_max_train`, `--breakdown`,
`--max_images` (smoke test).

### How to read the results

- `mean` is the trustworthy scientific claim (where signal lives). `native` is
  final-layer-biased by construction. `attention` = strong reader.
- If `attention` **flattens** the curve or pulls the peak toward the top, that
  reproduces PE's "attention pooling reshapes the curve" for IQA — a *finding*,
  not a bug; state it.
- Per-**group** SRCC is reliable; per-**type** slices are small (~25 test
  images/type on a 20% split) → noisy, treat as indicative. Firm up later by
  evaluating per-type on the full dataset if needed.

## Part 2 — aggregation design plan

The probe result selects the method. **Do not commit to a method before the
KADID/TID per-distortion curves exist** — that ordering is what keeps Parts 1
and 2 coupled instead of two disjoint mini-papers.

### Decision tree (probing outcome → method)

| If the probe shows… | Then the principled method is… | Existing module |
|---|---|---|
| **Different distortions peak at different depths** | **Distortion/quality-conditioned adaptive aggregation** — input-adaptive per-image layer weights. Static single-layer "lift" (PE) is provably suboptimal here. | `TokenAdaptiveFusion(conditioning="image")` |
| **Uniform mid-network peak**, decays to top | **Static aggregation** recovering intermediate signal; novelty rides on the mechanism, not the module. | `TokenCrossAttentionFusion`, `TokenMLSFusion` |
| **Intermediate layers transfer better cross-dataset** | Frame the method around **generalization** (the most-valued NR-IQA axis). | any; report cross-dataset SRCC |

### Candidate method hooks (novelty beyond vanilla attention)

Vanilla cross-attention aggregation is *not* a contribution on its own (it is
ported from VisualQuality-R1; see `docs/multi_layer_fusion.md`). Real hooks:

1. **Distortion-conditioned weights** — `TokenAdaptiveFusion` image-conditioned
   weights, framed as the *consequence* of distortion-dependent depth. Ablate
   `uniform` vs `static` vs `image` to quantify the gain the probe predicts.
2. **Learned quality-prototype queries** — replace the trunk-as-query in
   `TokenCrossAttentionFusion` with a small set of learned "quality" queries that
   attend across layers, decoupling aggregation from the semantic trunk.
3. **Frozen vs finetuned SigLIP** — see the migration experiment below.

The paper's punch line should be a closed loop: *the probe predicts X helps → we
build the minimal module exploiting X → it helps by ~the predicted amount →
ablating the insight-driven part removes the gain.*

### The finetuning-migration experiment (important if we finetune SigLIP)

If SigLIP is finetuned on MOS (not just frozen + aggregated), re-run the probe on
the **finetuned** backbone. Question: does the IQA peak **migrate toward the
output** (adaptation "lifts" good features, echoing PE), or does the profile
persist (so aggregation is still needed post-finetune)? Either answer is a
striking result and tells us whether aggregation and finetuning are redundant or
complementary. Distinct from PE: PE lifts via large-scale distillation into a
*general* encoder; IQA finetuning is task-specific adaptation on scarce labels.

### Planned experiments / ablations

- Probe curves on **KADID + TID** (synthetic, per-distortion) and
  **KonIQ + SPAQ** (authentic) — three probes each.
- Optional: same probe on **CLIP / DINOv2** to test whether the profile tracks
  the *contrastive objective* (elevates the mechanism from "SigLIP quirk" to
  "consequence of contrastive pretraining").
- Method: aggregation over probe-selected layers vs the `none` baseline; the
  `uniform → static → image` adaptive ladder; frozen vs finetuned.
- **Cross-dataset** SRCC throughout (generalization is the headline axis).

### Target venue / bar

TIP/TMM/WACV-tier: a thorough probing study + a well-motivated aggregation +
honest ablations is sufficient — the probing is the novelty insurance even if the
module is modest. The value is a **mechanistic explanation** of why pooled
foundation-model features are wrong for IQA, plus the corrective.

## Files

| File | Change |
|---|---|
| `probe_layers.py` | **New.** Frozen-backbone layer-wise probe: `extract_pooled`, `ridge_probe`, `AttnPool` + `attention_probe` (learned attention probe), `DISTORTION_GROUPS` + `_groupings` (group/type breakdown), CSV + matplotlib outputs. |
| `dataset.py` | **New** `TID2013` class; registered in `_DATASET_CTORS` and `_REFERENCE_GROUPED`. |
| `configs/default.py` | `TID2013` path in `_make_dataset_paths`. |

Reuses (no changes): `extract_token_features`, `native_pool`,
`backbone_num_hidden_layers`, `backbone_hidden_size` from
`models/multi_layer_fusion.py`; `build_splits` from `dataset.py`.

## Verification status

Offline (no GPU/model/data in the dev env): syntax of all edited files; Ridge
probe recovers a known signal and returns ~0 on noise; `AttnPool` learns a
content-addressable target and has correct output shape; KADID(25)/TID(24)
distortion groupings cover every type exactly once; `_groupings` produces the
right group/type items with family tags; heatmap-matrix assembly incl. the
small-slice skip.

**Not yet run end-to-end** (needs the GPU/data env). First integration test:
the smoke command above. Most likely first-run snags:

- **TID label-file format** — assumed `mos_with_names.txt`; adjust `TID2013` if
  the download differs.
- **Preprocessing** — mirrors `train.py` (`processor(images=...)`, relies on the
  HF processor's `do_rescale`). Flat SRCC across *all* layers ⇒ suspect
  rescaling.
- **Attention-probe memory** — all-layers trains H heads at once (~2–3 GB Adam
  state for H≈27); drop `--batch_size` or use `--attn_topk` if OOM.
