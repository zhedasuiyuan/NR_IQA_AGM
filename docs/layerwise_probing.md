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
- `{dataset}_by_level.csv` — SRCC × layer × probe broken down by **severity
  level** (1–5), always emitted for synthetic sets. Tests whether optimal depth
  depends on distortion *strength* (an IQA-specific axis PE never studied).
- `{dataset}_layerwise.png` — the three curves overlaid (the key figure).
- `{dataset}_heatmap_{group,type,level}.png` — SRCC over (distortion × layer),
  mean-pool. **Distortion-dependent peaks here are the novelty hinge vs PE.**

### Backbones

Default `--model_id` is `google/siglip2-so400m-patch16-512`. The probe is
backbone-agnostic (hidden size auto-detected), so **CLIP** and **DINOv2** work
for the pretraining-objective comparison — the key control being non-contrastive
DINOv2. Notes: for non-SigLIP backbones the `native` probe falls back to mean
pooling (they lack SigLIP's MAP head), so read `mean` and `attention` for those;
processor loading falls back to `AutoImageProcessor` for vision-only DINOv2.
*Code-verified only — run one CLIP/DINOv2 job before trusting a sweep.*

### Usage

```bash
# fast smoke test end-to-end (caps images, few epochs)
python probe_layers.py --dataset KonIQ_10K --max_images 300 --attention --attn_epochs 3

# full run: linear (mean+native) + learned attention probe on ALL layers, all breakdowns
python probe_layers.py --dataset KADID10K --attention
python probe_layers.py --dataset TID2013  --attention

# multi-GPU sweep (round-robins dataset/seed/backbone across GPUs)
./run_probes.sh
SEEDS="42 123 7" ./run_probes.sh                                   # robustness
MODELS="google/siglip2-so400m-patch16-512 facebook/dinov2-large" \
  DATASETS="KADID10K KonIQ_10K" ./run_probes.sh                    # backbone comparison

# restrict the attention probe (memory/time), or pin layers
python probe_layers.py --dataset KADID10K --attention --attn_topk 3
python probe_layers.py --dataset KADID10K --attention --attn_layers "8,16,20,27"
```

Key flags: `--attention` (add the learned probe), `--attn_topk 0` = all layers
(default), `--attn_epochs/--attn_lr/--attn_max_train`, `--breakdown`,
`--max_images` (smoke test).

**Speeding up the attention probe.** It re-runs the frozen backbone every epoch,
which is compute-bound (batch size won't help — the GPU is already saturated).
`--attn_cache_dir /data/probe_cache` forwards the backbone **once**, caching the
tapped layers' tokens as an fp16 memmap, then trains the heads off the cache
(~epochs× fewer forwards; test scored live). Cost: ~0.5 TB disk per dataset for
all 27 layers (63.7 MB/image; the OS page cache makes single-run reads near-RAM
speed). Deleted after the run unless `--keep_cache`. Cheaper alternatives if you
skip caching: fewer `--attn_epochs`, smaller `--attn_max_train`.

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

Vanilla cross-attention aggregation is *not* a contribution on its own -- it is a
common approach in the vision/IQA literature (and MLS is from RAEv2); cite those,
not this repo. So the novelty must come from the analysis/selection. Real hooks:

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
foundation-model features are wrong for IQA, plus the corrective. For the
**top-tier (ICLR/CVPR)** path — breadth beyond IQA + a mechanism — see
*Scaling to a top-tier venue* below (Part 3).

## Part 3 — the per-layer summarization-bottleneck study (`bottleneck_probe.py`)

Part 2 aggregates *whole layers*. Part 3 asks a sharper question one level down:
**how should each layer be summarized before it is fused?**

The motivation is ALF (*"Beyond the Final Layer: Attentive Multi-Layer Fusion"*,
[arXiv:2601.09322](https://arxiv.org/abs/2601.09322), ICML'26): it fuses all
layers but summarizes each one as **CLS + AP** (average-pool) — one or two vectors
per layer — and *explicitly acknowledges* the limitation that "spatial averaging
may neglect fine-grained spatial details ... precise localization." That is
exactly the signal NR-IQA needs (local blur, blocking, banding). So: **does
widening the per-layer summary recover IQA-relevant local-distortion detail?**

### Design (one head, one variable)

A single trainable readout: **per-layer summarizer → +layer embedding →
cross-layer PMA (one learned query) → linear MOS**. Everything downstream of the
summarizer is identical across arms, so the comparison isolates the summarizer.
CLS rides along in *every* arm (`--keep_cls`, default on) as a constant add-on —
so the variable under study is purely the **patch-summary width**:

| arm (`--summarizer`) | per-layer summary | width knob |
|---|---|---|
| `ap` | CLS + mean patch token | — (**== ALF baseline**) |
| `pma` | CLS + `k` learned-query tokens (Set Transformer PMA_k) | `k` = `--width` |
| `tome` | CLS + `r` ToMe-merged tokens (Bolya et al., parameter-free) | `r` = `--width` |

Patch tokens exclude any leading CLS/register tokens (auto-detected as
`Ntok − num_patches`); the prefix is excluded from `pma`/`tome` so ToMe never
merges the out-of-distribution CLS/register token and budget accounting stays
clean. On SigLIP2 (no CLS) the CLS prepend is a no-op, so `ap` == plain mean.

Reads: `pma k=1` vs `ap` = *learned vs mean single patch token* (ALF's motivation,
applied per layer); `k>1`/`r>1` vs `ap` = *does width matter*; `pma` vs `tome` at
matched width = *learned vs parameter-free allocation*.

### Infra

Reuses the Part-1 fp16 all-layer cache. `--cache_dir` is **optional**: empty ⇒
*no cache*, forwarding the frozen backbone on the fly each epoch (only the current
batch on GPU, zero disk) — use when the ~0.5 TB cache is disk-I/O-bound. The build
now flushes periodically to avoid dirty-page OOM on huge caches. `run_bottleneck.sh`
exposes `MODELS` (cross-backbone), `CACHE=0` (no cache), `BS`, `SEED`, and skips a
cache build whose split files already exist.

```bash
# smoke (caps splits, ~15 GB cache)
python bottleneck_probe.py --dataset KADID10K --summarizer pma --width 4 \
  --cache_dir /data/bneck_cache --max_images 800
# full sweep (ap + pma{1,2,4,8} + tome{2,4,8}) x datasets x backbones
./run_bottleneck.sh                          # cached
CACHE=0 BS=8 ./run_bottleneck.sh             # no-cache (backbone forwards per epoch)
```

### Early results (indicative, not final)

- **KADID10K, SigLIP2, 3 seeds:** `pma1` SRCC **0.9056**, `pma2` **0.9050**, vs
  `ap1` **0.8387** — a large +0.067 gain for learned single-query pooling.
  **Caution:** `pma1 ≈ pma2` ⇒ so far the win is *attention-vs-mean pooling per
  layer*, **not** bottleneck *width*. If `pma4/8` and `tome` stay flat, the story
  is "learned per-layer summarization," not "wider summary." Await the full table
  + KonIQ before committing framing. Verify the `ap` arm's val curve converged
  (same epochs/lr/early-stop as `pma`) before trusting the gap.
- **Cross-attention aggregation (Part 2), separate run:** using an **intermediate
  layer (L15) as the attention query** beats the last-layer/trunk query by
  ~0.01 SRCC — a second readout-design choice ALF/vanilla fusion gets wrong here.

## Scaling to a top-tier venue (ICLR / CVPR)

**Honest bar.** IQA-only, however thorough, tops out around WACV/TIP/TMM. Top-tier
needs **breadth + mechanism**: a finding that generalizes past one task and a
*why*, not just a table of wins. The two findings above (learned per-layer
summarization ≫ mean; intermediate-layer query > trunk query, both in the
*frozen*-backbone regime) are the seeds; the following turns them into a
top-tier-shaped paper.

### Two framings

**Framing A — "How to read out frozen vision foundation models" (ICLR-shaped).**
A design-space study of the readout, not an IQA method: per-layer summarizer
`{mean, CLS+AP, PMA-k, ToMe-r}` × cross-layer fusion `{learned query, trunk query,
intermediate-layer query}` × layer set. Our findings are cells; the paper names
the recipe that wins and *why*. Requires **tasks beyond IQA, stratified by
locality** — IQA (local), aesthetics/AVA, distortion-type classification, a dense
linear probe (depth/seg) — with the prediction that *the gain grows with task
locality*. That trend, if it holds, is the memorable law. Evidence bar ≈ ALF's:
3–4 backbones (SigLIP2/CLIP/DINOv2 + a scale point), ~10+ datasets.

**Framing B — frozen-readout IQA method + analysis (CVPR-shaped).** Match/beat
LoRA-finetuned and IQA SOTA (DEIQT, LoDa, TOPIQ) with a **frozen** backbone + tiny
head; sell = efficiency (≈100× fewer trained params) + the probing analysis. Full
7-dataset + cross-dataset tables. Harder: finetuned methods hit ~0.93+ on KADID,
so SOTA tables are brutal — worse expected value than A.

**Recommendation: Framing A.** The evidence already points there (readout choices,
frozen regime, ALF as foil), the query-layer finding folds in naturally, and the
mechanism experiments are cheap on the existing cache infra.

### Mechanism experiments (what lifts it above an empirical recipe)

1. **Locality-controlled synthetic experiment.** Distort a fraction `p` of the
   image; mean-pool signal dilutes ∝ `p`, attention-pool should stay ~invariant.
   Cheap, decisive, one figure — the *why* behind learned-pooling's win.
2. **Attention-map analysis.** Does the learned query attend to the distorted
   regions? Qualitative maps + quantitative attention-mass-on-distorted-patches.
3. **Distortion-type × layer × summarizer breakdown** (Part-1 infra already emits
   the facets).

### Immediate next steps

- Finish the sweep: **KonIQ + `tome` + width grid** — decides "width" vs "learned
  pooling" framing.
- Build the **locality-controlled synthetic** experiment on the cache infra.
- Add **one second task** (AVA aesthetics or distortion-type classification) —
  cheap, reuses the whole pipeline, and is what turns IQA-only into breadth.

## Files

| File | Change |
|---|---|
| `probe_layers.py` | **New.** Frozen-backbone layer-wise probe: `extract_pooled`, `ridge_probe`, `AttnPool` + `attention_probe` (learned attention probe), `DISTORTION_GROUPS` + `_breakdown_facets` (group/type/level breakdown), CSV + matplotlib outputs. CLIP/DINOv2-compatible (processor fallback). |
| `run_probes.sh` | **New.** Multi-GPU sweep: round-robins (model, dataset, seed) jobs across `GPUS`, per-job logs, `--tag` to keep run folders distinct. |
| `bottleneck_probe.py` | **New (Part 3).** Per-layer summarization-bottleneck study: `Readout` (summarizer → cross-layer PMA → linear), summarizers `ap`/`pma`/`tome` (`tome_reduce` = parameter-free bipartite merge), CLS/register auto-detection, cached + live (`train_head` / `train_head_live`) training paths, `model`-tagged results CSV. |
| `run_bottleneck.sh` | **New (Part 3).** Bottleneck sweep across `MODELS × DATASETS × {ap,pma,tome}`; `CACHE=0` no-cache mode, `BS`/`SEED` knobs, cache reuse + exit-code (`rc`) reporting. |
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
