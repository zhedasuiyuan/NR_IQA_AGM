# Cross-Device ISP Image Quality Assessment under Real Misalignment

Scoping document for a proposed dataset + benchmark + method on **comparing the
image-quality of real renderings produced by different camera / ISP pipelines on
the same scene**, where the two images are **geometrically and photometrically
misaligned** (parallax, FoV/crop, and different color science).

> **Status:** research proposal / positioning. Not implemented in this repo yet.
> Naming TBD — do **not** brand it "Non-Aligned Reference IQA" (taken; see below).

## TL;DR

Comparing two devices' renderings of the same scene is the decisive judgment in
ISP development, yet no dataset or validated metric exists for it: the pairs are
*really* misaligned (parallax, FoV/crop) and differ in color science, so aligned
FR metrics collapse and NR metrics ignore the reference. We propose — gated on a
pilot — (i) a real multi-device capture with a pairwise human study scaled to
JOD, (ii) a benchmark binned by *measured* misalignment, and (iii) **CALM**, a
two-stream metric that matches scene content with quality-invariant features
(partial OT) and compares rendering quality at the matched locations — never
warping pixels, because warping destroys the quality evidence.

**Contributions** (each must ship with its evidence):
- **C1 — Dataset**: ≥150–200 scenes × ≥5 devices, real geometric + photometric
  misalignment, JOD-scaled pairwise labels under fidelity *and* preference
  framings. *Evidence: reliability numbers (study §5), released metadata.*
- **C2 — Benchmark & protocol**: regime binning by measured residual (R0–R4),
  LODO split, human-ceiling-normalized scoring. *Evidence: baseline table.*
- **C3 — Method (CALM)**: structure-addressed, quality-valued, partial-OT
  cross-attention. *Evidence: wins R3–R4 under LODO; ablations.*
- **C4 — Finding**: the quantified fidelity–preference gap at R4 (color science
  is taste, not distortion). *Evidence: dual-framing study.*

## Motivation

When tuning an ISP for a device, the decisive question is comparative: *does our
render of this scene look better than the competitor's?* The two images are
captures of one scene by **different physical devices**, so they differ in
viewpoint (parallax), field of view / crop / resolution, and ISP rendering
(white balance, tone, denoise, sharpening). Classical full-reference (FR) metrics
assume pixel alignment and collapse here; no-reference (NR) metrics ignore the
"vs. a reference" nature of the decision; and the misalignment is *real* (optical
+ rendering), not a synthetic shift. There is no dataset or validated metric for
this exact task.

## Positioning vs. prior work

Classical FR metrics — PSNR, SSIM, and deep successors LPIPS, DISTS, DeepDC —
assume pixel-aligned reference/test pairs and degrade sharply under small shifts.
A line of work relaxes this: **ST-LPIPS** (Ghildyal & Liu, ECCV 2022) hardens
LPIPS against small pixel shifts; **DISTS** and the deep order-statistics metric
**DOSS** tolerate texture resampling and slight deformations for
image-reconstruction / texture-synthesis pairs; **CKDN** handles degraded
references; and, most recently, **NOVA** introduces *non-aligned reference* IQA
for **novel-view synthesis**, training on **synthetic** distortions and
evaluating NeRF / Gaussian-Splatting renders against a nearby, non-aligned frame.
These all target *synthetic* misalignment in rendering or restoration pipelines;
none addresses misalignment from photographing **one scene with different
physical devices**. (Real-world super-resolution IQA also meets misaligned
references — e.g., RealSR-style datasets — but resolves the misalignment by
registration during dataset construction, not in the metric.) On the data side,
cross-device smartphone collections answer
a different question: **SCPQD2020** assigns **no-reference** MOS to individual
photos (no reference image, no cross-device comparison), and **SPCD** measures
perceptual **color** differences only, partly from synthetic edits. Consequently,
the task that is decisive in ISP development — judging which of two *real*,
geometrically and photometrically misaligned device renderings of the same scene
looks better — has neither a dedicated benchmark nor a metric validated for it.

### Axes of difference (Table 1 candidate)

| Work | Images | Misalignment source | Reference | Task | Measures |
|---|---|---|---|---|---|
| LPIPS / DISTS / DeepDC | any | none assumed (aligned) | aligned | FR score | overall fidelity |
| ST-LPIPS | any | **synthetic** small global shifts | aligned-ish | FR score | overall fidelity |
| DOSS, CKDN | reconstruction / texture | **synthetic** shifts / deformation | (degraded) aligned-ish | FR score | overall fidelity |
| NOVA (NAR-IQA) | **synthetic** NeRF/GS renders | rendered viewpoint / synth. distortion | non-aligned frame | comparative | overall quality |
| SCPQD2020 | real phones | — (scored independently) | **none** | **NR** absolute | 4 attributes |
| SPCD | real phones + edits | not addressed | pairwise | pairwise | **color only** |
| **Ours** | **real multi-device ISP** | **real geo + photometric (color science)** | **high-qual. capture / pairwise** | **comparative FR + NR** | **overall perceived quality** |

The surviving novelty claim: **real-device ISP renderings × real geometric +
photometric misalignment × overall-quality comparative judgment** is the cell no
prior row occupies.

**Two non-novelties to *not* claim** (reviewers will catch these):
1. The term *"Non-Aligned Reference IQA"* is taken (NOVA). Rename — e.g.
   *Cross-Device ISP Quality Comparison* / *misalignment-robust FR-IQA for real
   camera renderings*.
2. "FR-IQA under misalignment is a new problem" is **false** (NOVA + DOSS + DISTS
   + ST-LPIPS already do misalignment-robust FR). The novelty is the **real-device ISP
   domain + overall-quality + comparative decision + a dataset that doesn't
   exist**, not the abstract idea.

## Misalignment regimes (benchmark columns)

Bin test pairs into regimes by **measured residual displacement after best-effort
global registration**, so the axis is objective, not eyeballed:

| Regime | Description | Typical residual (post-registration) |
|---|---|---|
| **R0 Aligned** | registered / near-zero residual (control) | ~0 px |
| **R1 Mild** | OIS jitter, burst, sub-pixel–few px | ≤ ~2 px |
| **R2 Parallax** | hand-held viewpoint change, depth-dependent | non-global; flow varies by depth |
| **R3 FoV/crop** | different focal length / zoom / resolution | global scale + crop |
| **R4 Cross-ISP** | different device color science **on top of** geometric shift | R2/R3 geometry + large ΔE |

A separate **color-science-gap sub-split** (by measured ΔE between devices on
aligned patches) isolates "rendering taste" from geometric distortion.

## Baseline benchmark skeleton

Cells = SRCC vs. JOD and 2AFC accuracy (see *Scoring metrics against pairwise
ground truth* below; PLCC/KRCC in appendix). `Ref?` = needs a reference;
`Align?` = uses an explicit alignment preprocessing step.

```
Method family            Ref?  Align?  | R0    R1    R2    R3    R4    | Avg
------------------------------------------------------------------------------
Classic FR
  PSNR                     FR    no     | __    __    __    __    __    | __
  SSIM / MS-SSIM           FR    no     | __    __    __    __    __    | __
  FSIM / GMSD / VIF        FR    no     | __    __    __    __    __    | __
Deep FR (aligned)
  LPIPS                    FR    no     | __    __    __    __    __    | __
  DISTS                    FR    no     | __    __    __    __    __    | __
  DeepDC / PieAPP / AHIQ   FR    no     | __    __    __    __    __    | __
Align-then-FR (pipelines)
  Homography(RANSAC)+LPIPS FR    yes    | __    __    __    __    __    | __
  OpticalFlow-warp+DISTS   FR    yes    | __    __    __    __    __    | __
Misalignment-robust FR
  ST-LPIPS                 FR    no     | __    __    __    __    __    | __
  CKDN                     FR    no     | __    __    __    __    __    | __
  DOSS                     FR    no     | __    __    __    __    __    | __
  NOVA (NAR-IQA, retrained)FR    no     | __    __    __    __    __    | __
No-reference
  NIQE / BRISQUE           NR    n/a    | __    __    __    __    __    | __
  MUSIQ / MANIQA           NR    n/a    | __    __    __    __    __    | __
  CLIP-IQA / Q-Align       NR    n/a    | __    __    __    __    __    | __
  SigLIP2+AGM (this repo)  NR    n/a    | __    __    __    __    __    | __
Comparative / LMM
  Compare2Score            pair  no     | __    __    __    __    __    | __
Ours
  [METHOD] (FR)            FR    no     | __    __    __    __    __    | __
  [METHOD] (NR)            NR    n/a    | __    __    __    __    __    | __
```

**The story each block is expected to land** (the pilot tests these; if a block
fails — e.g., align-then-FR holds up fine at R2 — narrow the claim honestly
rather than force the narrative):
- **Classic FR**: high at R0, *collapses* by R2–R4 → motivates the problem.
- **Align-then-FR**: recovers at R1–R2, **breaks at R3–R4** (parallax/FoV can't
  be homography-aligned) → kills the "just register first" rebuttal.
- **Misalignment-robust FR (ST-LPIPS/DOSS/CKDN/NOVA)**: better than classic, but
  **degrades on real cross-ISP (R4)** → the headline gap evidence.
- **NR**: flat across regimes but **ceiling-limited on fidelity / can't answer
  "which is better vs. a reference"** → NR alone is insufficient.
- **Ours**: holds through R4 in both FR and NR → the contribution.

### Splits & headline metrics

- **Disjoint scenes** between train and test (standard).
- **Leave-one-device-out (LODO)**: additionally hold out *entire devices* at test
  time. A learned comparative metric trained on N devices can simply memorize
  device rendering priors ("device X's look wins"); LODO is the rebuttal-killer
  and the **primary reported number for learned methods**.
- **Human-ceiling-normalized correlation**: report metric SRCC per regime
  *alongside* the split-half human SROCC ceiling (study §5). At R4, taste-driven
  observer disagreement caps achievable SRCC, so raw correlations mislead; the
  headline is the **fraction of the human ceiling achieved**.

### Scoring metrics against pairwise ground truth

The ground truth is pairwise, so each metric is scored in **two modes**:

- **(a) Correlation vs. JOD**: SRCC (PLCC/KRCC in appendix) between metric
  scores and the fitted JOD scale, per scene aggregated per regime.
- **(b) 2AFC accuracy**: does `sign(s(A,B))` agree with the majority human
  choice — the BAPPS / PieAPP evaluation mode, and the more natural number for
  pairwise data. **Exclude or down-weight near-ties** (|ΔJOD| below its
  bootstrap CI); on those pairs disagreement is noise, not error.
- **Significance**: bootstrap CIs on every cell (resample scenes *and*
  observers), and a stated **paired-bootstrap test** for any "X beats Y" claim.
  With ~5 regimes × ~20 methods, an un-tested table is unfalsifiable.

---

# Subjective-study protocol

The dataset's credibility rests almost entirely here. Misaligned pairs are harder
to rate than standard IQA stimuli, so the protocol must be airtight and standards-
compliant (ITU-R BT.500-14 / ITU-T P.910).

## 1. Method: pairwise forced choice (2AFC) as primary

Use **pairwise comparison (2AFC)**, not absolute category rating (ACR), as the
primary instrument:
- More reliable and less anchoring-prone when stimuli are misaligned and differ
  in rendering "taste" — observers compare, they don't calibrate an absolute
  scale across heterogeneous scenes.
- Naturally matches the ISP-decision task ("which is better?").
- Scales cleanly to a continuous quality scale via a psychometric model (§4).
- Precedent: **BAPPS** (the LPIPS dataset) and the **PieAPP** dataset both use
  2AFC — pairwise is the field's standard for perceptual ground truth.

Run **two instruction framings** as separate passes (or a between-subjects split)
to disentangle the confound flagged in review:
- **Fidelity**: "which render is more faithful / less degraded?"
- **Preference**: "which render do you prefer overall?"
Report both; their divergence is itself a finding (rendering taste ≠ distortion).
At R4 the **quantified fidelity–preference gap** is a potential standalone
contribution — the measured statement that color science is taste, not
distortion — so don't bury it in an appendix.
**Budget note**: two framings double the comparison budget. **Preference gets
the full sparse design; fidelity runs on a stratified subset** — enough pairs
per regime to estimate the gap with CIs, not a full second scale.

The **FR variant** additionally shows a **high-quality reference capture** of the
scene (tripod / high-end camera) above the two candidates; the **comparative-NR
variant** shows only the two candidates. Collecting both widens the benchmark and
pre-empts the "is this really FR?" objection.

**Reference role & circularity.** The reference camera has its own color
science, so every candidate-vs-reference comparison is itself an R4 pair. Treat
the reference as **scene-content ground truth, not a quality ideal**: capture
RAW and render with neutral, documented processing. Validate in the pilot that
observers actually prefer the reference over candidates; if they don't, anchor
the JOD scales without it and demote the reference to a content anchor only.

## 2. Stimuli & sampling design

- **Pairs binned by regime R0–R4** (§ regimes) so every regime has enough
  comparisons for per-regime scaling.
- **Full pairwise is O(n²) and infeasible.** Use a **sparse/active comparison
  design**:
  - a **balanced incomplete block design (BIBD)** for coverage, *or*
  - **active sampling** (Crowd-BT / TrueSkill-style adaptive pair selection,
    or HodgeRank-guided) to spend comparisons where the scale is most uncertain.
  - Target a fixed **#comparisons per condition** (pilot-tuned, §6) rather than a
    full matrix; ensure the comparison graph is **connected** within each scene/
    regime so the scale is identifiable.
- **Randomize** left/right placement and presentation order; balance so no device
  is systematically on one side.

## 3. Observers, environment, procedure

- **Observers**: target an *effective* ≥ 15 ratings per pair after screening
  (BT.500 guidance); recruit a pool sized to hit that given the sparse design.
  Mix **naïve** observers (generalizability) with a smaller **expert/photographer**
  cohort (sensitivity); analyze separately.
- **Screening**: visual acuity (Snellen, corrected OK) and color vision
  (Ishihara) — color-science differences make CVD a real confound here.
- **Environment**: ITU-R BT.500-14 — calibrated, profiled display (sRGB/Display-P3
  stated), fixed luminance, controlled ambient light, fixed viewing distance.
  State whether images are shown at native resolution or fit-to-screen (affects
  perceived sharpness/noise — keep it constant).
- **Procedure**: written instructions + a short **training set** with anchor
  examples spanning clearly-better → near-tie; un-timed choice (forced, no "equal"
  option in 2AFC, or a logged tie option mapped appropriately); breaks every
  ~20–30 min to limit fatigue; session ≤ ~45 min.

## 4. From comparisons to a quality scale

- Fit a **Thurstone Case V** / **Bradley–Terry–Luce** model to convert pairwise
  win-rates into a continuous **JOD** (just-objectionable-difference) scale per
  scene/regime (e.g., via Perez-Ortiz & Mantiuk's `pwcmp` scaling). 1 JOD ⇒ 75%
  of observers prefer A over B.
- Report **per-scene** scales (anchored), aggregated to per-regime; keep the
  reference (FR variant) as a fixed anchor where present.
- Provide **bootstrapped confidence intervals** on the JOD scores (resample
  observers) — essential given the sparse comparison design.

## 5. Quality control & reliability

- **Sentinel/gold pairs**: include obviously-different pairs; observers failing
  these are dropped (attention check).
- **Repeat pairs**: re-present ~5–10% of comparisons to measure **intra-observer
  consistency**; drop inconsistent observers.
- **Outlier rejection**: BT.500 observer-rejection procedure, *and/or* model-based
  (observers whose choices disagree systematically with the fitted BTL scale —
  large residuals / low log-likelihood).
- **Inter-observer agreement**: report **Kendall's W** / **Krippendorff's α** on
  the pairwise choices, and **transitivity-violation rate** (fraction of
  intransitive triads) as a misalignment-difficulty diagnostic.
- **Scale reliability**: **split-half SROCC** between two random halves of
  observers (with bootstrap CI) — the headline reliability number. Report it
  **per regime**: it is the human ceiling against which metric SRCC is
  normalized (see *Splits & headline metrics*).

## 6. Pilot (go / no-go gate)

Run a **pilot** (~10–20 scenes, 3–4 devices, ~15 screened observers) **before**
scaling, to:
1. confirm observers rate misalignment-robust **preference reliably**.
   Pre-registered thresholds (set numbers *before* unblinding; adjust only with
   written justification): split-half SROCC **≥ 0.8 on R0–R2**, **≥ 0.6 on R4**;
   transitivity-violation rate **≤ ~15%** per regime;
2. show **align-then-FR + DISTS/DOSS correlate poorly** with the human JOD scale
   on R3–R4, **and** produce the **resampling-penalty demo** (warp an image by a
   known ground-truth geometry; measure each metric against the unwarped
   original). Report it **per metric**: expect LPIPS to pay a large penalty and
   DISTS — *designed* to tolerate resampling — a smaller one. The claim is that
   **warp-based pipelines inherit the penalty**, not that every metric fails
   equally; overclaiming here is easy for a reviewer to falsify;
3. tune **#comparisons per condition** for the target CI width (power analysis on
   the pilot BTL fit);
4. validate the **reference anchor** (observers prefer the neutral-rendered
   reference; see *Reference role & circularity*) and check **DINO matching on
   night/low-texture scenes** (see *Risks*).
The pilot *is* the project's go/no-go and the motivating result.

---

# Capture & misalignment-binning pipeline (concise)

1. **Capture**: shared scenes photographed by N devices; mix **controlled**
   (tripod/rig, same instant where possible) and **in-the-wild** (hand-held).
   Capture a **high-quality reference** per scene (tripod + high-end camera) for
   the FR variant. Log device, focal length, exposure, timestamp.
   **Prefer static scenes** (no people / foliage / water / clouds in motion) or
   capture simultaneously on a rig: seconds-apart hand-held captures of dynamic
   content differ in *content*, not rendering, and contaminate the comparison.
   **Scale floor**: target **≥ 150–200 scenes × ≥ 5 devices**, with the
   comparison budget sized by the pilot power analysis (study §6); reviewers
   will benchmark size against SPCD/SCPQD2020, and below ~50 scenes × 3 devices
   this is a pilot study, not a benchmark.
   **Scene taxonomy (stratified, not uniform)**: daylight outdoor, indoor,
   **night/low-light**, **HDR/backlit**, **zoom/tele**, **portrait/skin tones**
   (with stated skin-tone diversity — reviewers will ask). ISP differences
   concentrate in the last four categories; allocate scenes accordingly.
   **Settings policy** ("the device's rendering" must be well-defined): default
   camera app, full auto; HDR/night mode as auto-triggered (log whether it
   fired); ship-format output (JPEG/HEIC as produced); device model, firmware,
   and app versions logged per capture.
2. **Global registration**: feature matches (SIFT/SuperPoint) + **RANSAC
   homography**; record inlier ratio and whether a global model fits at all
   (parallax/FoV cases will not).
3. **Residual displacement**: after global registration, run **dense optical flow
   (RAFT)**; summarize by **median end-point error (EPE)** and valid-overlap
   ratio → assign **R0–R4** by EPE/overlap thresholds (parallax = non-global flow
   despite low global residual).
4. **Photometric gap**: on the registered-overlap region, measure **ΔE** (CIEDE2000)
   between devices → the color-science sub-split.
5. **Content-change covariate**: flag residual *temporal* change in the overlap
   region (photometric-normalized difference inconsistent with the estimated
   flow); release it alongside EPE/ΔE, and **exclude or separately bin** pairs
   above threshold so dynamic-content differences don't masquerade as rendering
   differences.
6. **Release** both raw captures and the registration metadata (homography, flow
   stats, EPE, ΔE, content-change score, regime label) so the benchmark binning
   is reproducible.

---

# Method sketch: a correspondence-coupled comparative metric

Working name: **CALM** (Correspondence-Aligned, alignment-free Local-quality
Matching). The thesis in one line: *don't warp pixels — establish
quality-invariant soft correspondences, then compare perceptual quality both at
matched locations and as alignment-free distributions.* This generalizes DOSS
(order statistics = a 1-D alignment-free distribution comparison) to learned,
OT-coupled, **two-stream** features, and — unlike NOVA/DOSS — is trained for
**real device misalignment** and **comparative** judgment.

**The falsifiable claim that motivates the design:** align-then-compare is not
merely brittle — **warping destroys the quality evidence itself**. Resampling
during warping alters exactly the statistics being judged (noise texture,
sharpening halos, demosaic artifacts), so even a *geometrically perfect*
registration corrupts the measurement. One-figure demo (run in the pilot, study §6):
warp an image by a known ground-truth geometry and compare against the
unwarped original — LPIPS/DISTS assign a large penalty to the resampling alone.
This upgrades "no homography / no warping" from an engineering choice to a
necessity, and it is the paper's central technical thesis.

## Core idea: separate "where" (ignore) from "how good" (measure)

Misalignment moves scene content around the frame; quality is about *local
degradation statistics* and *global rendering*. So we use two streams:

- **Structural stream** `S = Enc_struct(·) ∈ R^{N×d}` — for *matching*. Must be
  **quality-invariant** so corresponding scene points match regardless of which
  device rendered them. A **frozen DINO** encoder gives this almost for free
  (augmentation/degradation-robust structural features — exactly the property our
  dual-encoder analysis found DINO strong at). Reuses
  [`dual_encoder_fusion.py`](../models/dual_encoder_fusion.py).
- **Quality stream** `Q = Enc_qual(·) ∈ R^{N×c}` — for *comparison*. Carries
  blur/noise/sharpening/exposure/color cues. Instantiate as SigLIP2 + the AGM
  pipeline (this repo), with low-level conv features concatenated.

**Native-resolution caveat — the method's own version of the warping problem.**
SigLIP2 ingests 384–512 px; noise texture, sharpening halos, and demosaic
artifacts live at *native* resolution and are erased by global downsampling —
the same "destroying the evidence" failure we charge align-then-FR with. A
stated strategy is required: **native-resolution patch sampling** (NaFlex
variable-resolution input, or quality features pooled from native-res crops)
and/or **multi-scale low-level conv features** computed at native resolution.
The "low-level conv features" above is load-bearing, not optional; ablate it.

## Step 1 — Structure-addressed, quality-valued cross-attention (partial-OT-normalized, no homography)

One module instantiates "separate *where* from *how good*" — present it as a
single cross-attention design, not three assembled parts:

```
A_ij  = (S_A^i · S_B^j) / √d        # attention logits on the STRUCTURAL stream
                                    #   (quality-invariant addressing: "where")
T     = UnbalancedSinkhorn(−A; ε, τ)# normalization: PARTIAL OT (KL-relaxed
                                    #   marginals / dustbin), not softmax
Q̃_A^i = Σ_j T_ij · Q_B^j            # values from the QUALITY stream ("how good")
```

This is cross-attention with **two deliberate substitutions** relative to its
standard FR-IQA usage (IQT, AHIQ) — and the substitutions, not the attention,
are the claim:

1. **Cross-stream Q/K vs. V**: attention logits come from the structural
   encoder (frozen DINO); the aggregated *values* come from the quality encoder
   (SigLIP2 + AGM, this repo). IQT/AHIQ compute both from one backbone, so
   their addressing is contaminated by the very rendering differences being
   judged.
2. **Partial-OT normalization vs. softmax**: softmax is a *one-sided* matching
   that forces every token to attend somewhere — garbage matches exactly in the
   non-overlap regions of R3 (FoV/crop). A few Sinkhorn iterations on the
   logits (Sinkformer-style, made **unbalanced**: KL-relaxed marginals with
   mass parameter `τ`, or a SuperGlue-style dustbin) let unmatched mass drop
   out; the dropped-mass map doubles as a *valid-overlap estimate* for free.

`T_ij` is then a soft, many-to-many correspondence plan: differentiable, and it
tolerates parallax / FoV / crop where a single homography cannot. No positional
prior is required (positions genuinely differ); add a weak one only if pilots
show drift (e.g., repeated-texture mismatches).

## Step 2 — Compare quality two ways (both alignment-free)

1. **Correspondence-aligned local difference** (signed, carries *direction*):
   ```
   d_local = Σ_ij T_ij · w(Q_A^i, Q_B^j)        # w: learned signed relative-quality
           ≈ Σ_i  w(Q_A^i, Q̃_A^i)               # each token vs. its attention-
                                                #   aggregated counterpart (Step 1);
                                                #   exact iff w linear in 2nd arg —
                                                #   the cheap variant to implement
   ```
   Comparing quality descriptors at *matched* scene points isolates rendering
   quality from content.
2. **Alignment-free distribution distance** (symmetric, carries *magnitude*):
   ```
   d_set = SlicedWasserstein_2({Q_A^i}_i, {Q_B^j}_j)
   ```
   A set-level comparison of the two quality-token distributions — robust when OT
   correspondences are noisy under heavy parallax. This is the learned, multi-dim
   generalization of DOSS's order-statistics.

## Step 3 — Comparative head, antisymmetric by construction

The metric must satisfy `s(A,B) = −s(B,A)`. Build it that way:
```
ψ(A,B) = MLP_AGM([ d_local(A,B) ; pool(Q_A) − pool(Q_B) ])    # signed branch
s(A,B) = ψ(A,B) − ψ(B,A) + γ · sign-free(d_set)               # antisymmetric
```
`d_set` is symmetric (a distance), so it contributes *magnitude of disagreement*,
while the signed branches set the *direction*. Reuse the gated **AGM head** as
`MLP_AGM`.

**FR vs. NR unification.** Define a misalignment-robust relative quality
`r(X | Ref)` by running Steps 1–2 between candidate `X` and the reference capture
`Ref`. Then:
- **FR mode**: `s(A,B) = r(A|Ref) − r(B|Ref)`.
- **Comparative-NR mode**: `s(A,B)` directly (Ref omitted; Siamese over A,B).
One architecture, two settings — matches the dual-protocol dataset.

## Training: synthetic pretrain → real fine-tune

**Stage 1 — synthetic, label-free (large).** Take an image, apply a *known*
geometric warp `T_geo` and a degradation `d`. This hands you, for free:
- ground-truth correspondences (from `T_geo`) → supervise the OT plan;
- a ground-truth quality order (more degradation ⇒ worse) → supervise quality;
- additionally apply a **synthetic color-science gap** to one side (random tone
  curves, white-balance shifts, saturation, sharpening profiles) — *without* a
  quality label (taste has no ground-truth order). This exposes matching and
  quality streams to R4-style photometric gaps; otherwise pretraining covers
  only R0–R3 geometry and the headline regime is left entirely to the small
  real dataset.

**Stage 2 — real, human-labeled (small).** Fine-tune on the pairwise JOD dataset.

**Losses:**
```
L = L_rank            # human pairwise: −log σ( y · s(A,B) ),  y = ±1 (BT/logistic)
  + λ1 · L_corr       # OT plan vs GT correspondence from T_geo (Stage 1)
  + λ2 · L_deg        # margin rank: quality(strong-deg) < quality(mild-deg)
```
No antisymmetry loss: `s(A,B) = ψ(A,B) − ψ(B,A) + …` is antisymmetric **by
construction** (Step 3); penalizing it would be redundant and reads as not
trusting the architecture.

Freezing `Enc_struct` (DINO) gives the quality-invariance of the matching stream
for free, so no explicit invariance loss is needed; only the quality stream +
OT-coupling + head train.

## Why it should beat the baselines (per regime)

- **vs align-then-FR**: no homography, no warping → survives **R3 (FoV/crop)**
  and **R4 (parallax + cross-ISP)** where global registration fails, and never
  pays the resampling penalty (central thesis above).
- **vs ST-LPIPS**: shift tolerance covers small *global* translations only, not
  parallax, FoV/crop, or color-science gaps.
- **vs IQT / AHIQ (cross-attention FR)**: softmax cross-attention with one
  backbone for both addressing and values — addressing contaminated by
  rendering differences, and one-sided matching breaks in non-overlap regions
  (Step 1's two substitutions fix exactly these).
- **vs DISTS / DOSS / CKDN**: two-stream content/quality disentanglement + learned
  OT correspondences + **comparative training on real device data**, rather than a
  fixed feature-statistics comparison built for synthetic reconstruction pairs.
- **vs NR (incl. this repo's SigLIP2+AGM)**: reference-aware and comparative —
  answers "which is better vs. the scene", not just "how good in isolation".
- **vs NOVA**: trained for **real** geometric+photometric device misalignment, not
  synthetic NVS distortions.

## Compute

Sinkhorn over `N×N` tokens (≈1024 at SigLIP2 512/patch16) is a few cheap
iterations on GPU; sliced-Wasserstein is `O(N log N · #slices)`. Token
downsampling (e.g., stride-pool to 256) is the lever if memory is tight.

## Ablations (build the table to isolate each claim)

| Variant | Tests |
|---|---|
| − OT (mean-pool match) | value of learned correspondences (esp. R2–R3) |
| softmax normalization (IQT/AHIQ-style attention) | necessity of OT over one-sided matching — preempts "why the OT machinery?" |
| balanced (uniform-marginal) OT | necessity of **partial/unbalanced** OT under FoV/crop (R3) |
| same-stream Q/K (logits from quality stream) | value of structure-addressed attention vs. one-backbone attention |
| − `d_set` | value of the alignment-free distribution branch (R4) |
| − two-stream (single encoder for both) | value of content/quality disentanglement |
| − Stage-1 synthetic pretrain | value of free correspondence/degradation supervision |
| − synthetic color-science gap (Stage 1) | whether R4 robustness needs photometric pretraining |
| − native-res low-level features | whether downsampled SigLIP2 alone erases the quality cues |
| structural stream: DINO vs SigLIP vs early-layer | which gives quality-invariant matching |
| hard-align (homography/RAFT) + FR head | the "just register first" upper-baseline |

The headline ablation figure: **SRCC vs. measured residual displacement (R0→R4)**
for CALM vs. each baseline — CALM stays flat where the others fall off.

---

## Risks & mitigations

| Risk | Detection / mitigation |
|---|---|
| Humans can't rate R4 reliably (taste-dominated) | pilot gate thresholds (study §6); report ceiling-normalized numbers; fidelity framing as the fallback scale |
| DINO matching fails on night / low-texture scenes — exactly where ISPs differ most | pilot check on night scenes; weak positional prior; fall back to `d_set`-dominated scoring when dropped-mass is high |
| Downsampling erases low-level quality cues | native-res patch sampling / low-level conv features (see *Native-resolution caveat*); ablate |
| Real labeled data too small to fine-tune a learned metric | Stage-1 pretrain incl. synthetic color-science gap; LODO split verifies no device memorization |
| Reference not actually preferred (circularity) | RAW + neutral rendering; pilot validation; demote to content anchor if it fails |
| Resampling-penalty demo half-fails (DISTS tolerates resampling) | report per metric; claim scoped to warp-based *pipelines* |

## Ethics & release

- **Human study**: IRB / institutional ethics approval before the pilot;
  informed consent; observer data anonymized. Top venues require this in the
  checklist (NeurIPS) or expect it in the paper (CVPR).
- **In-the-wild captures**: consent for identifiable people or blur faces /
  exclude; avoid license plates and private interiors. State the policy in the
  release.
- **License**: pick the dataset license before capture (e.g., CC BY-NC-SA for
  images + permissive license for metadata/code); device names are factual and
  fine to publish, but avoid ranking *named* devices in marketing-quotable form
  if legal review objects — regime-level results don't need it.

## Open decisions (resolve on paper before capture)

1. **FR vs. comparative-NR framing** — do **both** (reference-based FR *and*
   pairwise). The chosen capture protocol must include the reference shot.
2. **Quality vs. preference** — collect under explicit fidelity *and* preference
   instructions; report the gap.
3. **Naming** — avoid "NAR-IQA"; pick a real-device / ISP-centric name.
4. **Method (the top-tier lift)** — a baseline alone reads as D&B/journal. Aim for
   a misalignment-robust **comparative** metric (dense-correspondence / attention
   / optimal-transport matching of *distributions of local quality* rather than
   aligned pixels) that **beats DOSS / DISTS / CKDN / NOVA / ST-LPIPS /
   align-then-FR** on R3–R4 *and survives leave-one-device-out*.
5. **Venue (let the pilot decide)** — strong method margin on R3–R4 + LODO
   survival → CVPR/ICCV. Solid dataset but thin method margin → NeurIPS
   Datasets & Benchmarks or TIP; forcing the thin version into CVPR earns the
   "new dataset + assembled method" rejection.

## References (verify before citing)

- ITU-R BT.500-14; ITU-T P.910 — subjective assessment methodology.
- Thurstone (1927) Case V; Bradley–Terry–Luce; Perez-Ortiz & Mantiuk, *pwcmp*
  (pairwise → JOD scaling).
- Crowd-BT (active pairwise sampling); TrueSkill; HodgeRank.
- DISTS; DOSS (order-statistics FR); CKDN (degraded-reference); NOVA (NAR-IQA,
  arXiv 2511.08155).
- ST-LPIPS — Ghildyal & Liu, *Shift-Tolerant Perceptual Similarity Metric*,
  ECCV 2022 (arXiv 2207.13686).
- SCPQD2020 (arXiv 2003.01299); SPCD (arXiv 2205.13489).
- RealSR (real-world SR; registered-reference dataset construction) — context
  for how prior work sidesteps misaligned references.
- IQT — Cheon et al., *Perceptual Image Quality Assessment with Transformers*
  (CVPRW 2021); AHIQ — prior cross-attention FR-IQA to cite against Step 1.
- Sinkformer — Sander et al. (AISTATS 2022), Sinkhorn-normalized attention;
  SuperGlue (dustbin for unmatched tokens).
- BAPPS (LPIPS 2AFC dataset) and the PieAPP dataset — precedent for pairwise
  perceptual ground truth and 2AFC-accuracy evaluation.
- LPIPS, DeepDC, PieAPP, AHIQ; NIQE, BRISQUE, MUSIQ, MANIQA, CLIP-IQA, Q-Align;
  Compare2Score (arXiv 2405.19298).
