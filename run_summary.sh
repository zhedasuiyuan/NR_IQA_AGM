#!/usr/bin/env bash
#
# LoRA fine-tuning comparison for the summary-fusion arm. Central question:
# can ALF / the widened per-layer summary MATCH spatial cross-attention
# aggregation at a fraction of its compute? Every run fine-tunes the same
# backbone (LoRA) over ALL layers (--fusion_stride 1); only the aggregator
# differs, and we log wall-clock next to SRCC so the accuracy/compute trade-off
# is visible.
#
#   none        vanilla last-layer (get_image_features)         -- floor
#   xattn_all   cross-attention, trunk queries L*N tokens       -- the expensive arm to beat
#   alf_all     ALF: attentive fusion of per-layer CLS+AP        -- cheap baseline
#   sum_ap      summary, ap (== ALF + in_norm/layer_emb; control)
#   sum_pma{1,2,4}  summary, learned k-query pool (L*k tokens)   -- our arm, learned
#   sum_tome{2,4}   summary, parameter-free merge (L*r tokens)   -- our arm, free
#
# Cross-layer K/V size: xattn = L*N (~27*1024); summary/alf = L*width(+CLS) (~27*k).
# That ~N/width reduction is the efficiency claim.
#
# CLS: alf/summary arms keep the CLS token per layer for CLS-bearing backbones
# (CLIP/DINO) so the whole family matches ALF's access; SigLIP2 has no CLS, so
# --alf_use_cls is (correctly) NOT set there -- it would mislabel patch 0.
# The flag handles a single leading CLS; DINOv2 *register* variants (>1 prefix
# token) are only approximately handled (registers counted as patches).
#
# Results (test SRCC/PLCC + wall-clock) -> summary_out/<timestamp>/summary_results.csv
# (per-run subfolder so reruns don't overwrite; RUN=<name> overrides the stamp).
#
# Usage:
#   ./run_summary.sh                          # SigLIP2, KADID10K + KonIQ_10K, 3 GPUs
#   MODELS="google/siglip2-so400m-patch16-512 openai/clip-vit-large-patch14 facebook/dinov2-large" \
#     ./run_summary.sh                        # cross-backbone (CLS auto-set per model)
#   EPOCHS=8 GPUS="0 1" DATASETS="KADID10K" ./run_summary.sh

set -uo pipefail
cd "$(dirname "$0")"

MODELS="${MODELS:-google/siglip2-so400m-patch16-512}"   # space-separated HF ids
DATASETS="${DATASETS:-KADID10K KonIQ_10K}"
GPUS="${GPUS:-0 1 2}"
EPOCHS="${EPOCHS:-15}"
PEFT="${PEFT:-LoRA}"          # same backbone adaptation for every run -> isolates the aggregator
SEEDS="${SEEDS:-8 19 25}"     # random-split seeds; report mean+/-std (8 == seed.py default)

read -ra MODEL_ARR <<< "$MODELS"
read -ra DS_ARR  <<< "$DATASETS"
read -ra SEED_ARR <<< "$SEEDS"
read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}

# Per-run dir (timestamp) so reruns don't overwrite each other's logs/results.
RUN="${RUN:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="summary_out/${RUN}"
mkdir -p "$RUN_DIR"
echo "Outputs -> $RUN_DIR"

# ---- aggregator configs (all over ALL layers via --fusion_stride 1) --------
# alf/summary arms get --alf_use_cls appended per-model (see cls_flag below).
CONFIGS=(
  "none|--fusion_type none"
  "xattn_all|--fusion_type cross_attention --fusion_stride 1"
  "alf_all|--fusion_type alf --fusion_stride 1"
  "sum_ap|--fusion_type summary --fusion_stride 1 --summarizer ap"
  "sum_pma1|--fusion_type summary --fusion_stride 1 --summarizer pma --summary_width 1"
  "sum_pma2|--fusion_type summary --fusion_stride 1 --summarizer pma --summary_width 2"
  "sum_pma4|--fusion_type summary --fusion_stride 1 --summarizer pma --summary_width 4"
  "sum_tome2|--fusion_type summary --fusion_stride 1 --summarizer tome --summary_width 2"
  "sum_tome4|--fusion_type summary --fusion_stride 1 --summarizer tome --summary_width 4"
)

# CLS-bearing backbone? (SigLIP2 has none.) Off for SigLIP, on otherwise.
cls_flag_for() { case "$1" in *[Ss]ig[Ll][Ii][Pp]*) echo "" ;; *) echo "--alf_use_cls" ;; esac; }

JOBS=()
for m in "${MODEL_ARR[@]}"; do
  for ds in "${DS_ARR[@]}"; do
    for sd in "${SEED_ARR[@]}"; do
      for c in "${CONFIGS[@]}"; do JOBS+=("$m|$ds|$sd|$c"); done
    done
  done
done
echo "Models: $MODELS  |  seeds: $SEEDS"
echo "Jobs: ${#JOBS[@]}  |  GPUs: $GPUS  |  epochs=$EPOCHS peft=$PEFT"
echo

dispatch() {
  local slot=$1 gpu=${GPU_ARR[$slot]} idx=0 job
  for job in "${JOBS[@]}"; do
    if (( idx % NGPU == slot )); then
      IFS='|' read -r m ds sd label flags <<< "$job"
      # CLS access for the alf/summary family on CLS-bearing backbones.
      case "$label" in alf*|sum*) flags="$flags $(cls_flag_for "$m")" ;; esac
      local mtag="${m##*/}"
      local stage="summary_${mtag}_${ds}_s${sd}_${label}"
      local log="$RUN_DIR/${stage}.log"
      echo "[GPU $gpu] START $mtag $ds s$sd $label -> $log"
      local start=$SECONDS
      CUDA_VISIBLE_DEVICES="$gpu" python train.py \
            --model_id "$m" --dataset "$ds" --seed "$sd" --peft_method "$PEFT" --epochs "$EPOCHS" \
            --stage_name "$stage" $flags > "$log" 2>&1
      local rc=$? dur=$((SECONDS - start))
      if [ $rc -eq 0 ]; then echo "[GPU $gpu] DONE  $mtag $ds s$sd $label (${dur}s)"
      else echo "[GPU $gpu] FAIL rc=$rc $mtag $ds s$sd $label (see $log)"; fi
      local res="results/results_${stage}_Train_${ds}_Test_${ds}.json"
      python - "$res" "$mtag" "$ds" "$sd" "$label" "$dur" > "$RUN_DIR/row_${stage}.csv" <<'PY'
import json, sys
res, m, ds, sd, label, dur = sys.argv[1:7]
try:
    d = json.load(open(res))
    print(f"{m},{ds},{sd},{label},{d['val_SRCC']:.4f},{d['test_SRCC']:.4f},{d['test_PLCC']:.4f},{dur}")
except Exception:
    print(f"{m},{ds},{sd},{label},NA,NA,NA,{dur}")
PY
    fi
    idx=$((idx + 1))
  done
}
for slot in "${!GPU_ARR[@]}"; do dispatch "$slot" & done
wait

# ---- assemble + report -----------------------------------------------------
out="$RUN_DIR/summary_results.csv"
echo "model,dataset,seed,aggregator,val_SRCC,test_SRCC,test_PLCC,wall_s" > "$out"
cat "$RUN_DIR"/row_*.csv 2>/dev/null | sort >> "$out"
echo
echo "=== summary-fusion LoRA comparison, per-seed rows ($out) ==="
column -s, -t "$out" 2>/dev/null || cat "$out"
echo
echo "=== mean +/- std over seeds ==="
python - "$out" <<'PY'
import csv, sys, statistics as st
from collections import defaultdict
agg = defaultdict(lambda: defaultdict(list))  # (model,ds) -> label -> [(srcc, wall), ...]
for r in csv.DictReader(open(sys.argv[1])):
    try: agg[(r["model"], r["dataset"])][r["aggregator"]].append((float(r["test_SRCC"]), float(r["wall_s"])))
    except ValueError: pass
def ms(xs):
    return (st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0))
for (m, ds), d in agg.items():
    xa = d.get("xattn_all")
    xam = ms([s for s, _ in xa])[0] if xa else None
    hdr = f"\n{m} / {ds}:"
    print(f"{hdr}  cross_attention SRCC={xam:.4f} (n={len(xa)})" if xa else hdr)
    for label in sorted(d):
        rows = d[label]
        smean, sstd = ms([s for s, _ in rows])
        wmean = ms([w for _, w in rows])[0]
        tag = "  <-- xattn" if label == "xattn_all" else ""
        extra = ""
        if xam and label != "xattn_all":
            spd = (ms([w for _, w in xa])[0] / wmean) if wmean else float("nan")
            extra = f"  ({smean - xam:+.4f} vs xattn, {spd:.1f}x faster)"
        print(f"  {label:10s} SRCC={smean:.4f}+/-{sstd:.4f}  {wmean:.0f}s  (n={len(rows)}){extra}{tag}")
    print("  => an arm at ~xattn SRCC (within std) but multiple-x faster = the efficiency win.")
PY
