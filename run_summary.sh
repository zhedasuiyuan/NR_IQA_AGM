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
# Results (test SRCC/PLCC + wall-clock) -> summary_out/summary_results.csv.
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

read -ra MODEL_ARR <<< "$MODELS"
read -ra DS_ARR  <<< "$DATASETS"
read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
mkdir -p summary_out
rm -f summary_out/row_*.csv

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
  for ds in "${DS_ARR[@]}"; do for c in "${CONFIGS[@]}"; do JOBS+=("$m|$ds|$c"); done; done
done
echo "Models: $MODELS"
echo "Jobs: ${#JOBS[@]}  |  GPUs: $GPUS  |  epochs=$EPOCHS peft=$PEFT"
echo

dispatch() {
  local slot=$1 gpu=${GPU_ARR[$slot]} idx=0 job
  for job in "${JOBS[@]}"; do
    if (( idx % NGPU == slot )); then
      IFS='|' read -r m ds label flags <<< "$job"
      # CLS access for the alf/summary family on CLS-bearing backbones.
      case "$label" in alf*|sum*) flags="$flags $(cls_flag_for "$m")" ;; esac
      local mtag="${m##*/}"
      local stage="summary_${mtag}_${ds}_${label}"
      local log="summary_out/${stage}.log"
      echo "[GPU $gpu] START $mtag $ds $label -> $log"
      local start=$SECONDS
      CUDA_VISIBLE_DEVICES="$gpu" python train.py \
            --model_id "$m" --dataset "$ds" --peft_method "$PEFT" --epochs "$EPOCHS" \
            --stage_name "$stage" $flags > "$log" 2>&1
      local rc=$? dur=$((SECONDS - start))
      if [ $rc -eq 0 ]; then echo "[GPU $gpu] DONE  $mtag $ds $label (${dur}s)"
      else echo "[GPU $gpu] FAIL rc=$rc $mtag $ds $label (see $log)"; fi
      local res="results/results_${stage}_Train_${ds}_Test_${ds}.json"
      python - "$res" "$mtag" "$ds" "$label" "$dur" > "summary_out/row_${stage}.csv" <<'PY'
import json, sys
res, m, ds, label, dur = sys.argv[1:6]
try:
    d = json.load(open(res))
    print(f"{m},{ds},{label},{d['val_SRCC']:.4f},{d['test_SRCC']:.4f},{d['test_PLCC']:.4f},{dur}")
except Exception:
    print(f"{m},{ds},{label},NA,NA,NA,{dur}")
PY
    fi
    idx=$((idx + 1))
  done
}
for slot in "${!GPU_ARR[@]}"; do dispatch "$slot" & done
wait

# ---- assemble + report -----------------------------------------------------
out="summary_out/summary_results.csv"
echo "model,dataset,aggregator,val_SRCC,test_SRCC,test_PLCC,wall_s" > "$out"
cat summary_out/row_*.csv 2>/dev/null | sort >> "$out"
echo
echo "=== summary-fusion LoRA comparison ($out) ==="
column -s, -t "$out" 2>/dev/null || cat "$out"
echo
python - "$out" <<'PY'
import csv, sys
from collections import defaultdict
by = defaultdict(dict)  # (model,ds) -> {label: (srcc, wall)}
for r in csv.DictReader(open(sys.argv[1])):
    try: by[(r["model"], r["dataset"])][r["aggregator"]] = (float(r["test_SRCC"]), float(r["wall_s"]))
    except ValueError: pass
for (m, ds), d in by.items():
    xa = d.get("xattn_all")
    print(f"\n{m} / {ds}:  cross_attention = " + (f"SRCC {xa[0]:.4f}, {xa[1]:.0f}s" if xa else "NA"))
    for label in sorted(d):
        if label == "xattn_all": continue
        s, w = d[label]
        if xa:
            dlt, spd = s - xa[0], xa[1] / w if w else float("nan")
            print(f"  {label:10s} SRCC={s:.4f} ({dlt:+.4f} vs xattn)  {w:.0f}s ({spd:.1f}x faster)")
        else:
            print(f"  {label:10s} SRCC={s:.4f}  {w:.0f}s")
    print("  => an arm at ~xattn SRCC but multiple-x faster is the efficiency win.")
PY
