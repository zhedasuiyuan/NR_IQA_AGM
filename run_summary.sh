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
#   sum_ap      summary, ap (== ALF; sanity vs alf_all)
#   sum_pma{1,2,4}  summary, learned k-query pool (L*k tokens)   -- our arm, learned
#   sum_tome{2,4}   summary, parameter-free merge (L*r tokens)   -- our arm, free
#
# Cross-layer K/V size: xattn = L*N (~27*1024); summary/alf = L*width(+CLS) (~27*k).
# That ~N/width reduction is the efficiency claim.
#
# Results (test SRCC/PLCC + wall-clock) -> summary_out/summary_results.csv.
#
# Usage:
#   ./run_summary.sh                          # KADID10K + KonIQ_10K, 3 GPUs
#   EPOCHS=8 GPUS="0 1" ./run_summary.sh
#   DATASETS="KADID10K" ./run_summary.sh

set -uo pipefail
cd "$(dirname "$0")"

DATASETS="${DATASETS:-KADID10K KonIQ_10K}"
GPUS="${GPUS:-0 1 2}"
EPOCHS="${EPOCHS:-15}"
PEFT="${PEFT:-LoRA}"          # same backbone adaptation for every run -> isolates the aggregator

read -ra DS_ARR  <<< "$DATASETS"
read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
mkdir -p summary_out
rm -f summary_out/row_*.csv

# ---- aggregator configs (all over ALL layers via --fusion_stride 1) --------
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

JOBS=()
for ds in "${DS_ARR[@]}"; do for c in "${CONFIGS[@]}"; do JOBS+=("$ds|$c"); done; done
echo "Jobs: ${#JOBS[@]}  |  GPUs: $GPUS  |  epochs=$EPOCHS peft=$PEFT"
echo

dispatch() {
  local slot=$1 gpu=${GPU_ARR[$slot]} idx=0 job
  for job in "${JOBS[@]}"; do
    if (( idx % NGPU == slot )); then
      IFS='|' read -r ds label flags <<< "$job"
      local stage="summary_${ds}_${label}"
      local log="summary_out/${stage}.log"
      echo "[GPU $gpu] START $ds $label -> $log"
      local start=$SECONDS
      CUDA_VISIBLE_DEVICES="$gpu" python train.py \
            --dataset "$ds" --peft_method "$PEFT" --epochs "$EPOCHS" \
            --stage_name "$stage" $flags > "$log" 2>&1
      local rc=$? dur=$((SECONDS - start))
      if [ $rc -eq 0 ]; then echo "[GPU $gpu] DONE  $ds $label (${dur}s)"
      else echo "[GPU $gpu] FAIL rc=$rc $ds $label (see $log)"; fi
      local res="results/results_${stage}_Train_${ds}_Test_${ds}.json"
      python - "$res" "$ds" "$label" "$dur" > "summary_out/row_${stage}.csv" <<'PY'
import json, sys
res, ds, label, dur = sys.argv[1:5]
try:
    d = json.load(open(res))
    print(f"{ds},{label},{d['val_SRCC']:.4f},{d['test_SRCC']:.4f},{d['test_PLCC']:.4f},{dur}")
except Exception:
    print(f"{ds},{label},NA,NA,NA,{dur}")
PY
    fi
    idx=$((idx + 1))
  done
}
for slot in "${!GPU_ARR[@]}"; do dispatch "$slot" & done
wait

# ---- assemble + report -----------------------------------------------------
out="summary_out/summary_results.csv"
echo "dataset,aggregator,val_SRCC,test_SRCC,test_PLCC,wall_s" > "$out"
cat summary_out/row_*.csv 2>/dev/null | sort >> "$out"
echo
echo "=== summary-fusion LoRA comparison ($out) ==="
column -s, -t "$out" 2>/dev/null || cat "$out"
echo
python - "$out" <<'PY'
import csv, sys
from collections import defaultdict
by = defaultdict(dict)  # ds -> {label: (srcc, wall)}
for r in csv.DictReader(open(sys.argv[1])):
    try: by[r["dataset"]][r["aggregator"]] = (float(r["test_SRCC"]), float(r["wall_s"]))
    except ValueError: pass
for ds, d in by.items():
    xa = d.get("xattn_all")
    print(f"\n{ds}:  cross_attention = " + (f"SRCC {xa[0]:.4f}, {xa[1]:.0f}s" if xa else "NA"))
    for label in sorted(d):
        if label == "xattn_all": continue
        s, w = d[label]
        if xa:
            ds_delta, spd = s - xa[0], xa[1] / w if w else float("nan")
            print(f"  {label:10s} SRCC={s:.4f} ({ds_delta:+.4f} vs xattn)  {w:.0f}s ({spd:.1f}x faster)")
        else:
            print(f"  {label:10s} SRCC={s:.4f}  {w:.0f}s")
    print("  => an arm at ~xattn SRCC but multiple-x faster is the efficiency win.")
PY
