#!/usr/bin/env bash
#
# Aggregation pilot: does complementarity-selected / distortion-conditioned
# fusion beat ALF-style all-layer cross-attention on IQA? Runs A-G per dataset
# (below), parallelised across GPUs, and collects test SRCC/PLCC into one table.
#
#   A_none         vanilla last-layer (get_image_features)      -- floor
#   B_allattn      cross-attention over ALL layers              -- the ALF analog to beat
#   C_selattn      cross-attention over the SELECTED layers     -- selection vs all (efficiency)
#   D_adaptstatic  adaptive static weights, selected layers
#   E_adaptimage   adaptive IMAGE-conditioned, selected layers  -- the leg ALF lacks
#   F_mls          MLS (RAEv2) over selected layers             -- baseline
#   G_topkattn     cross-attention over TOP-k-by-SRCC layers    -- complementarity vs marginal
#
# Selected sets come from each dataset's newest probe_out/<ds>_*/ complementarity
# run (via select_layers.py). Every run shares one training budget; only the
# fusion config varies. Results -> pilot_out/pilot_results.csv.
#
# Prereq: a complementarity run per dataset. This script calls select_layers.py
# itself, but to *preview* the selected sets first, run:
#   python probe_layers.py --dataset KADID10K --complementarity   # if not done yet
#   python select_layers.py probe_out/KADID10K_*/ --k 5           # prints greedy/top_k sets
#   python select_layers.py probe_out/KonIQ_10K_*/ --k 5
#
# Usage:
#   ./run_pilot.sh                         # KADID10K + KonIQ_10K, 3 GPUs
#   EPOCHS=8 GPUS="0 1" ./run_pilot.sh     # faster / fewer GPUs

set -uo pipefail
cd "$(dirname "$0")"

DATASETS="${DATASETS:-KADID10K KonIQ_10K}"
GPUS="${GPUS:-0 1 2}"
EPOCHS="${EPOCHS:-15}"
PEFT="${PEFT:-LoRA}"       # same backbone-adaptation for every run -> isolates fusion
# NOTE: train.py has no --seed flag; the split seed is fixed in seed.py, so all
# runs share one deterministic split (fine -- only the fusion config varies).
K="${K:-5}"               # target selected-set size

read -ra DS_ARR  <<< "$DATASETS"
read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
mkdir -p pilot_out
rm -f pilot_out/row_*.csv   # clear stale rows from a previous invocation

latest_comp() {  # newest probe_out/<ds>_*/ dir that has a complementarity CSV
  local d
  for d in $(ls -dt "probe_out/${1}_"*/ 2>/dev/null); do
    if ls "$d"/*_complementarity.csv >/dev/null 2>&1; then echo "$d"; return; fi
  done
}

emit_set() {  # $1 = run dir, $2 = key in selected_layers.json -> comma list
  python -c "import json;print(','.join(map(str,json.load(open('$1/selected_layers.json'))['$2'])))"
}

# ---- build job list: "DS|LABEL|NLAYERS|FLAGS" -----------------------------
JOBS=()
for ds in "${DS_ARR[@]}"; do
  dir="$(latest_comp "$ds")"
  if [ -z "$dir" ]; then
    echo "!! no complementarity run for $ds; run: python probe_layers.py --dataset $ds --complementarity"
    continue
  fi
  python select_layers.py "$dir" --k "$K" >/dev/null
  sel="$(emit_set "$dir" greedy_complementary)"
  topk="$(emit_set "$dir" top_k_srcc)"
  nsel=$(( $(tr -cd ',' <<<"$sel"  | wc -c) + 1 ))
  ntop=$(( $(tr -cd ',' <<<"$topk" | wc -c) + 1 ))
  echo "[$ds] selected(greedy)=$sel   top_k=$topk"
  JOBS+=("$ds|A_none|1|--fusion_type none")
  JOBS+=("$ds|B_allattn|all|--fusion_type cross_attention --fusion_stride 1")
  JOBS+=("$ds|C_selattn|$nsel|--fusion_type cross_attention --fusion_layers $sel")
  JOBS+=("$ds|D_adaptstatic|$nsel|--fusion_type adaptive --adaptive_conditioning static --fusion_layers $sel")
  JOBS+=("$ds|E_adaptimage|$nsel|--fusion_type adaptive --adaptive_conditioning image --fusion_layers $sel")
  JOBS+=("$ds|F_mls|$nsel|--fusion_type mls --fusion_layers $sel")
  JOBS+=("$ds|G_topkattn|$ntop|--fusion_type cross_attention --fusion_layers $topk")
done
[ ${#JOBS[@]} -eq 0 ] && { echo "No jobs to run."; exit 1; }

echo "Jobs: ${#JOBS[@]}  |  GPUs: $GPUS  |  epochs=$EPOCHS peft=$PEFT"
echo

# ---- one worker per GPU; round-robin ---------------------------------------
dispatch() {
  local slot=$1 gpu=${GPU_ARR[$slot]} idx=0 job
  for job in "${JOBS[@]}"; do
    if (( idx % NGPU == slot )); then
      IFS='|' read -r ds label nlayers flags <<< "$job"
      local stage="pilot_${ds}_${label}"
      local log="pilot_out/${stage}.log"
      echo "[GPU $gpu] START $ds $label -> $log"
      if CUDA_VISIBLE_DEVICES="$gpu" python train.py \
            --dataset "$ds" --peft_method "$PEFT" --epochs "$EPOCHS" \
            --stage_name "$stage" $flags > "$log" 2>&1; then
        echo "[GPU $gpu] DONE  $ds $label"
      else
        echo "[GPU $gpu] FAIL  $ds $label (see $log)"
      fi
      # within-dataset: train_db == test_db == ds
      local res="results/results_${stage}_Train_${ds}_Test_${ds}.json"
      python - "$res" "$ds" "$label" "$nlayers" > "pilot_out/row_${stage}.csv" <<'PY'
import json, sys
res, ds, label, nl = sys.argv[1:5]
try:
    d = json.load(open(res))
    print(f"{ds},{label},{nl},{d['val_SRCC']:.4f},{d['test_SRCC']:.4f},{d['test_PLCC']:.4f}")
except Exception:
    print(f"{ds},{label},{nl},NA,NA,NA")
PY
    fi
    idx=$((idx + 1))
  done
}

for slot in "${!GPU_ARR[@]}"; do dispatch "$slot" & done
wait

# ---- assemble the results table --------------------------------------------
out="pilot_out/pilot_results.csv"
echo "dataset,run,n_layers,val_SRCC,test_SRCC,test_PLCC" > "$out"
cat pilot_out/row_*.csv 2>/dev/null | sort >> "$out"
echo
echo "=== pilot results ($out) ==="
column -s, -t "$out" 2>/dev/null || cat "$out"
echo
echo "Reads: B_allattn is the ALF analog. C beats B with fewer layers => efficiency;"
echo "E beats B => a real method delta; C beats G => complementarity > marginal selection."
