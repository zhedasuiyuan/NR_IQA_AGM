#!/usr/bin/env bash
#
# Query-layer sweep for cross-attention fusion. Tests the probing-motivated
# hypothesis: the best attention QUERY is a quality-rich intermediate layer, not
# the semantically-invariant final-layer trunk.
#
# Fixes the K/V set to the selected complementary layers (same as run_pilot.sh's
# C_selattn) and varies --fusion_query_layer over each of those layers. The
# trunk-query baseline is NOT run here -- it's run_pilot.sh's C_selattn (same
# K/V, default trunk query); compare against that.
#
# Selected sets come from each dataset's newest probe_out/<ds>_*/ complementarity
# run (via select_layers.py). Results -> query_out/query_results.csv.
#
# Usage:
#   ./run_query_sweep.sh                     # KADID10K + KonIQ_10K, 3 GPUs
#   EPOCHS=8 GPUS="0 1" ./run_query_sweep.sh

set -uo pipefail
cd "$(dirname "$0")"

DATASETS="${DATASETS:-KADID10K KonIQ_10K}"
GPUS="${GPUS:-0 1 2}"
EPOCHS="${EPOCHS:-15}"
PEFT="${PEFT:-LoRA}"
K="${K:-5}"           # selected-set size (must match run_pilot.sh for a fair baseline)

read -ra DS_ARR  <<< "$DATASETS"
read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
mkdir -p query_out
rm -f query_out/row_*.csv

latest_comp() {  # newest probe_out/<ds>_*/ dir that has a complementarity CSV
  local d
  for d in $(ls -dt "probe_out/${1}_"*/ 2>/dev/null); do
    if ls "$d"/*_complementarity.csv >/dev/null 2>&1; then echo "$d"; return; fi
  done
}
emit_set() { python -c "import json;print(','.join(map(str,json.load(open('$1/selected_layers.json'))['$2'])))"; }

# ---- build jobs: "DS|QLAYER|KVSET" (one per query layer in the selected set) --
JOBS=()
for ds in "${DS_ARR[@]}"; do
  dir="$(latest_comp "$ds")"
  if [ -z "$dir" ]; then
    echo "!! no complementarity run for $ds; run: python probe_layers.py --dataset $ds --complementarity"
    continue
  fi
  python select_layers.py "$dir" --k "$K" >/dev/null
  sel="$(emit_set "$dir" greedy_complementary)"     # K/V set (fixed)
  echo "[$ds] K/V = selected = $sel   (query sweeps over these)"
  for q in ${sel//,/ }; do
    JOBS+=("$ds|$q|$sel")
  done
done
[ ${#JOBS[@]} -eq 0 ] && { echo "No jobs to run."; exit 1; }

echo "Jobs: ${#JOBS[@]}  |  GPUs: $GPUS  |  epochs=$EPOCHS peft=$PEFT"
echo

dispatch() {
  local slot=$1 gpu=${GPU_ARR[$slot]} idx=0 job
  for job in "${JOBS[@]}"; do
    if (( idx % NGPU == slot )); then
      IFS='|' read -r ds q sel <<< "$job"
      local nkv; nkv=$(( $(tr -cd ',' <<<"$sel" | wc -c) + 1 ))
      local stage="qsweep_${ds}_q${q}"
      local log="query_out/${stage}.log"
      echo "[GPU $gpu] START $ds query=$q -> $log"
      if CUDA_VISIBLE_DEVICES="$gpu" python train.py \
            --dataset "$ds" --peft_method "$PEFT" --epochs "$EPOCHS" --stage_name "$stage" \
            --fusion_type cross_attention --fusion_layers "$sel" --fusion_query_layer "$q" \
            > "$log" 2>&1; then
        echo "[GPU $gpu] DONE  $ds query=$q"
      else
        echo "[GPU $gpu] FAIL  $ds query=$q (see $log)"
      fi
      local res="results/results_${stage}_Train_${ds}_Test_${ds}.json"
      python - "$res" "$ds" "$q" "$nkv" > "query_out/row_${stage}.csv" <<'PY'
import json, sys
res, ds, q, nkv = sys.argv[1:5]
try:
    d = json.load(open(res))
    print(f"{ds},{q},{nkv},{d['val_SRCC']:.4f},{d['test_SRCC']:.4f},{d['test_PLCC']:.4f}")
except Exception:
    print(f"{ds},{q},{nkv},NA,NA,NA")
PY
    fi
    idx=$((idx + 1))
  done
}

for slot in "${!GPU_ARR[@]}"; do dispatch "$slot" & done
wait

# ---- assemble table + report the best query layer per dataset --------------
out="query_out/query_results.csv"
echo "dataset,query_layer,kv_size,val_SRCC,test_SRCC,test_PLCC" > "$out"
cat query_out/row_*.csv 2>/dev/null | sort -t, -k1,1 -k2,2n >> "$out"
echo
echo "=== query-layer sweep ($out) ==="
column -s, -t "$out" 2>/dev/null || cat "$out"
echo
python - "$out" <<'PY'
import csv, sys
from collections import defaultdict
best = defaultdict(lambda: (-2.0, None))
for r in csv.DictReader(open(sys.argv[1])):
    try:
        s = float(r["test_SRCC"])
    except ValueError:
        continue
    if s > best[r["dataset"]][0]:
        best[r["dataset"]] = (s, r["query_layer"])
for ds, (s, q) in best.items():
    print(f"  best query layer for {ds}: L{q}  (test SRCC {s:.4f})  "
          f"-- intermediate => hypothesis holds; compare to run_pilot C_selattn (trunk query)")
PY
