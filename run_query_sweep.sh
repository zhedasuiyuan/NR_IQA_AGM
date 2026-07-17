#!/usr/bin/env bash
#
# Query-layer sweep for cross-attention fusion. Tests the probing-motivated
# hypothesis: the best attention QUERY is a quality-rich intermediate layer, not
# the semantically-invariant final-layer trunk.
#
# K/V = all layers EXCEPT the query layer (fusion_stride 1, query layer excluded)
# -- so this does NOT depend on the (still-unvalidated) layer selection; only the
# query layer varies, and each run is symmetric (query from q, K/V = the rest).
#
# Comparison points:
#   * per query layer q: cross-attention, query = layer q, K/V = all others  (sweep)
#   * q = the last layer (e.g. 27): the FINAL-LAYER baseline, done consistently
#     with exclusion -- this is the baseline, NOT run_pilot's B_allattn (which
#     keeps the last block in its K/V and so isn't comparable once we exclude).
#   * ALF, all layers, learned query (one run per dataset).
# So: does an intermediate query beat the final-layer query and ALF's learned query?
#
# Results -> query_out/query_results.csv.
#
# Usage:
#   ./run_query_sweep.sh                                  # KADID10K + KonIQ_10K, 3 GPUs
#   QUERY_LAYERS="6,12,18" EPOCHS=8 ./run_query_sweep.sh  # fewer query points / faster

set -uo pipefail
cd "$(dirname "$0")"

DATASETS="${DATASETS:-KADID10K KonIQ_10K}"
GPUS="${GPUS:-0 1 2}"
EPOCHS="${EPOCHS:-15}"
PEFT="${PEFT:-LoRA}"
# Query layers to try (hidden_states indices). Default is a coarse grid for the
# 27-layer SigLIP2-so400m; adjust for other backbone depths.
QUERY_LAYERS="${QUERY_LAYERS:-3,6,9,12,15,18,21,24,27}"

read -ra DS_ARR  <<< "$DATASETS"
read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
mkdir -p query_out
rm -f query_out/row_*.csv

# ---- build jobs: "DS|QLAYER" (K/V is always all layers) --------------------
# q is a layer index for the cross-attention query sweep, or "alf" for the
# ALF-all-layers reference (learned query, no query layer).
JOBS=()
for ds in "${DS_ARR[@]}"; do
  for q in ${QUERY_LAYERS//,/ }; do
    JOBS+=("$ds|$q")
  done
  JOBS+=("$ds|alf")
done
[ ${#JOBS[@]} -eq 0 ] && { echo "No jobs to run."; exit 1; }

echo "Datasets: $DATASETS | query layers: $QUERY_LAYERS | K/V: all layers"
echo "Jobs: ${#JOBS[@]}  |  GPUs: $GPUS  |  epochs=$EPOCHS peft=$PEFT"
echo

dispatch() {
  local slot=$1 gpu=${GPU_ARR[$slot]} idx=0 job
  for job in "${JOBS[@]}"; do
    if (( idx % NGPU == slot )); then
      IFS='|' read -r ds q <<< "$job"
      local stage="qsweep_${ds}_q${q}"
      local log="query_out/${stage}.log"
      # ALF reference (learned query) vs cross-attention query-from-layer-q.
      # Both use K/V = all layers via --fusion_stride 1.
      local fusion_flags="--fusion_type cross_attention --fusion_stride 1 --fusion_query_layer $q"
      [ "$q" = "alf" ] && fusion_flags="--fusion_type alf --fusion_stride 1"
      echo "[GPU $gpu] START $ds query=$q -> $log"
      if CUDA_VISIBLE_DEVICES="$gpu" python train.py \
            --dataset "$ds" --peft_method "$PEFT" --epochs "$EPOCHS" --stage_name "$stage" \
            $fusion_flags > "$log" 2>&1; then
        echo "[GPU $gpu] DONE  $ds query=$q"
      else
        echo "[GPU $gpu] FAIL  $ds query=$q (see $log)"
      fi
      local res="results/results_${stage}_Train_${ds}_Test_${ds}.json"
      python - "$res" "$ds" "$q" > "query_out/row_${stage}.csv" <<'PY'
import json, sys
res, ds, q = sys.argv[1:4]
try:
    d = json.load(open(res))
    print(f"{ds},{q},{d['val_SRCC']:.4f},{d['test_SRCC']:.4f},{d['test_PLCC']:.4f}")
except Exception:
    print(f"{ds},{q},NA,NA,NA")
PY
    fi
    idx=$((idx + 1))
  done
}

for slot in "${!GPU_ARR[@]}"; do dispatch "$slot" & done
wait

# ---- assemble table + report the best query layer per dataset --------------
out="query_out/query_results.csv"
echo "dataset,query_layer,val_SRCC,test_SRCC,test_PLCC" > "$out"
cat query_out/row_*.csv 2>/dev/null | sort -t, -k1,1 -k2,2n >> "$out"
echo
echo "=== query-layer sweep, K/V = all layers ($out) ==="
column -s, -t "$out" 2>/dev/null || cat "$out"
echo
python - "$out" <<'PY'
import csv, sys
from collections import defaultdict
pts = defaultdict(list)   # dataset -> [(query_layer_int, srcc)]
alf = {}                  # dataset -> ALF srcc
for r in csv.DictReader(open(sys.argv[1])):
    try:
        s = float(r["test_SRCC"])
    except ValueError:
        continue
    ds, q = r["dataset"], r["query_layer"]
    if q == "alf":
        alf[ds] = s
    else:
        pts[ds].append((int(q), s))
for ds in sorted(pts):
    last_q, last_s = max(pts[ds], key=lambda t: t[0])   # final-layer baseline (largest q)
    best_q, best_s = max(pts[ds], key=lambda t: t[1])   # best query layer
    a = alf.get(ds)
    a_str = f"ALF={a:.4f}" if a is not None else "ALF=NA"
    vs_last = "beats" if best_s > last_s else "<="
    vs_alf = "beats" if (a is None or best_s > a) else "<="
    print(f"  {ds}: best query L{best_q}={best_s:.4f}  |  final-layer L{last_q}={last_s:.4f} "
          f"({vs_last})  |  {a_str} ({vs_alf})")
    if best_q != last_q and best_s > last_s:
        print("       => an intermediate query beats the final-layer query: hypothesis holds")
PY
