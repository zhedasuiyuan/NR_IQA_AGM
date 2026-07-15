#!/usr/bin/env bash
#
# Layer-wise probing experiment sweep, parallelised across GPUs.
#
# Each job = one (dataset, seed) probe run with all three probes (mean-pool
# linear, native-pool, learned attention). Jobs are round-robin assigned to the
# GPUs in GPUS and each GPU runs its queue sequentially. Per-run outputs land in
# probe_out/<dataset>_<tag>_<timestamp>/ (with config.json); stdout/stderr go to
# probe_out/logs/.
#
# Usage:
#   ./run_probes.sh                 # Tier 1: 5 datasets x seed 42
#   SEEDS="42 123 7" ./run_probes.sh   # Tier 2: add seeds for robustness
#   DATASETS="KADID10K TID2013" GPUS="0 1" ./run_probes.sh   # subset
#
# Override any of these via env vars.
#
# ---- Recommended runs, in priority order -----------------------------------
# P1  Pretraining-objective comparison (the mechanism argument vs PE): probe
#     the SAME two datasets across contrastive (SigLIP2, CLIP) and non-contrastive
#     (DINOv2) encoders. 3 models x 2 datasets = 6 jobs across 3 GPUs.
#     VERIFY one CLIP/DINOv2 run first (see docs/layerwise_probing.md).
#
#   MODELS="google/siglip2-so400m-patch16-512 openai/clip-vit-large-patch14-336 facebook/dinov2-large" \
#     DATASETS="KADID10K KonIQ_10K" ./run_probes.sh
#
# P2  Main layer-wise curves: all 5 datasets on SigLIP2 (the script default).
#
#   ./run_probes.sh
#
# P3  Robustness: repeat P2 over multiple seeds.
#
#   SEEDS="42 123 7" ./run_probes.sh

set -uo pipefail
cd "$(dirname "$0")"

# ---- experiment grid -------------------------------------------------------
# Synthetic (per-distortion breakdown) + authentic (overall curve).
DATASETS="${DATASETS:-KADID10K TID2013 KonIQ_10K SPAQ CLIVE}"
SEEDS="${SEEDS:-42}"
GPUS="${GPUS:-0 1 2}"
# Backbones. Default = SigLIP2 only. For the pretraining-objective comparison
# add e.g. openai/clip-vit-large-patch14-336 facebook/dinov2-large (hidden size
# is auto-detected, so no other flags needed). Verify one CLIP/DINO run first.
MODELS="${MODELS:-google/siglip2-so400m-patch16-512}"
# Extra args passed to every run. --attention gives the PE-style probe on all
# layers; drop it (or add --attn_max_train 3000) if a full run is too slow.
# batch_size defaults to 32 (frozen backbone runs under no_grad); lower if OOM.
EXTRA="${EXTRA:---attention}"

read -ra DS_ARR  <<< "$DATASETS"
read -ra GPU_ARR <<< "$GPUS"
read -ra SEED_ARR <<< "$SEEDS"
read -ra MODEL_ARR <<< "$MODELS"
NGPU=${#GPU_ARR[@]}

# ---- build the job list (model | dataset | seed) ---------------------------
JOBS=()
for model in "${MODEL_ARR[@]}"; do
  for ds in "${DS_ARR[@]}"; do
    for seed in "${SEED_ARR[@]}"; do
      JOBS+=("$model|$ds|$seed")
    done
  done
done

mkdir -p probe_out/logs
echo "Models:   $MODELS"
echo "Datasets: $DATASETS"
echo "Seeds:    $SEEDS"
echo "GPUs:     $GPUS   |   ${#JOBS[@]} jobs total, extra args: $EXTRA"
echo

# ---- one worker per GPU; pull every Nth job (round-robin) ------------------
dispatch() {
  local slot=$1                      # 0..NGPU-1
  local gpu=${GPU_ARR[$slot]}
  local idx=0
  for job in "${JOBS[@]}"; do
    if (( idx % NGPU == slot )); then
      IFS='|' read -r model ds seed <<< "$job"
      local mshort="${model##*/}"          # short backbone name for the tag
      local tag="${mshort}_seed${seed}"
      local log="probe_out/logs/${ds}_${tag}_gpu${gpu}.log"
      echo "[GPU $gpu] START $mshort $ds seed=$seed -> $log"
      if CUDA_VISIBLE_DEVICES="$gpu" python probe_layers.py \
            --model_id "$model" --dataset "$ds" --seed "$seed" --tag "$tag" $EXTRA > "$log" 2>&1; then
        echo "[GPU $gpu] DONE  $mshort $ds seed=$seed"
      else
        echo "[GPU $gpu] FAIL  $mshort $ds seed=$seed (see $log)"
      fi
    fi
    idx=$((idx + 1))
  done
}

for slot in "${!GPU_ARR[@]}"; do
  dispatch "$slot" &
done
wait
echo
echo "All jobs finished. Results under probe_out/, logs under probe_out/logs/."
