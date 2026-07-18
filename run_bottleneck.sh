#!/usr/bin/env bash
#
# Per-layer summarization-bottleneck sweep (bottleneck_probe.py). Tests whether
# widening ALF's per-layer summary recovers IQA-relevant local-distortion detail.
# CLS rides along in every arm (keep_cls, default on); only the PATCH-summary
# WIDTH differs. All arms share one cross-layer PMA + linear head:
#
#   ap              -- CLS + mean patch token   (== ALF baseline; the arm to beat)
#   pma  k=1,2,4,8  -- CLS + k learned tokens   (task-driven patch width)
#   tome r=2,4,8    -- CLS + r merged tokens    (parameter-free, content-adaptive)
#
# pma k=1 vs ap = "learned vs mean single patch token" (ALF motivation, per layer).
# k>1 / r>1 vs ap = "does WIDTH matter". pma vs tome at matched width = "learned vs
# parameter-free allocation". If ap ties the widest arm everywhere -> the per-layer
# bottleneck is not real for IQA (clean negative); if width helps on KADID (local
# synthetic distortions) but not KonIQ -> the finding is locality-gated.
#
# The frozen backbone forwards ONCE per dataset (cache keyed by dataset+seed and
# REUSED across configs). Phase 1 builds caches serially; phase 2 sweeps configs
# round-robin across GPUs off the shared read-only cache.
#
# Usage:
#   ./run_bottleneck.sh                         # KADID10K + KonIQ_10K, 3 GPUs
#   SEEDS=3 EPOCHS=25 ./run_bottleneck.sh       # smoother curves
#   DATASETS="KADID10K" GPUS="0" ./run_bottleneck.sh

set -uo pipefail
cd "$(dirname "$0")"

DATASETS="${DATASETS:-KADID10K KonIQ_10K}"
GPUS="${GPUS:-0 1 2}"
EPOCHS="${EPOCHS:-20}"
SEEDS="${SEEDS:-3}"                       # avg out head-init variance (cache is free)
BS="${BS:-16}"                           # per-step batch (reads [B,L,N,D] into GPU)
CACHE_DIR="${CACHE_DIR:-/data/bneck_cache}"
KEEP_CACHE="${KEEP_CACHE:-0}"             # 1 = keep the (large) token cache after sweep
OUT="${OUT:-bneck_out/bneck_results.csv}"

read -ra DS_ARR  <<< "$DATASETS"
read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
mkdir -p bneck_out
rm -f "$OUT"                              # fresh table; per-run rows are appended

# ---- summarizer configs: "summarizer|width" -------------------------------
CONFIGS=("ap|1" "pma|1" "pma|2" "pma|4" "pma|8" "tome|2" "tome|4" "tome|8")

# ---- phase 1: build one cache per dataset (serial; first GPU) --------------
echo "=== phase 1: build caches (serial) ==="
for ds in "${DS_ARR[@]}"; do
  echo "[cache] $ds -> $CACHE_DIR"
  CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python bottleneck_probe.py \
    --dataset "$ds" --cache_dir "$CACHE_DIR" --batch_size "$BS" --build_cache_only \
    > "bneck_out/cache_${ds}.log" 2>&1 || { echo "  FAIL (see bneck_out/cache_${ds}.log)"; exit 1; }
done

# ---- phase 2: sweep configs round-robin across GPUs ------------------------
echo "=== phase 2: sweep ${#CONFIGS[@]} configs x ${#DS_ARR[@]} datasets ==="
JOBS=()
for ds in "${DS_ARR[@]}"; do for c in "${CONFIGS[@]}"; do JOBS+=("$ds|$c"); done; done

dispatch() {
  local slot=$1 gpu=${GPU_ARR[$slot]} idx=0 job
  for job in "${JOBS[@]}"; do
    if (( idx % NGPU == slot )); then
      IFS='|' read -r ds summ width <<< "$job"
      local log="bneck_out/${ds}_${summ}${width}.log"
      echo "[GPU $gpu] START $ds $summ w=$width"
      if CUDA_VISIBLE_DEVICES="$gpu" python bottleneck_probe.py \
            --dataset "$ds" --summarizer "$summ" --width "$width" \
            --epochs "$EPOCHS" --seeds "$SEEDS" --batch_size "$BS" --cache_dir "$CACHE_DIR" \
            --out_csv "$OUT" > "$log" 2>&1; then
        echo "[GPU $gpu] DONE  $ds $summ w=$width"
      else
        echo "[GPU $gpu] FAIL  $ds $summ w=$width (see $log)"
      fi
    fi
    idx=$((idx + 1))
  done
}
for slot in "${!GPU_ARR[@]}"; do dispatch "$slot" & done
wait

# ---- report ----------------------------------------------------------------
echo
echo "=== bottleneck sweep ($OUT) ==="
column -s, -t "$OUT" 2>/dev/null || cat "$OUT"
echo
python - "$OUT" <<'PY'
import csv, sys
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1])))
by_ds = defaultdict(list)
for r in rows:
    by_ds[r["dataset"]].append(r)
for ds, rs in by_ds.items():
    base = next((float(r["test_SRCC"]) for r in rs if r["summarizer"] == "ap"), None)
    best = max(rs, key=lambda r: float(r["test_SRCC"]))
    print(f"\n{ds}:  ALF(ap) baseline SRCC={base:.4f}" if base is not None else f"\n{ds}:")
    for r in sorted(rs, key=lambda r: (r["summarizer"], int(r["width"]))):
        d = (float(r["test_SRCC"]) - base) if base is not None else 0.0
        flag = "  <-- best" if r is best else ""
        print(f"  {r['summarizer']:4s} w={r['width']:>1}  T={r['tokens']:>3}  "
              f"SRCC={r['test_SRCC']}+/-{r['test_SRCC_std']}  d(alf)={d:+.4f}{flag}")
    if base is not None and float(best['test_SRCC']) - base < 0.003:
        print("  => widest arm ~= ALF: per-layer bottleneck NOT limiting here (negative result)")
PY

if [ "$KEEP_CACHE" != "1" ]; then
  echo; echo "Removing cache ($CACHE_DIR). Set KEEP_CACHE=1 to keep it."
  rm -rf "$CACHE_DIR"
fi
