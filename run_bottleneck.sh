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
#   ./run_bottleneck.sh                         # SigLIP2, KADID10K + KonIQ_10K, 3 GPUs
#   SEEDS=3 EPOCHS=25 ./run_bottleneck.sh       # smoother curves
#   DATASETS="KADID10K" GPUS="0" ./run_bottleneck.sh
#   CACHE=0 ./run_bottleneck.sh                  # no disk cache (forward per epoch)
#   MODELS="google/siglip2-so400m-patch16-512 openai/clip-vit-large-patch14 facebook/dinov2-large" \
#     ./run_bottleneck.sh                        # cross-backbone breadth

set -uo pipefail
cd "$(dirname "$0")"

MODELS="${MODELS:-google/siglip2-so400m-patch16-512}"   # space-separated HF ids
DATASETS="${DATASETS:-KADID10K KonIQ_10K}"
GPUS="${GPUS:-0 1 2}"
EPOCHS="${EPOCHS:-20}"
SEEDS="${SEEDS:-3}"                       # avg out head-init variance (cache is free)
SEED="${SEED:-42}"                        # split seed (cache path embeds it)
BS="${BS:-16}"                           # per-step batch (reads [B,L,N,D] into GPU)
CACHE="${CACHE:-1}"                       # 0 = no cache: forward backbone per epoch
                                          #     (avoids disk I/O; re-forwards per config)
CACHE_DIR="${CACHE_DIR:-/data/bneck_cache}"
KEEP_CACHE="${KEEP_CACHE:-1}"             # 1 = keep the token cache (default); 0 = delete after sweep

# Per-run output dir (mode + timestamp) so repeated / cache-vs-no-cache runs of
# the same setup don't overwrite each other's logs and results table.
MODE=$([ "$CACHE" = "1" ] && echo cache || echo nocache)
RUN="${RUN:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="bneck_out/${MODE}_${RUN}"
OUT="${OUT:-$RUN_DIR/bneck_results.csv}"

read -ra MODEL_ARR <<< "$MODELS"
read -ra DS_ARR  <<< "$DATASETS"
read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
mkdir -p "$RUN_DIR"
rm -f "$OUT"                              # fresh table; per-run rows are appended
echo "Run outputs -> $RUN_DIR"

# ---- summarizer configs: "summarizer|width" -------------------------------
CONFIGS=("ap|1" "pma|1" "pma|2" "pma|4" "pma|8" "tome|2" "tome|4" "tome|8")

# --cache_dir flag shared by both phases; empty in no-cache mode.
CACHE_FLAG=""
[ "$CACHE" = "1" ] && CACHE_FLAG="--cache_dir $CACHE_DIR"

# ---- phase 1: build one cache per (model, dataset) (serial; first GPU) ------
# Skip a build whose 3 split files already exist (cache path mirrors
# bottleneck_probe.py: <dir>/<ds>_s<seed>_<model-basename>/).
if [ "$CACHE" = "1" ]; then
  echo "=== phase 1: build caches (serial) ==="
  for m in "${MODEL_ARR[@]}"; do
    for ds in "${DS_ARR[@]}"; do
      cr="$CACHE_DIR/${ds}_s${SEED}_${m##*/}"
      if [ -f "$cr/train.f16" ] && [ -f "$cr/val.f16" ] && [ -f "$cr/test.f16" ]; then
        echo "[cache] ${m##*/} $ds -> reuse ($cr)"
        continue
      fi
      echo "[cache] ${m##*/} $ds -> build ($cr)"
      clog="$RUN_DIR/cache_${m##*/}_${ds}.log"
      CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python bottleneck_probe.py \
        --model_id "$m" --dataset "$ds" --seed "$SEED" --cache_dir "$CACHE_DIR" \
        --batch_size "$BS" --build_cache_only > "$clog" 2>&1
      rc=$?
      # rc 137 = SIGKILL (usually OOM: check `dmesg | tail`); 124 = launcher timeout.
      [ $rc -ne 0 ] && { echo "  FAIL rc=$rc (see $clog)"; exit 1; }
    done
  done
else
  echo "=== no-cache mode (CACHE=0): backbone forwards on the fly each epoch ==="
fi

# ---- phase 2: sweep configs round-robin across GPUs ------------------------
echo "=== phase 2: sweep ${#CONFIGS[@]} configs x ${#DS_ARR[@]} datasets x ${#MODEL_ARR[@]} models ==="
JOBS=()
for m in "${MODEL_ARR[@]}"; do
  for ds in "${DS_ARR[@]}"; do for c in "${CONFIGS[@]}"; do JOBS+=("$m|$ds|$c"); done; done
done

dispatch() {
  local slot=$1 gpu=${GPU_ARR[$slot]} idx=0 job
  for job in "${JOBS[@]}"; do
    if (( idx % NGPU == slot )); then
      IFS='|' read -r m ds summ width <<< "$job"
      local log="$RUN_DIR/${m##*/}_${ds}_${summ}${width}.log"
      echo "[GPU $gpu] START ${m##*/} $ds $summ w=$width"
      CUDA_VISIBLE_DEVICES="$gpu" python bottleneck_probe.py \
            --model_id "$m" --dataset "$ds" --seed "$SEED" --summarizer "$summ" --width "$width" \
            --epochs "$EPOCHS" --seeds "$SEEDS" --batch_size "$BS" $CACHE_FLAG \
            --out_csv "$OUT" > "$log" 2>&1
      rc=$?
      if [ $rc -eq 0 ]; then
        echo "[GPU $gpu] DONE  ${m##*/} $ds $summ w=$width"
      else  # rc 137 = SIGKILL/OOM (check `dmesg | tail`); 124 = launcher timeout
        echo "[GPU $gpu] FAIL rc=$rc ${m##*/} $ds $summ w=$width (see $log)"
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
by_grp = defaultdict(list)
for r in rows:
    by_grp[(r.get("model", "?"), r["dataset"])].append(r)
for (model, ds), rs in by_grp.items():
    base = next((float(r["test_SRCC"]) for r in rs if r["summarizer"] == "ap"), None)
    best = max(rs, key=lambda r: float(r["test_SRCC"]))
    head = f"\n{model} / {ds}:"
    print(f"{head}  ALF(ap) baseline SRCC={base:.4f}" if base is not None else head)
    for r in sorted(rs, key=lambda r: (r["summarizer"], int(r["width"]))):
        d = (float(r["test_SRCC"]) - base) if base is not None else 0.0
        flag = "  <-- best" if r is best else ""
        print(f"  {r['summarizer']:4s} w={r['width']:>1}  T={r['tokens']:>3}  "
              f"SRCC={r['test_SRCC']}+/-{r['test_SRCC_std']}  d(alf)={d:+.4f}{flag}")
    if base is not None and float(best['test_SRCC']) - base < 0.003:
        print("  => widest arm ~= ALF: per-layer bottleneck NOT limiting here (negative result)")
PY

if [ "$CACHE" = "1" ] && [ "$KEEP_CACHE" != "1" ]; then
  echo; echo "Removing cache ($CACHE_DIR). Set KEEP_CACHE=1 to keep it."
  rm -rf "$CACHE_DIR"
fi
