#!/bin/bash
# Maximally-parallel (seed, NFE, kind, sigma, [lam]) grid launcher.
# Packs multiple evals per GPU. Uses xargs -P for concurrency.
#
# Usage:
#   bash scripts/tune_oracle_launch.sh [n_workers_per_gpu] [n_gpus]
# defaults to 4 workers/GPU across 4 GPUs = 16-way parallel.

set -e
cd "$(dirname "$0")/.."
mkdir -p outputs/tune_oracle_grid logs

NW=${1:-4}      # workers per GPU
GPUS=${GPUS:-"1 2 3"}   # space-separated GPU ids (default skip GPU 0 for oracle sweep)
GPU_ARR=(${GPUS})
NG=${#GPU_ARR[@]}
TOTAL=$((NW*NG))

SEEDS=(0 1 42)
NFES=(10 100 1000)
FIXED_SIGMAS=(10 30 50 100 150 200 300)
ADAPTIVE_SIGMAS=(0.5 1 2 5)
ADAPTIVE_LAMS=(256 1024 4096 16384)

# Emit job lines: "<gpu_id>:<cmd>"
JOBS_FILE=$(mktemp /tmp/tune_jobs_XXXXXX.txt)
idx=0
for seed in "${SEEDS[@]}"; do
  for nfe in "${NFES[@]}"; do
    for sigma in "${FIXED_SIGMAS[@]}"; do
      gpu=${GPU_ARR[$(( idx % NG ))]}
      idx=$((idx+1))
      echo "${gpu}|python scripts/tune_oracle_one.py --seed ${seed} --nfe ${nfe} --kind fixed --sigma ${sigma}" >> "${JOBS_FILE}"
    done
    for sigma in "${ADAPTIVE_SIGMAS[@]}"; do
      for lam in "${ADAPTIVE_LAMS[@]}"; do
        gpu=${GPU_ARR[$(( idx % NG ))]}
        idx=$((idx+1))
        echo "${gpu}|python scripts/tune_oracle_one.py --seed ${seed} --nfe ${nfe} --kind adaptive --sigma ${sigma} --lam ${lam}" >> "${JOBS_FILE}"
      done
    done
  done
done

N=$(wc -l < "${JOBS_FILE}")
echo "[launcher] $N jobs, $TOTAL concurrent (${NW}/GPU x ${NG} GPUs)"

# Run in parallel with CUDA_VISIBLE_DEVICES set per line
xargs -P "${TOTAL}" -a "${JOBS_FILE}" -I{} bash -c '
  line="{}"
  gpu="${line%%|*}"
  cmd="${line#*|}"
  CUDA_VISIBLE_DEVICES=${gpu} ${cmd}
' 2>&1 | tee "logs/tune_oracle_grid_$(date +%Y%m%d_%H%M%S).log"

echo "[launcher] done"
