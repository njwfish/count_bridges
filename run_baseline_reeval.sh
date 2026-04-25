#!/bin/bash
# Re-evaluate the existing cont2_* baseline checkpoints at NFE=100 and 1000.
# These are the same trained models used in the original NFE=10 sweeps; we
# just want the additional NFE points for the fixed-vs-clocked comparison.
#
# Each invocation reproduces the original training overrides so main.py finds
# the existing checkpoint via get_model_hash. Adds +n_steps=100/1000.
set -e
cd "$(dirname "$0")"

SEEDS=(0 1 42)
NFES=(100 1000)

log() { echo "[reeval $(date '+%H:%M:%S')] $*"; }

mkdir -p logs/baseline_reeval

run_eval() {
  local kind=$1            # 'adapt' or 'gauss'
  local loss=$2            # 'es' or 'mse'
  local seed=$3
  local nfe=$4
  local name="cont2_${kind}_${loss}_s${seed}"

  if [ "$loss" = "es" ]; then
    model_arg="model=energy_score"; in_dims="with_noise"
  else
    model_arg="model=mse"; in_dims="standard"
  fi
  if [ "$kind" = "adapt" ]; then
    bridge_arg="bridge=adaptive_gaussian_bridge bridge.sigma=1.0 bridge.lam=1024.0"
  else
    bridge_arg="bridge=gaussian_bridge bridge.sigma=100"
  fi

  python main.py ${model_arg} ${bridge_arg} \
    architecture/in_dims=${in_dims} architecture.act_fn=identity averaging=ema \
    dataset.data_dim=50 dataset.latent_dim=20 dataset.continuous=true dataset.seed=${seed} \
    seed=${seed} experiment.name=${name} \
    bridge.mode=bridge bridge.eta=1.0 \
    +n_steps=${nfe} +n_samples=10000 \
    > logs/baseline_reeval/${name}_n${nfe}.log 2>&1
}

# 4 baselines (adapt/gauss x es/mse) x 3 seeds (where available) x 2 NFEs.
# cont2_adapt_mse only has seeds 1, 42; cont2_gauss_mse only seed 0.
for nfe in "${NFES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    log "EVAL cont2_adapt_es_s${seed}  NFE=${nfe}"
    run_eval adapt es ${seed} ${nfe} || log "  failed"
    log "EVAL cont2_gauss_es_s${seed}  NFE=${nfe}"
    run_eval gauss es ${seed} ${nfe} || log "  failed"
  done
  for seed in 1 42; do
    log "EVAL cont2_adapt_mse_s${seed}  NFE=${nfe}"
    run_eval adapt mse ${seed} ${nfe} || log "  failed"
  done
  log "EVAL cont2_gauss_mse_s0  NFE=${nfe}"
  run_eval gauss mse 0 ${nfe} || log "  failed"
done

log "REEVAL COMPLETE"
