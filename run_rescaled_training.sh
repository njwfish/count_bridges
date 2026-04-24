#!/bin/bash
# Rescaled [-1, 1] training sweep on d=50 r=20 continuous GMM.
# 2 bridges x 2 losses x 3 seeds = 12 runs, evals at NFE={10, 100, 1000}.
# Tests whether data scale is important (matched-dimensionless bridge params).
#
# Usage: CUDA_VISIBLE_DEVICES=<gpu_id> SEED=<seed> bash run_rescaled_training.sh
set -e
cd "$(dirname "$0")"
mkdir -p logs

SEED=${SEED:-42}
NFES=(10 100 1000)

log() { echo "[rtrain gpu=${CUDA_VISIBLE_DEVICES:-?} seed=${SEED} $(date '+%H:%M:%S')] $*"; }

DATA_OVR="dataset.min_value=-1 dataset.value_range=1 dataset.noise_scale=0.1 dataset.projection_scale=0.1"
DATA_COMMON="dataset.data_dim=50 dataset.latent_dim=20 dataset.continuous=true"

run_cell() {
  local model_cfg=$1          # energy_score or mse
  local bridge_cfg=$2         # gaussian_bridge or adaptive_gaussian_bridge
  local bridge_overrides=$3   # extra sigma/lam for this bridge
  local tag=$4

  # round_output only on energy_score; architecture in_dims differs
  local in_dims="standard"
  local extra=""
  if [[ "${model_cfg}" == "energy_score" ]]; then
    extra="model.round_output=false"
    in_dims="with_noise"
  fi

  local name="rtrain_${tag}_s${SEED}"
  local common="model=${model_cfg} ${extra} bridge=${bridge_cfg} ${bridge_overrides} architecture/in_dims=${in_dims} averaging=ema ${DATA_COMMON} ${DATA_OVR} dataset.seed=${SEED} architecture.act_fn=identity seed=${SEED} experiment.name=${name}"

  log "TRAIN ${name}"
  python main.py ${common} > logs/${name}.log 2>&1 || {
    log "  train FAILED"; return 1;
  }

  for nfe in "${NFES[@]}"; do
    log "EVAL  ${name} NFE=${nfe} sde eta=1.0"
    python main.py ${common} bridge.mode=sde bridge.eta=1.0 +n_steps=${nfe} >> logs/${name}.log 2>&1 || {
      log "  eval NFE=${nfe} FAILED"; continue;
    }
  done
}

run_cell energy_score gaussian_bridge          "bridge.sigma=10"                     "es_fixed"
run_cell energy_score adaptive_gaussian_bridge "bridge.sigma=0.3162 bridge.lam=10.24" "es_adapt"
run_cell mse          gaussian_bridge          "bridge.sigma=10"                     "mse_fixed"
run_cell mse          adaptive_gaussian_bridge "bridge.sigma=0.3162 bridge.lam=10.24" "mse_adapt"

log "SEED ${SEED} SWEEP COMPLETE"
