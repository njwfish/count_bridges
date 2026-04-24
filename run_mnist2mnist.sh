#!/bin/bash
# MNIST2MNIST conditional-bridge 2x2 training sweep.
# 2 losses (MSE, energy-score) x 2 bridges (fixed sigma, adaptive).
# CUDA_VISIBLE_DEVICES selects GPU externally.
set -e
cd "$(dirname "$0")"
mkdir -p logs

SEED=${SEED:-42}
EPOCHS=${EPOCHS:-50}

log() { echo "[mnist2mnist gpu=${CUDA_VISIBLE_DEVICES:-?} seed=${SEED} $(date '+%H:%M:%S')] $*"; }

# Bridge params tuned for images in [-1, 1]:
#   fixed:    sigma=1.0  (bridge noise std at t=0.5 -> 0.5, ~data scale)
#   adaptive: sigma=1.0, lam=0.25  (matches avg var at u=0 to fixed; grows linearly at large u)
FIXED_BR="bridge=gaussian_bridge bridge.sigma=1.0"
ADAPT_BR="bridge=adaptive_gaussian_bridge bridge.sigma=1.0 bridge.lam=0.25"

# MSE cells (architecture = uvit_small_cond, no noise channel)
run_mse() {
  local bridge_cfg=$1
  local tag=$2
  local name="m2m_mse_${tag}_s${SEED}"
  log "TRAIN ${name}"
  python main.py --config-name=mnist2mnist \
    model=mse \
    ${bridge_cfg} \
    training.num_epochs=${EPOCHS} \
    seed=${SEED} experiment.name=${name} > logs/${name}.log 2>&1 || { log "  FAILED"; return 1; }

  for nfe in 10 100; do
    log "EVAL ${name} NFE=${nfe}"
    python main.py --config-name=mnist2mnist \
      model=mse ${bridge_cfg} \
      bridge.mode=sde bridge.eta=1.0 \
      training.num_epochs=${EPOCHS} \
      seed=${SEED} experiment.name=${name} +n_steps=${nfe} >> logs/${name}.log 2>&1 || log "  eval NFE=${nfe} FAILED"
  done
}

# Energy-score cells (architecture = uvit_small_cond WITH noise channel)
run_es() {
  local bridge_cfg=$1
  local tag=$2
  local name="m2m_es_${tag}_s${SEED}"
  log "TRAIN ${name}"
  python main.py --config-name=mnist2mnist \
    model=energy_score model.round_output=false \
    architecture.noise_dim=100 \
    ${bridge_cfg} \
    training.num_epochs=${EPOCHS} \
    seed=${SEED} experiment.name=${name} > logs/${name}.log 2>&1 || { log "  FAILED"; return 1; }

  for nfe in 10 100; do
    log "EVAL ${name} NFE=${nfe}"
    python main.py --config-name=mnist2mnist \
      model=energy_score model.round_output=false \
      architecture.noise_dim=100 \
      ${bridge_cfg} \
      bridge.mode=bridge bridge.eta=1.0 \
      training.num_epochs=${EPOCHS} \
      seed=${SEED} experiment.name=${name} +n_steps=${nfe} >> logs/${name}.log 2>&1 || log "  eval NFE=${nfe} FAILED"
  done
}

run_mse "${FIXED_BR}" "fixed"
run_mse "${ADAPT_BR}" "adapt"
run_es  "${FIXED_BR}" "fixed"
run_es  "${ADAPT_BR}" "adapt"

log "ALL 4 CELLS COMPLETE"
