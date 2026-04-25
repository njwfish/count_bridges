#!/bin/bash
# Train ClockedGaussianBridge + EnergyScoreLoss on the d=50 r=20 continuous
# LowRank-GMM benchmark, matching the existing cont2_adapt_es / cont2_gauss_es
# trained checkpoints (500 epochs, MLP hidden=128 act=identity, batch=256).
#
# IG-clock calibration: gamma=1, nu=64 (matches the heuristic adaptive bridge
# at u=0 with lam=1024, since r*(0) = nu/gamma = sqrt(4*lam) = 64).
#
# Single GPU; runs all seeds sequentially.
set -e
cd "$(dirname "$0")"

SEEDS=(0 1 42)

log() { echo "[clocked-train $(date '+%H:%M:%S')] $*"; }

mkdir -p logs/clocked_train
for seed in "${SEEDS[@]}"; do
  name="cont2_clocked_es_s${seed}"
  log "TRAIN ${name}"
  python main.py model=energy_score \
    bridge=clocked_gaussian_bridge bridge.sigma=1.0 bridge.gamma=1.0 bridge.nu=64.0 \
    architecture/in_dims=with_noise architecture.act_fn=identity averaging=ema \
    dataset.data_dim=50 dataset.latent_dim=20 dataset.continuous=true dataset.seed=${seed} \
    seed=${seed} experiment.name=${name} +n_steps=10 +n_samples=2000 \
    > logs/clocked_train/s${seed}.log 2>&1 || {
    log "  TRAIN FAILED for seed=${seed}"; continue;
  }
  log "  done (seed=${seed})"
done

log "SWEEP COMPLETE"
