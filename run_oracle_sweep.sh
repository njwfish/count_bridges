#!/bin/bash
# IG-clocked vs fixed Bayes-oracle sweep on the d=50 r=20 continuous
# LowRank-GMM benchmark. Each run: stratified-SNIS oracle (kind='fixed' or
# kind='clocked') + matched reverse sampler ('sde' mode, eta=1.0). Sweep
# is 2 bridge kinds x 3 seeds x 3 NFEs = 18 eval runs + 6 checkpoint builds.
set -e
cd "$(dirname "$0")"

SEEDS=(0 1 42)
NFES=(10 100 1000)

log() { echo "[sweep $(date '+%H:%M:%S')] $*"; }

build_and_eval() {
  local kind=$1                # SNIS oracle kind: 'fixed' or 'clocked'
  local bridge_cfg=$2          # bridge yaml
  local bridge_overrides=$3
  local seed=$4

  local name="snis_${kind}_s${seed}"
  local common="model=bayes_mc model.kind=${kind} bridge=${bridge_cfg} ${bridge_overrides} architecture/in_dims=standard averaging=ema dataset.data_dim=50 dataset.latent_dim=20 dataset.continuous=true dataset.seed=${seed} architecture.act_fn=identity seed=${seed} experiment.name=${name}"

  log "BUILD ${name}"
  python scripts/build_bayes_checkpoint.py ${common} > /dev/null 2>&1 || {
    log "  build FAILED"; return 1;
  }

  for nfe in "${NFES[@]}"; do
    log "EVAL  ${name} NFE=${nfe} mode=sde eta=1.0"
    python main.py ${common} bridge.mode=sde bridge.eta=1.0 +n_steps=${nfe} > /dev/null 2>&1 || {
      log "  eval FAILED (nfe=${nfe})"; continue;
    }
  done
}

# IG-clocked: gamma=1, eta=64 matches the legacy heuristic adaptive bridge
# at u=0 (with lam=1024).
for seed in "${SEEDS[@]}"; do
  build_and_eval clocked clocked_gaussian_bridge "" "${seed}"
done

# Fixed: bridge sigma=100 to match the data scale.
for seed in "${SEEDS[@]}"; do
  build_and_eval fixed gaussian_bridge "bridge.sigma=100" "${seed}"
done

log "SWEEP COMPLETE"
