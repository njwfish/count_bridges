"""
Build a Bayes-oracle checkpoint that lives at the same outputs/<hash>/model.pt
location main.py would write a trained checkpoint to. After running this,
`python main.py model=bayes_{fixed,adaptive} ...` finds the checkpoint, sees
training is complete, loads the buffers, and runs the existing sampling/eval
pipeline -- no separate evaluation code path needed.

Usage:
    python scripts/build_bayes_checkpoint.py model=bayes_fixed bridge=gaussian_bridge \
        bridge.sigma=100 averaging=ema dataset.data_dim=50 dataset.latent_dim=20 \
        dataset.continuous=true seed=0 experiment.name=oracle_fixed_s0

Then:
    python main.py model=bayes_fixed bridge=gaussian_bridge bridge.sigma=100 \
        averaging=ema dataset.data_dim=50 dataset.latent_dim=20 dataset.continuous=true \
        seed=0 experiment.name=oracle_fixed_s0 +n_steps=10
"""

from pathlib import Path
import os
import sys
import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Make package imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import get_model_hash
from models.bayes import (
    BayesDenoiserGMMFixed,
    BayesDenoiserMC,
)

ORIGINAL_CWD = Path(__file__).resolve().parent.parent


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Compute the same output dir / hash main.py would use
    config_hash = get_model_hash(cfg)
    output_dir = ORIGINAL_CWD / "outputs" / config_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Bayes checkpoint dir: {output_dir}")

    # Save resolved config (mirrors main.py's behaviour)
    config_path = output_dir / "config.yaml"
    if not config_path.exists():
        config_path.write_text(OmegaConf.to_yaml(cfg))

    # Build dataset and Bayes denoiser via from_dataset
    dataset = hydra.utils.instantiate(cfg.dataset)
    target = cfg.model._target_
    if target.endswith("BayesDenoiserGMMFixed"):
        model = BayesDenoiserGMMFixed.from_dataset(
            dataset, sigma=cfg.bridge.sigma, mode=cfg.model.get('mode', 'mean')
        )
    elif target.endswith("BayesDenoiserMC"):
        # Stratified SNIS over (c, c') pairs.
        kind = cfg.model.get('kind', 'fixed')
        gamma = float(cfg.bridge.get('gamma', cfg.model.get('gamma', 1.0)))
        # Allow either `nu` (preferred) or fallback to `model.nu` for old configs.
        nu = float(cfg.bridge.get('nu', cfg.model.get('nu', 64.0)))
        model = BayesDenoiserMC.from_dataset(
            dataset,
            kind=kind,
            sigma=float(cfg.bridge.sigma),
            gamma=gamma, nu=nu,
            n_per_pair=int(cfg.model.get('n_per_pair', 4096)),
            mode=cfg.model.get('mode', 'mean'),
            seed=int(cfg.model.get('seed', 0)),
        )
    else:
        raise ValueError(f"Unknown Bayes model target: {target}")

    logging.info(f"Built {type(model).__name__} from dataset (k={dataset.k}, d={dataset.data_dim})")

    # Save a "trained" checkpoint at outputs/<hash>/model.pt.
    # If `averaging=ema` is in the config, main.py builds a torch.optim.swa_utils.
    # AveragedModel(model) and (in the training-complete branch) calls
    # avg_model.load_state_dict(ckpt['avg_model_state_dict']). AveragedModel's
    # state_dict prefixes the wrapped model's keys with `module.` and adds an
    # `n_averaged` long buffer, so we mirror that layout here.
    num_epochs = cfg.training.num_epochs
    sd = model.state_dict()
    avg_sd = None
    if 'averaging' in cfg and cfg.averaging is not None:
        avg_sd = {f'module.{k}': v for k, v in sd.items()}
        avg_sd['n_averaged'] = torch.tensor(1, dtype=torch.long)
    ckpt = {
        'model_state_dict': sd,
        'avg_model_state_dict': avg_sd,
        'optimizer_state_dict': {},
        'scheduler_state_dict': None,
        'losses': [0.0],
        'current_epoch': num_epochs,   # mark training as complete
        'total_epochs': num_epochs,
    }
    ckpt_path = output_dir / "model.pt"
    torch.save(ckpt, ckpt_path)
    logging.info(f"Wrote Bayes checkpoint -> {ckpt_path} (current_epoch={num_epochs}, total_epochs={num_epochs})")
    logging.info(f"Now run: python main.py [same overrides] +n_steps=N")


if __name__ == "__main__":
    main()
