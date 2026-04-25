"""
Run the Bayes oracle denoiser through the existing reverse sampler and
report metrics. mode='mean' returns the conditional mean E[X_0 | X_t]
(the MSE-style ablation); mode='sample' returns one draw from p(X_0 | X_t).

Bridge / sampler dispatch:
  - GaussianBridge(sigma=...): GaussianBridge.sample_step is the full
    bridge conditional given x_0_pred (or the reverse-SDE Euler step
    when mode='sde').
  - ClockedGaussianBridge(sigma=..., gamma=..., eta=...): IG-clocked
    bridge with mode='bridge' or mode='sde' (D_IG quadrature).

Outputs metrics to outputs/oracle_<cell>_n<NFE>.yaml.
"""

import argparse
import logging
import sys
import os
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.gaussian_mixture import LowRankGaussianMixtureDataset
from bridges.torch.gaussian_bridge import GaussianBridge
from bridges.torch.clocked_gaussian_bridge import ClockedGaussianBridge
from models.bayes import BayesDenoiserGMMFixed, BayesDenoiserMC
from evaluate import evaluate_model


def build_dataset(seed):
    return LowRankGaussianMixtureDataset(
        size=50000,
        data_dim=50,
        latent_dim=20,
        value_range=256,
        min_value=0,
        k=5,
        mean_scale=20.0,
        cov_scale=10.0,
        noise_scale=1.0,
        projection_scale=1.0,
        min_eigenvalue=0.1,
        seed=seed,
        continuous=True,
    )


def build_oracle(bridge_kind, dataset, sigma, gamma, nu):
    if bridge_kind == "fixed":
        return BayesDenoiserGMMFixed.from_dataset(dataset, sigma=sigma, mode="mean")
    elif bridge_kind == "clocked":
        return BayesDenoiserMC.from_dataset(
            dataset, kind="clocked", sigma=sigma, gamma=gamma, nu=nu,
            n_per_pair=4096, mode="mean", seed=0,
        )
    raise ValueError(bridge_kind)


def build_bridge(bridge_kind, sigma, gamma, nu):
    if bridge_kind == "fixed":
        # MSE-style: reverse-SDE mode (the Bayes oracle returns E[X_0|X_t]).
        return GaussianBridge(sigma=sigma, mode="sde", eta=1.0)
    elif bridge_kind == "clocked":
        return ClockedGaussianBridge(sigma=sigma, gamma=gamma, nu=nu,
                                     mode="sde", eta=1.0)
    raise ValueError(bridge_kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", choices=["fixed", "clocked"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_steps", type=int, default=10)
    ap.add_argument("--n_samples", type=int, default=10000)
    ap.add_argument("--sigma", type=float, default=None,
                    help="bridge noise scalar; default 100 for fixed, 1.0 for clocked")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="IG-clock gamma (only for kind='clocked')")
    ap.add_argument("--nu", type=float, default=64.0,
                    help="IG-clock rate `nu` (the note's eta; only for kind='clocked')")
    ap.add_argument("--out_dir", type=str, default="outputs/bayes_oracle")
    args = ap.parse_args()

    if args.sigma is None:
        args.sigma = 100.0 if args.bridge == "fixed" else 1.0

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info(f"Building continuous GMM dataset (seed={args.seed})")
    dataset = build_dataset(args.seed)
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    _, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    logging.info(f"Building bridge ({args.bridge}, sigma={args.sigma}, gamma={args.gamma}, nu={args.nu})")
    bridge = build_bridge(args.bridge, args.sigma, args.gamma, args.nu)

    logging.info(f"Building Bayes oracle denoiser ({args.bridge}, mode='mean')")
    oracle = build_oracle(args.bridge, dataset, args.sigma, args.gamma, args.nu).cuda()

    out_dir = Path(args.out_dir) / f"{args.bridge}_seed{args.seed}_n{args.n_steps}"
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(vars(args)))

    logging.info(f"Running evaluation: n_steps={args.n_steps}, n_samples={args.n_samples}")
    eval_result, _ = evaluate_model(
        oracle, bridge, eval_dataset,
        config_path=config_path,
        n_samples=args.n_samples, n_steps=args.n_steps,
        force_regenerate=True,
    )

    metrics = eval_result["metrics"]
    print("\n=== Bayes Oracle Metrics ===")
    print(f"  cell: bayes-{args.bridge}-MSE-mode  seed={args.seed}  n_steps={args.n_steps}")
    for k in ['energy_distance', 'wasserstein_distance', 'mmd_rbf',
              'covariance_frobenius', 'mean_error', 'variance_error',
              'skewness_error', 'kurtosis_error']:
        if k in metrics:
            print(f"    {k}: {metrics[k]:.4f}")


if __name__ == "__main__":
    main()
