"""
Grid-search (sigma, lam) for the Bayes-optimal SNIS oracle.

Given the d=50 r=20 continuous LowRankGMM benchmark, for each bridge kind and
each (sigma, [lam]) combination:
  1. Build the stratified-SNIS BayesDenoiserMC with those hyperparameters.
  2. Sample x_0 from x_1 ~ p_1 through the matched reverse SDE at fixed NFE.
  3. Compute energy_distance vs the held-out target population.

Bypasses Hydra / main.py and reuses the dataset + sampler directly, so each
eval is ~3-5x faster than a full main.py run.

Usage (single GPU, all configs):
  CUDA_VISIBLE_DEVICES=<gpu> python scripts/tune_oracle_bridge.py

Usage (split across 2 GPUs):
  CUDA_VISIBLE_DEVICES=1 python scripts/tune_oracle_bridge.py --kind fixed    &
  CUDA_VISIBLE_DEVICES=2 python scripts/tune_oracle_bridge.py --kind adaptive &
"""

import argparse
from pathlib import Path
import sys
import time
import json
import itertools

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.gaussian_mixture import LowRankGaussianMixtureDataset
from models.bayes import BayesDenoiserMC
from bridges.torch.gaussian_bridge import GaussianBridge
from bridges.torch.adaptive_gaussian_bridge import AdaptiveGaussianBridge
from metrics import energy_distance


def build_dataset(seed=42):
    return LowRankGaussianMixtureDataset(
        size=50000, data_dim=50, latent_dim=20,
        value_range=256, min_value=0,
        k=5, mean_scale=20.0, cov_scale=10.0,
        noise_scale=1.0, projection_scale=1.0,
        min_eigenvalue=0.1,
        seed=seed, continuous=True,
    )


def build_bridge(kind, sigma, lam, device):
    if kind == "fixed":
        return GaussianBridge(sigma=float(sigma), device=device,
                              homogeneous_time=False, mode="sde", eta=1.0)
    else:
        return AdaptiveGaussianBridge(sigma=float(sigma), lam=float(lam),
                                       device=device, homogeneous_time=False,
                                       mode="sde", eta=1.0, gl_n=128, t_eps=1e-3)


def build_oracle(dataset, kind, sigma, lam, device, n_per_pair=4096, seed=0):
    m = BayesDenoiserMC.from_dataset(
        dataset, kind=kind, sigma=float(sigma),
        lam=float(lam) if lam is not None else 1024.0,
        n_per_pair=n_per_pair, mode='mean', seed=seed,
    )
    return m.to(device)


def run_eval(dataset, kind, sigma, lam, n_steps, n_samples, device, seed=0):
    """Return energy_distance and elapsed seconds."""
    t0 = time.time()
    bridge = build_bridge(kind, sigma, lam, device)
    model = build_oracle(dataset, kind, sigma, lam, device, seed=seed)

    # Sample a batch of x_1 from the dataset's target distribution
    n = min(n_samples, len(dataset))
    # Take the first n entries for determinism
    x_1 = dataset.x1_data[:n].float().to(device)
    x_0_target = dataset.x0_data[:n].float().cpu().numpy()

    with torch.no_grad():
        x0_generated = bridge.sampler(
            x_1=x_1, z={}, model=model,
            n_steps=n_steps,
            return_trajectory=False, return_x_hat=False,
        )
    x0_generated_np = x0_generated.detach().cpu().numpy()
    e = energy_distance(x_0_target, x0_generated_np)
    elapsed = time.time() - t0
    return e, elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["fixed", "adaptive", "both"], default="both")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nfe", type=int, default=100)
    p.add_argument("--n_samples", type=int, default=2000)
    args = p.parse_args()

    device = "cuda"

    dataset = build_dataset(seed=args.seed)

    fixed_sigmas = [10.0, 30.0, 50.0, 100.0, 150.0, 200.0, 300.0]
    adaptive_sigmas = [0.5, 1.0, 2.0, 5.0]
    adaptive_lams = [256.0, 1024.0, 4096.0, 16384.0]

    out = {"fixed": [], "adaptive": [], "meta": {
        "seed": args.seed, "n_samples": args.n_samples, "n_steps": args.nfe,
        "dataset": "LowRankGMM d=50 r=20 continuous k=5",
        "kind_run": args.kind,
    }}

    if args.kind in ("fixed", "both"):
        print(f"Fixed bridge sweep (NFE={args.nfe}, n_samples={args.n_samples}, seed={args.seed}):")
        for sigma in fixed_sigmas:
            e, dt = run_eval(dataset, "fixed", sigma, None, args.nfe, args.n_samples, device)
            print(f"  sigma={sigma:>6g}  E-dist={e:>8.4f}  ({dt:.1f}s)")
            out["fixed"].append({"sigma": sigma, "energy_distance": e, "seconds": dt})

    if args.kind in ("adaptive", "both"):
        print(f"\nAdaptive bridge sweep (NFE={args.nfe}):")
        for sigma, lam in itertools.product(adaptive_sigmas, adaptive_lams):
            e, dt = run_eval(dataset, "adaptive", sigma, lam, args.nfe, args.n_samples, device)
            print(f"  sigma={sigma:>6g}  lam={lam:>6g}  E-dist={e:>8.4f}  ({dt:.1f}s)")
            out["adaptive"].append({"sigma": sigma, "lam": lam, "energy_distance": e, "seconds": dt})

    if out["fixed"]:
        best_f = min(out["fixed"], key=lambda r: r["energy_distance"])
        print(f"\n=== best fixed ===")
        print(f"sigma={best_f['sigma']}  E-dist={best_f['energy_distance']:.4f}")
    if out["adaptive"]:
        best_a = min(out["adaptive"], key=lambda r: r["energy_distance"])
        print(f"\n=== best adaptive ===")
        print(f"sigma={best_a['sigma']} lam={best_a['lam']}  E-dist={best_a['energy_distance']:.4f}")

    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "tune_oracle"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sweep_seed{args.seed}_nfe{args.nfe}_{args.kind}.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
