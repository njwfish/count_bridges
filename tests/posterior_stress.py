"""Stress-test the Laplace u-posterior sampler across extreme regimes.

Regimes:
  - Tiny |z|:   z in {0, 1e-3, 1e-1, 1.0}
  - Moderate:   z in {10, 100}
  - Huge:       z in {1e3, 1e4, 1e5, 1e6}
  - Edge t:     t in {0.01, 0.1, 0.5, 0.9, 0.99}
  - Small / normal / large lam: lam in {64, 1024, 16384}

For each (t, sigma, lam, z) we compute a fine-grid reference (grid range
scaled to the regime) and compare Laplace samples to it.
"""

import math, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from posterior_samplers_compare import grid_sample_1d, laplace_sample, moments


def rel(a, b, eps=1e-9):
    return abs(a - b) / max(abs(b), eps)


def run(n_samples=80_000):
    print(f"{'t':>5} {'σ':>4} {'λ':>7} {'z':>10} │ "
          f"{'true_u ≈':>10}  {'u_mode Lap':>10}  {'mean_REF':>10}  {'mean_Lap':>10} │ "
          f"{'std_REF':>10}  {'std_Lap':>10}  {'|Δstd|%':>8}")
    print("-" * 130)

    torch.manual_seed(0)
    for t_val in [0.01, 0.1, 0.5, 0.9, 0.99]:
        for sigma in [1.0]:
            for lam in [64.0, 1024.0, 16384.0]:
                for z_val in [0.0, 1e-3, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6]:
                    # bridge std at this (t, lam, z_val/t roughly) — pick grid range 10x that
                    implied_u = abs(z_val) / t_val
                    sp = sigma * (4 * lam + implied_u ** 2) ** 0.25 * math.sqrt(t_val * (1 - t_val))
                    ref_range = max(10.0 * sp + 5.0 * implied_u + 10.0, 100.0)

                    try:
                        u_ref = grid_sample_1d(z_val, t_val, sigma, lam,
                                               16384, ref_range, n_samples,
                                               dtype=torch.float64)
                    except Exception as e:
                        print(f"{t_val:>5.2f} {sigma:>4.1f} {lam:>7.0f} {z_val:>10.2e} │ REF FAILED: {e}")
                        continue

                    z_tensor = torch.full((n_samples,), z_val, dtype=torch.float64)
                    try:
                        u_lap = laplace_sample(z_tensor, t_val, sigma, lam,
                                               kappa=1.0, newton_steps=4)
                    except Exception as e:
                        print(f"{t_val:>5.2f} {sigma:>4.1f} {lam:>7.0f} {z_val:>10.2e} │ LAP FAILED: {e}")
                        continue

                    m_r, s_r, _, _ = moments(u_ref)
                    m_l, s_l, _, _ = moments(u_lap)
                    rel_std = 100.0 * rel(s_l, s_r)

                    # also capture Laplace mode via one zero-noise call
                    # (repeat sample with r = r_mode)
                    from posterior_samplers_compare import laplace_sample as _ls
                    # hackily get mode: sample once with very small tau by kappa→0
                    u_mode_samples = laplace_sample(z_tensor[:8], t_val, sigma, lam,
                                                    kappa=1e-10, newton_steps=4)
                    u_mode = u_mode_samples[0].item()

                    print(f"{t_val:>5.2f} {sigma:>4.1f} {lam:>7.0f} {z_val:>10.2e} │ "
                          f"{implied_u:>10.2e}  {u_mode:>10.2e}  {m_r:>10.2e}  {m_l:>10.2e} │ "
                          f"{s_r:>10.3e}  {s_l:>10.3e}  {rel_std:>7.2f}%")


if __name__ == "__main__":
    run()
