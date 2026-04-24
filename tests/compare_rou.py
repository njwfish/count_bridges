"""Compare RoU sampler vs Laplace vs fine-grid reference across regimes."""

import math, os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.posterior_samplers_compare import grid_sample_1d, laplace_sample, moments
from tests.rou_sampler import sample_u_conditional_rou


def rel(a, b, eps=1e-9):
    return abs(a - b) / max(abs(b), eps)


def run(n_samples=50_000):
    print(f"{'t':>5} {'σ':>4} {'λ':>6} {'z':>10} │ "
          f"{'std_REF':>10}  {'std_Lap':>10}  {'|ΔLap|%':>8}  {'std_RoU':>10}  {'|ΔRoU|%':>8}  {'RoU_accept':>10}")
    print("-" * 130)

    torch.manual_seed(0)
    for t_val in [0.1, 0.5, 0.9]:
        for sigma in [1.0]:
            for lam in [64.0, 1024.0, 16384.0]:
                for z_val in [0.0, 1.0, 100.0, 1e4, 1e6]:
                    implied_u = abs(z_val) / t_val
                    sp = sigma * (4 * lam + implied_u ** 2) ** 0.25 * math.sqrt(t_val * (1 - t_val))
                    ref_range = max(10.0 * sp + 5.0 * implied_u + 10.0, 100.0)

                    u_ref = grid_sample_1d(z_val, t_val, sigma, lam,
                                           16384, ref_range, n_samples,
                                           dtype=torch.float64)
                    _, s_r, _, _ = moments(u_ref)

                    z_tensor = torch.full((n_samples,), z_val, dtype=torch.float64)
                    u_lap = laplace_sample(z_tensor, t_val, sigma, lam, kappa=1.0, newton_steps=4)
                    _, s_l, _, _ = moments(u_lap)

                    try:
                        u_rou, rounds, total = sample_u_conditional_rou(
                            z_tensor,
                            torch.full_like(z_tensor, t_val),
                            torch.full_like(z_tensor, sigma),
                            torch.full_like(z_tensor, lam),
                        )
                        _, s_rou, _, _ = moments(u_rou)
                        accept_rate = n_samples / total
                        rou_str = f"{s_rou:>10.3e}  {100*rel(s_rou,s_r):>7.2f}%  {accept_rate*100:>8.1f}%"
                    except Exception as e:
                        rou_str = f"FAILED ({type(e).__name__})"

                    print(f"{t_val:>5.2f} {sigma:>4.1f} {lam:>6.0f} {z_val:>10.2e} │ "
                          f"{s_r:>10.3e}  {s_l:>10.3e}  {100*rel(s_l,s_r):>7.2f}%  {rou_str}")


if __name__ == "__main__":
    run()
