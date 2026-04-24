"""
Compare approximate u-posterior samplers to a fine-grid reference.

Posterior:
    p(u | z, t) ∝ (4λ + u²)^(-1/4) · exp(-(z - t·u)² / (2 σ² √(4λ+u²) · t(1-t)))

Samplers tested per (t, σ, λ, z):
  (A) inference-default grid (G=512, range=2000) — current sampler
  (B) Laplace-saddlepoint in r = asinh(u/(2√λ))   — user's proposal

Reference: fine-grid with G=16384 and range adaptive to z scale. We evaluate
the density ONCE on the reference grid (shape (G,)) then draw `n_samples`
samples from that 1-D CDF. Cost: O(G + n_samples), not O(G·n_samples).
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---- grid sampler helper (efficient: one shared grid density per z) ---------
def grid_sample_1d(z_val, t_val, sigma, lam, n_grid, u_range, n_samples,
                   device="cpu", dtype=torch.float64):
    """Sample n_samples u-values from p(u | z_val, t_val) on a grid of length n_grid
    covering [-u_range, u_range]. Efficient: density is computed once, samples
    via searchsorted on shared CDF."""
    u = torch.linspace(-u_range, u_range, n_grid, device=device, dtype=dtype)  # (G,)
    four_lam_u2 = 4.0 * lam + u ** 2                                           # (G,)
    alpha = sigma ** 2 * t_val * (1 - t_val)
    var = alpha * torch.sqrt(four_lam_u2)                                      # (G,)
    logp = -0.5 * (z_val - t_val * u) ** 2 / var - 0.25 * torch.log(four_lam_u2)
    logp = logp - logp.max()
    p = logp.exp()
    cdf = p.cumsum(dim=-1)
    cdf = cdf / cdf[-1]
    r = torch.rand(n_samples, device=device, dtype=dtype)
    idx = torch.searchsorted(cdf, r).clamp(max=n_grid - 1)
    return u[idx]


# ---- Laplace-saddlepoint sampler (from user's proposal) ---------------------
@torch.no_grad()
def laplace_sample(z, t, sigma, lam, kappa=1.25, newton_steps=4, eps=1e-8):
    dtype, device = z.dtype, z.device
    t = torch.as_tensor(t, dtype=dtype, device=device).clamp(eps, 1.0 - eps)
    sigma = torch.as_tensor(sigma, dtype=dtype, device=device).clamp_min(eps)
    lam = torch.as_tensor(lam, dtype=dtype, device=device).clamp_min(eps)
    sigma2 = sigma * sigma
    delta2 = 4.0 * lam
    delta = torch.sqrt(delta2)
    omt = 1.0 - t

    u = z / t
    for _ in range(newton_steps):
        s = torch.sqrt(u * u + delta2)
        v = z - t * u
        g = (u / (2.0 * s * s)
             + v / (sigma2 * omt * s)
             + u * v.square() / (2.0 * sigma2 * t * omt * s.pow(3)))
        H = ((delta2 - u * u) / (2.0 * s.pow(4))
             - t / (sigma2 * omt * s)
             - u * v / (sigma2 * omt * s.pow(3))
             + (v.square() - 2.0 * t * u * v) / (2.0 * sigma2 * t * omt * s.pow(3))
             - 3.0 * u.square() * v.square() / (2.0 * sigma2 * t * omt * s.pow(5)))
        H_safe = torch.minimum(H, torch.full_like(H, -eps))
        step = g / H_safe
        cap = 0.5 * u.abs() + delta + 1.0
        step = torch.maximum(torch.minimum(step, cap), -cap)
        u = u - step

    s = torch.sqrt(u * u + delta2)
    v = z - t * u
    H = ((delta2 - u * u) / (2.0 * s.pow(4))
         - t / (sigma2 * omt * s)
         - u * v / (sigma2 * omt * s.pow(3))
         + (v.square() - 2.0 * t * u * v) / (2.0 * sigma2 * t * omt * s.pow(3))
         - 3.0 * u.square() * v.square() / (2.0 * sigma2 * t * omt * s.pow(5)))
    r_mode = torch.asinh(u / delta)
    H_r = H * s * s
    tau = torch.sqrt(kappa / (-H_r).clamp_min(eps))
    r = r_mode + tau * torch.randn_like(z)
    return delta * torch.sinh(r)


# ---- moments ----------------------------------------------------------------
def moments(x):
    mu = x.mean().item()
    sd = x.std().item()
    zc = (x - mu) / max(sd, 1e-12)
    skew = (zc ** 3).mean().item()
    kurt = (zc ** 4).mean().item() - 3
    return mu, sd, skew, kurt


def run(n_samples=100_000, device="cpu"):
    inf_grid = 512
    inf_range = 2000.0
    ref_grid = 16384

    header = (
        f"{'t':>4} {'σ':>4} {'λ':>5} {'z':>7} │"
        f"   {'mean_REF':>8} {'_grid':>8} {'_lapl':>8}"
        f" │ {'std_REF':>7} {'_grid':>6} {'_lapl':>6}"
        f" │ {'kur_REF':>7} {'_grid':>7} {'_lapl':>7}"
    )
    print(header)
    print("-" * len(header))

    torch.manual_seed(0)
    rows = []
    for t_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for sigma, lam in [(1.0, 1024.0), (1.0, 64.0), (0.5, 1024.0)]:
            for true_u in [0.0, 5.0, 50.0, 500.0]:
                sp_sq = sigma ** 2 * math.sqrt(4 * lam + true_u ** 2) * t_val * (1 - t_val)
                z_val = t_val * true_u  # typical z (mean of forward)

                # Reference grid: adaptive range
                ref_range = max(3.0 * (abs(z_val) / t_val + 5 * math.sqrt(sp_sq)), 100.0)
                u_ref = grid_sample_1d(z_val, t_val, sigma, lam,
                                       ref_grid, ref_range, n_samples)
                u_inf = grid_sample_1d(z_val, t_val, sigma, lam,
                                       inf_grid, inf_range, n_samples)
                # Laplace needs a (n_samples,) z tensor
                z_tensor = torch.full((n_samples,), z_val, dtype=torch.float64)
                u_lap = laplace_sample(z_tensor, t_val, sigma, lam)

                m_r, s_r, _, k_r = moments(u_ref)
                m_g, s_g, _, k_g = moments(u_inf)
                m_l, s_l, _, k_l = moments(u_lap)

                rows.append((t_val, sigma, lam, true_u, z_val,
                             m_r, s_r, k_r, m_g, s_g, k_g, m_l, s_l, k_l))
                print(
                    f"{t_val:>4.2f} {sigma:>4.1f} {lam:>5.0f} {z_val:>7.1f} │ "
                    f"  {m_r:>8.2f} {m_g:>8.2f} {m_l:>8.2f} │ "
                    f"{s_r:>7.2f} {s_g:>6.2f} {s_l:>6.2f} │ "
                    f"{k_r:>7.3f} {k_g:>7.3f} {k_l:>7.3f}"
                )

    # Aggregate error summary
    import statistics
    def rel(a, b):
        return abs(a - b) / max(abs(b), 1e-9)

    grid_mean_err = [rel(r[8], r[5]) for r in rows]
    lap_mean_err = [rel(r[11], r[5]) for r in rows]
    grid_std_err = [rel(r[9], r[6]) for r in rows]
    lap_std_err = [rel(r[12], r[6]) for r in rows]

    print()
    print(f"median |mean rel error|: grid={statistics.median(grid_mean_err):.4f}"
          f"  laplace={statistics.median(lap_mean_err):.4f}")
    print(f"median |std  rel error|: grid={statistics.median(grid_std_err):.4f}"
          f"  laplace={statistics.median(lap_std_err):.4f}")
    print(f"max    |std  rel error|: grid={max(grid_std_err):.4f}"
          f"  laplace={max(lap_std_err):.4f}")


if __name__ == "__main__":
    run()
