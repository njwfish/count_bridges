"""Trimmed RoU sampler with fixed-round rejection + Laplace fallback."""

import math
import torch
from tests.rou_sampler import _logq_y_stable, _grad_hess
from tests.posterior_samplers_compare import laplace_sample


@torch.no_grad()
def sample_u_rou_fast(
    z_t, t, sigma, lam,
    mode_iters: int = 2,
    side_iters: int = 2,
    fixed_rounds: int = 5,
    safety: float = 1.01,
):
    """Faster RoU sampler.

    Changes vs the full version:
      - 2 Newton iters for mode (was 6) — Newton converges in ~2 from asinh init.
      - 2 Newton iters per side (was 5).
      - Fixed number of rejection rounds (vectorized, no index-shuffling).
      - Stragglers after `fixed_rounds` fall back to Laplace (rare at 73% accept).
    """
    z_t, t, sigma, lam = torch.broadcast_tensors(
        *(torch.as_tensor(x, dtype=z_t.dtype if isinstance(z_t, torch.Tensor) else torch.float64)
          for x in (z_t, t, sigma, lam))
    )
    dtype = z_t.dtype
    device = z_t.device

    sqrt_lam = torch.sqrt(lam)
    sig2 = sigma * sigma
    one_minus_t = 1.0 - t

    A = z_t / (sig2 * one_minus_t)
    B = t * sqrt_lam / (sig2 * one_minus_t)
    C = (t * sqrt_lam - z_t * z_t / (4.0 * t * sqrt_lam)) / (sig2 * one_minus_t)

    # Mode solve (2 iters typically sufficient)
    m = torch.asinh(z_t / (2.0 * t * sqrt_lam))
    for _ in range(mode_iters):
        g, h = _grad_hess(m, A, B, C)
        step = torch.where(h < 0.0, g / h, torch.zeros_like(g))
        step = torch.clamp(step, -1.0, 1.0)
        m = m - step

    _, h_mode = _grad_hess(m, A, B, C)
    vloc = torch.clamp(-1.0 / h_mode, min=torch.finfo(dtype).tiny)
    xr = torch.sqrt(2.0 * vloc)
    xl = xr.clone()

    for _ in range(side_iters):
        g, h = _grad_hess(m + xr, A, B, C)
        g = 1.0 / xr + 0.5 * g
        h = -1.0 / (xr * xr) + 0.5 * h
        step = torch.where(h < 0.0, g / h, torch.zeros_like(g))
        step = torch.clamp(step, -0.5 * xr, 0.5 * xr)
        xr = torch.clamp(xr - step, min=1e-8)

        g, h = _grad_hess(m - xl, A, B, C)
        g = 1.0 / xl - 0.5 * g
        h = -1.0 / (xl * xl) + 0.5 * h
        step = torch.where(h < 0.0, g / h, torch.zeros_like(g))
        step = torch.clamp(step, -0.5 * xl, 0.5 * xl)
        xl = torch.clamp(xl - step, min=1e-8)

    vmax = torch.exp(0.5 * _logq_y_stable(m, A, B, C)) * safety
    umax = xr * torch.exp(0.5 * _logq_y_stable(m + xr, A, B, C)) * safety
    umin = -xl * torch.exp(0.5 * _logq_y_stable(m - xl, A, B, C)) * safety

    # Fixed-rounds rejection: generate all N proposals per round, mask accepts.
    # After a few rounds, most samples have accepted; stragglers go to Laplace.
    N = z_t.numel()
    out = torch.empty_like(z_t).reshape(-1)
    flat_m = m.reshape(-1)
    flat_A = A.reshape(-1)
    flat_B = B.reshape(-1)
    flat_C = C.reshape(-1)
    flat_lo = umin.reshape(-1)
    flat_hi = umax.reshape(-1)
    flat_vmax = vmax.reshape(-1)
    flat_sqrt_lam = sqrt_lam.reshape(-1)
    done = torch.zeros(N, dtype=torch.bool, device=device)

    for _ in range(fixed_rounds):
        U = flat_lo + (flat_hi - flat_lo) * torch.rand(N, device=device, dtype=dtype)
        V = flat_vmax * torch.rand(N, device=device, dtype=dtype)
        Y = flat_m + U / V
        newly = (~done) & (2.0 * torch.log(V) <= _logq_y_stable(Y, flat_A, flat_B, flat_C))
        out = torch.where(newly, 2.0 * flat_sqrt_lam * torch.sinh(Y), out)
        done = done | newly
        if done.all():
            break

    # Stragglers → Laplace fallback
    if not done.all():
        z_flat = z_t.reshape(-1)
        t_flat = t.reshape(-1)
        fallback = laplace_sample(z_flat, t_flat[0].item() if torch.all(t_flat == t_flat[0]) else t_flat,
                                  sigma.reshape(-1)[0].item(), lam.reshape(-1)[0].item(),
                                  kappa=1.0, newton_steps=4)
        out = torch.where(done, out, fallback)

    return out.reshape(z_t.shape)
