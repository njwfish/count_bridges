"""RoU exact sampler (user-proposed). Stored as a module so other tests can
import it."""

import math
import torch

LOG2 = math.log(2.0)


def _logq_y_stable(y, A, B, C):
    ay = y.abs()
    em2 = torch.exp(-2.0 * ay)
    th = torch.sign(y) * (1.0 - em2) / (1.0 + em2)
    se = 2.0 * torch.exp(-ay) / (1.0 + em2)
    logch = ay - LOG2 + torch.log1p(em2)
    log_Bch = torch.log(B) + logch
    neg_Bch = -torch.exp(torch.clamp(log_Bch, max=80.0))
    lp = 0.5 * logch + A * th + C * se + neg_Bch
    lp = torch.where(log_Bch > 80.0, torch.full_like(lp, -torch.inf), lp)
    return lp


def _grad_hess(y, A, B, C):
    ch = torch.cosh(y)
    th = torch.tanh(y)
    se = 1.0 / ch
    sh = th * ch
    se2 = se * se
    g = 0.5 * th + A * se2 - B * sh - C * se * th
    h = 0.5 * se2 - 2.0 * A * se2 * th - B * ch + C * se * (2.0 * th * th - 1.0)
    return g, h


@torch.no_grad()
def sample_u_conditional_rou(z_t, t, sigma, lam, generator=None,
                             safety=1.001, max_rounds=200):
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

    # Mode solve in y
    m = torch.asinh(z_t / (2.0 * t * sqrt_lam))
    for _ in range(6):
        g, h = _grad_hess(m, A, B, C)
        step = torch.where(h < 0.0, g / h, torch.zeros_like(g))
        step = torch.clamp(step, -1.0, 1.0)
        m = m - step

    _, h_mode = _grad_hess(m, A, B, C)
    vloc = torch.clamp(-1.0 / h_mode, min=torch.finfo(dtype).tiny)
    xr = torch.sqrt(2.0 * vloc)
    xl = xr.clone()

    for _ in range(5):
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

    out = torch.empty_like(z_t)
    flat_out = out.reshape(-1)
    flat_m = m.reshape(-1)
    flat_A = A.reshape(-1)
    flat_B = B.reshape(-1)
    flat_C = C.reshape(-1)
    flat_vmax = vmax.reshape(-1)
    flat_umax = umax.reshape(-1)
    flat_umin = umin.reshape(-1)
    flat_sqrt_lam = sqrt_lam.reshape(-1)

    active = torch.arange(flat_out.numel(), device=device)

    rounds = 0
    total_proposals = 0
    while active.numel() > 0:
        rounds += 1
        if rounds > max_rounds:
            raise RuntimeError(
                f"RoU sampler exceeded max_rounds={max_rounds}, still {active.numel()} active"
            )
        n = active.numel()
        total_proposals += n
        lo = flat_umin[active]
        hi = flat_umax[active]
        U = lo + (hi - lo) * torch.rand(n, device=device, dtype=dtype, generator=generator)
        V = flat_vmax[active] * torch.rand(n, device=device, dtype=dtype, generator=generator)
        X = U / V
        Y = flat_m[active] + X
        accept = 2.0 * torch.log(V) <= _logq_y_stable(Y, flat_A[active], flat_B[active], flat_C[active])
        if accept.any():
            flat_out[active[accept]] = 2.0 * flat_sqrt_lam[active[accept]] * torch.sinh(Y[accept])
        active = active[~accept]

    return out, rounds, total_proposals
