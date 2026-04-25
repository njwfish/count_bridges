"""
Generalised inverse-Gaussian (GIG) and inverse-Gaussian (IG) primitives for
the inverse-Gaussian-clocked Gaussian bridge.

Parametrisation (matches adaptive_gaussian_bridge.tex):
    GIG(p, a, b):  f(c) ∝ c^{p-1} exp(-(a c + b/c) / 2),  c > 0,  a, b > 0.

Special cases used by the IG clock:
    GIG(-1/2, a, b)  =  IG(mu = sqrt(b/a), lam = b)        [exact sampler]
    GIG( 1/2, a, b)  : 1/X ~ GIG(-1/2, b, a)               [reciprocal IG]
    GIG(-1,   a, b)  :  C_t posterior given y              [generic GIG]

IG sampler is Michael-Schucany-Haas (1976): exact, one normal + one uniform.

Generic GIG(p, a, b) sampler is mode-centered ratio-of-uniforms in x = log c.
The log-density on x is g(x) = p x - (a e^x + b e^{-x}) / 2, which is strictly
log-concave for any (p, a, b) (Hessian is -(a e^x + b e^{-x})/2 < 0). RoU on
this is robust: a fixed-budget vectorised rejection loop with MAP fallback on
the residual stragglers (P(reach fallback) ~ 1e-7 at n_rounds=12).

The transition density p_t(y) = sigma B_{C_t} marginalised over the IG clock,
its log, and the local diffusion coefficient D_IG have closed forms in terms
of K_0, K_1 and the scaled complementary error function.
"""

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch


__all__ = [
    "sample_ig",
    "sample_gig_pmh",       # GIG(p=±1/2) closed-form via IG
    "sample_gig",            # generic GIG(p, a, b) via RoU
    "sample_clock_split",
    "gig_log_density",
    "gig_mode",
    "log_pt",
    "diffusion_coeff_ig",
    "Lambda_erfcx",
]


def _broadcast(*tensors):
    """Broadcast tensors / scalars to a common shape and floating dtype."""
    arrs = [torch.as_tensor(t) for t in tensors]
    out_dtype = arrs[0].dtype
    for a in arrs[1:]:
        out_dtype = torch.promote_types(out_dtype, a.dtype)
    if not out_dtype.is_floating_point:
        out_dtype = torch.float32
    arrs = [a.to(out_dtype) for a in arrs]
    return torch.broadcast_tensors(*arrs)


# ----------------------------------------------------------------------------
#  Inverse-Gaussian sampler  (Michael, Schucany, Haas 1976)
# ----------------------------------------------------------------------------
def sample_ig(mu: torch.Tensor, lam: torch.Tensor,
              generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Exact IG(mu, lam) sampler.

    Density:
        f(x; mu, lam) = sqrt(lam / (2 pi x^3)) exp(-lam (x-mu)^2 / (2 mu^2 x))

    Equivalent to GIG(-1/2, lam/mu^2, lam).

    Vectorised; mu, lam may be any broadcastable shapes. Returns a tensor of
    the broadcast shape.
    """
    mu, lam = _broadcast(mu, lam)
    device, dtype = mu.device, mu.dtype
    n = torch.randn(mu.shape, device=device, dtype=dtype, generator=generator)
    y = n * n                                                       # chi-square_1
    # x_+ root of the quadratic from the Michael-Schucany-Haas transformation.
    inside = (4.0 * mu * lam * y + (mu * y) ** 2).clamp_min(0.0)
    x_p = (mu
           + 0.5 * (mu * mu * y) / lam.clamp_min(1e-30)
           - 0.5 * mu / lam.clamp_min(1e-30) * torch.sqrt(inside))
    u = torch.rand(mu.shape, device=device, dtype=dtype, generator=generator)
    accept_small = u <= mu / (mu + x_p.clamp_min(1e-30))
    return torch.where(accept_small, x_p, (mu * mu) / x_p.clamp_min(1e-30))


def sample_gig_pmh(p_sign: int, a: torch.Tensor, b: torch.Tensor,
                   generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Closed-form sampler for GIG(p=+/-1/2, a, b).

    p_sign = -1 -> GIG(-1/2, a, b) = IG(mu=sqrt(b/a), lam=b).
    p_sign = +1 -> GIG(+1/2, a, b): 1/X ~ GIG(-1/2, b, a) = IG(sqrt(a/b), a).
    """
    if p_sign not in (-1, +1):
        raise ValueError(f"p_sign must be -1 or +1, got {p_sign}")
    a, b = _broadcast(a, b)
    if p_sign == -1:
        mu = torch.sqrt(b / a.clamp_min(1e-30))
        return sample_ig(mu, b, generator=generator)
    mu = torch.sqrt(a / b.clamp_min(1e-30))
    y = sample_ig(mu, a, generator=generator)
    return 1.0 / y.clamp_min(1e-30)


# ----------------------------------------------------------------------------
#  Generic GIG(p, a, b) sampler via mode-centered ratio-of-uniforms in log c
# ----------------------------------------------------------------------------
def gig_log_density(c: torch.Tensor, p: Union[float, torch.Tensor],
                    a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """log f(c) for GIG(p, a, b) up to an additive constant.

    f(c) ∝ c^{p-1} exp(-(a c + b/c) / 2).
    """
    return ((p - 1.0) * torch.log(c.clamp_min(1e-300))
            - 0.5 * (a * c + b / c.clamp_min(1e-300)))


def _logg_x(x: torch.Tensor, p: Union[float, torch.Tensor],
            a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """log density on x = log c, up to an additive constant.

    g(x) = p x - (a e^x + b e^{-x}) / 2.

    Log-sum-exp on (log a + x, log b - x) avoids overflow when |x| is large.
    """
    log_a = torch.log(a.clamp_min(1e-300))
    log_b = torch.log(b.clamp_min(1e-300))
    e1 = log_a + x
    e2 = log_b - x
    m = torch.maximum(e1, e2)
    sum_term = m + torch.log(torch.exp(e1 - m) + torch.exp(e2 - m))
    return p * x - 0.5 * torch.exp(sum_term)


def _grad_hess_x(x: torch.Tensor, p: Union[float, torch.Tensor],
                 a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gradient and Hessian of log g(x).

        g'(x)  = p - (a e^x - b e^{-x}) / 2
        g''(x) = -(a e^x + b e^{-x}) / 2  < 0  (strictly log-concave)
    """
    aex = a * torch.exp(x)
    bemx = b * torch.exp(-x)
    g = p - 0.5 * (aex - bemx)
    h = -0.5 * (aex + bemx)
    return g, h


def gig_mode(p: Union[float, torch.Tensor], a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mode of GIG(p, a, b) on c.

    Solves a c^2 - 2(p-1) c - b = 0 (positive root):
        c* = ((p-1) + sqrt((p-1)^2 + a b)) / a.
    """
    pm1 = (p - 1.0) if isinstance(p, float) else (p - 1.0)
    return (pm1 + torch.sqrt(pm1 * pm1 + a * b)) / a.clamp_min(1e-30)


def _logg_x_mode(p: Union[float, torch.Tensor], a: torch.Tensor, b: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mode and curvature of log g(x).

    Returns (x_mode, log_g_mode, sigma_x) where:
        x_mode      = 0.5 log(b/a) + asinh(p / sqrt(a b))      [stable form]
        log_g_mode  = p x_mode - (a e^x + b e^{-x})/2
        sigma_x     = (p^2 + a b)^{-1/4}                       [Laplace stddev]

    At the mode, a e^x + b e^{-x} = 2 sqrt(p^2 + a b), so
    g''(x*) = -sqrt(p^2 + a b) and the Laplace stddev is as stated.
    """
    omega = torch.sqrt(a * b)
    if isinstance(p, float):
        p_t = torch.full_like(a, float(p))
    else:
        p_t = p.to(dtype=a.dtype, device=a.device).expand_as(a)
    x_mode = (0.5 * (torch.log(b.clamp_min(1e-300)) - torch.log(a.clamp_min(1e-300)))
              + torch.asinh(p_t / omega.clamp_min(1e-30)))
    curv = torch.sqrt(p_t * p_t + a * b)
    sigma_x = 1.0 / curv.clamp_min(1e-30).sqrt()
    log_g_mode = _logg_x(x_mode, p, a, b)
    return x_mode, log_g_mode, sigma_x


def sample_gig(p: Union[float, torch.Tensor], a: torch.Tensor, b: torch.Tensor,
               n_rounds: int = 12, safety: float = 1.05,
               side_iters: int = 4,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Sample from GIG(p, a, b) via mode-centered ratio-of-uniforms in x = log c.

    Returns a tensor of the broadcast shape of (a, b). p may be a scalar or
    any tensor broadcastable to that shape.

    For p in {-1/2, +1/2}, the closed-form `sample_gig_pmh(...)` is faster
    and exact. This generic path is for p outside those values (e.g. p = -1
    for the C_t posterior in the IG clock).
    """
    a, b = _broadcast(a, b)
    device, dtype = a.device, a.dtype
    if isinstance(p, float):
        p_t = torch.full_like(a, float(p))
    else:
        p_t = p.to(dtype=dtype, device=device).expand_as(a)

    x_mode, log_g_mode, sigma_x = _logg_x_mode(p_t, a, b)

    # RoU box around x_mode in x = log c.
    # u_max = sup_{x>0} x sqrt(g(m+x) / g(m)),
    # u_min = -sup_{x>0} x sqrt(g(m-x) / g(m)),
    # v_max = 1.
    # For Gaussian, u_max = sqrt(2/e) * sigma_x. We Newton-solve
    #   d/dx [x^2 g(m+x)/g(m)] = 0  =>  2/x + g'(m+x) = 0  (right side)
    #   d/dx [x^2 g(m-x)/g(m)] = 0  =>  2/x - g'(m-x) = 0  (left side)
    # from the Gaussian guess.
    init_w = math.sqrt(2.0 / math.e) * sigma_x
    xr = init_w.clone()
    xl = init_w.clone()
    for _ in range(side_iters):
        gr, hr = _grad_hess_x(x_mode + xr, p_t, a, b)
        f = 2.0 / xr.clamp_min(1e-30) + gr
        fp = -2.0 / xr.clamp_min(1e-30) ** 2 + hr
        step = torch.where(fp < 0.0, f / fp, torch.zeros_like(f))
        step = torch.clamp(step, -0.5 * xr, 0.5 * xr)
        xr = (xr - step).clamp_min(1e-8)

        gl, hl = _grad_hess_x(x_mode - xl, p_t, a, b)
        f = 2.0 / xl.clamp_min(1e-30) - gl
        fp = -2.0 / xl.clamp_min(1e-30) ** 2 + hl
        step = torch.where(fp < 0.0, f / fp, torch.zeros_like(f))
        step = torch.clamp(step, -0.5 * xl, 0.5 * xl)
        xl = (xl - step).clamp_min(1e-8)

    u_max = (xr * torch.exp(0.5 * (_logg_x(x_mode + xr, p_t, a, b) - log_g_mode))
             * safety)
    u_min = -(xl * torch.exp(0.5 * (_logg_x(x_mode - xl, p_t, a, b) - log_g_mode))
              * safety)

    shape = a.shape
    N = a.numel()
    fa = a.reshape(-1)
    fb = b.reshape(-1)
    fp = p_t.reshape(-1)
    fm = x_mode.reshape(-1)
    flo = u_min.reshape(-1)
    fhi = u_max.reshape(-1)
    flogg_m = log_g_mode.reshape(-1)

    done = torch.zeros(N, dtype=torch.bool, device=device)
    out_x = fm.clone()                                              # MAP fallback
    tiny = torch.finfo(dtype).tiny
    for _ in range(n_rounds):
        U = flo + (fhi - flo) * torch.rand(N, device=device, dtype=dtype, generator=generator)
        V = torch.rand(N, device=device, dtype=dtype, generator=generator).clamp_min(tiny)
        X = fm + U / V
        logg = _logg_x(X, fp, fa, fb)
        accept = (2.0 * torch.log(V) <= (logg - flogg_m)) & torch.isfinite(X)
        newly = (~done) & accept
        out_x = torch.where(newly, X, out_x)
        done = done | newly
        if bool(done.all()):
            break
    return torch.exp(out_x).reshape(shape)


# ----------------------------------------------------------------------------
#  Inverse-Gaussian-clock split sampler
# ----------------------------------------------------------------------------
def sample_clock_split(c_t: torch.Tensor, s: float, t: float, eta: float,
                       generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Sample R = C_s / C_t given C_t = c_t for the IG(eta, gamma) clock.

    Conditional on C_t = c, the displacement W = R/(1-R) = C_s / (C_t - C_s)
    is the *probability* mixture
        W ~ (s/t) GIG(+1/2, a, b)  +  ((t-s)/t) GIG(-1/2, a, b),
        a = eta^2 (t - s)^2 / c,
        b = eta^2 s^2 / c.

    The mixture weights are eta-free (s/t and (t-s)/t). The within-branch
    parameters carry eta via lam = eta^2 s^2 / c (IG branch) or lam = eta^2
    (t-s)^2 / c (reciprocal branch). Gamma drops out entirely (it is the
    ambient drift parameter for C_t and is conditioned away by fixing C_t).

    Args:
        c_t:  any tensor shape; c_t > 0.
        s, t: scalar Brownian times with 0 < s < t.
        eta:  IG-clock rate parameter.
    Returns:
        R of the same shape as c_t, with values in (0, 1).
    """
    if not (0.0 < s < t):
        raise ValueError(f"need 0 < s < t, got s={s}, t={t}")
    device, dtype = c_t.device, c_t.dtype
    eta2 = float(eta) ** 2
    a = eta2 * (t - s) ** 2 / c_t.clamp_min(1e-30)
    b = eta2 * s ** 2 / c_t.clamp_min(1e-30)

    pw = float(s) / float(t)
    pick_pos = torch.rand(c_t.shape, device=device, dtype=dtype, generator=generator) < pw
    w_pos = sample_gig_pmh(+1, a, b, generator=generator)           # GIG(+1/2): reciprocal IG
    w_neg = sample_gig_pmh(-1, a, b, generator=generator)           # GIG(-1/2): IG
    w = torch.where(pick_pos, w_pos, w_neg)
    r = w / (1.0 + w)
    return r.clamp(min=1e-8, max=1.0 - 1e-8)


# ----------------------------------------------------------------------------
#  IG-clock transition density and local diffusion coefficient
# ----------------------------------------------------------------------------
def _log_modified_bessel_k(nu: int, x: torch.Tensor) -> torch.Tensor:
    """log K_nu(x) for nu in {0, 1}, vectorised, stable for large x.

    For x large, K_nu underflows to 0 in float32; switch to the asymptotic
        log K_nu(x) = -x + 0.5 (log pi - log 2 - log x) + O(1/x).
    """
    if nu == 0:
        kn = torch.special.modified_bessel_k0(x)
    elif nu == 1:
        kn = torch.special.modified_bessel_k1(x)
    else:
        raise ValueError("only nu in {0, 1} supported here")
    asym = -x + 0.5 * (math.log(math.pi) - math.log(2.0) - torch.log(x.clamp_min(1e-30)))
    tiny = torch.finfo(x.dtype).tiny
    return torch.where(kn > tiny, torch.log(kn.clamp_min(tiny)), asym)


def log_pt(y: torch.Tensor, t: Union[float, torch.Tensor],
           sigma: float, gamma: float, eta: float) -> torch.Tensor:
    """log p_t(y), the IG-clocked transition density per coordinate.

    From the note (§"Gaussian-mixture oracle"):
        p_t(y) = (eta gamma t)/(pi sigma r_t(y)) e^{eta gamma t} K_1(gamma r_t(y)),
        r_t(y) = sqrt(eta^2 t^2 + y^2 / sigma^2).
    """
    t_t = torch.as_tensor(t, dtype=y.dtype, device=y.device)
    r = torch.sqrt(eta * eta * t_t * t_t + (y * y) / (sigma * sigma))
    log_K1 = _log_modified_bessel_k(1, gamma * r)
    return (math.log(eta) + math.log(gamma) + torch.log(t_t.clamp_min(1e-30))
            - math.log(math.pi) - math.log(sigma)
            - torch.log(r.clamp_min(1e-30))
            + eta * gamma * t_t
            + log_K1)


def Lambda_erfcx(rho: torch.Tensor) -> torch.Tensor:
    """Lambda(rho) = sqrt(pi) rho exp(rho^2) erfc(rho), stable for any rho >= 0.

    Uses torch.special.erfcx(rho) = exp(rho^2) erfc(rho), which is bounded
    on [0, inf) and approaches 1/(sqrt(pi) rho) as rho -> inf.
    """
    return math.sqrt(math.pi) * rho * torch.special.erfcx(rho)


def diffusion_coeff_ig(y: torch.Tensor, t: Union[float, torch.Tensor],
                       sigma: float, gamma: float, eta: float,
                       n_quad: int = 32, half_width: float = 6.0
                       ) -> torch.Tensor:
    """Per-coord local diffusion coefficient D_IG(y, t) for the IG clock.

    Per the note's §"Generator and MSE reverse sampler":
        D_IG(y, t) = (1/t) E_{C_t | y} [ (1 - L(rho)) y^2 + sigma^2 C_t L(rho) ],
        rho(c, t) = eta t / sqrt(2 c),
        L(rho) = sqrt(pi) rho exp(rho^2) erfc(rho).

    The expectation is over the GIG posterior
        C_t | y ~ GIG(-1, gamma^2, eta^2 t^2 + y^2 / sigma^2),
    computed by Gauss-Legendre quadrature on x = log c, centered on the GIG
    mode with half-window of `half_width * sigma_x` Laplace stddevs.

    Returns a tensor with the same shape as y.
    """
    t_t = torch.as_tensor(t, dtype=y.dtype, device=y.device)
    a = torch.full_like(y, float(gamma) ** 2)
    b = (float(eta) ** 2) * t_t * t_t + (y * y) / (float(sigma) ** 2)

    p = -1.0
    x_mode, log_g_mode, sigma_x = _logg_x_mode(p, a, b)
    n_np, w_np = np.polynomial.legendre.leggauss(int(n_quad))
    n = torch.as_tensor(n_np, dtype=y.dtype, device=y.device)
    w = torch.as_tensor(w_np, dtype=y.dtype, device=y.device)

    h = half_width * sigma_x
    n_b = n.view(*([1] * y.ndim), -1)
    w_b = w.view(*([1] * y.ndim), -1)
    x = x_mode.unsqueeze(-1) + h.unsqueeze(-1) * n_b               # (..., G)

    logf = _logg_x(x, p, a.unsqueeze(-1), b.unsqueeze(-1)) - log_g_mode.unsqueeze(-1)
    log_qweight = torch.log(w_b.clamp_min(1e-300))
    log_w = logf + log_qweight                                      # unnormalised log weights

    c = torch.exp(x)
    rho = float(eta) * t_t.unsqueeze(-1) / torch.sqrt(2.0 * c.clamp_min(1e-30))
    Lam = Lambda_erfcx(rho)
    integrand = (1.0 - Lam) * (y * y).unsqueeze(-1) + (float(sigma) ** 2) * c * Lam

    m_lse = log_w.max(dim=-1, keepdim=True).values
    ww = torch.exp(log_w - m_lse)
    num = (ww * integrand).sum(dim=-1)
    den = ww.sum(dim=-1).clamp_min(1e-30)
    return (num / den) / t_t.clamp_min(1e-30)
