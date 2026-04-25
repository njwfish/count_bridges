"""
Inverse-Gaussian-clocked Gaussian bridge (rigorous version).

See cell_types/generative/counting_flows/adaptive_gaussian_bridge.tex for the
mathematical setup. This module supersedes the heuristic
`AdaptiveGaussianBridge` (which plugs the deterministic per-pair clock
r(u) = sqrt(4 lam + u^2) into the bridge variance) with the rigorous
construction in which the clock is an internal Markov variable, sampled
from its posterior given the observed local displacement y_t = X_t - X_0.

Forward corruption (per-coord, axis-aligned):
    Given x_0, x_1, set u = x_1 - x_0.
    1. Sample C_1 ~ GIG(-1, gamma^2, nu^2 + u^2 / sigma^2).
    2. Split C_t = R * C_1 with R ~ p(R | C_1, t, 1) -- the IG clock split.
    3. X_t = x_0 + R u + sigma sqrt(C_1 R (1-R)) eps.

Reverse step (mode='bridge'):
    Given x_0_pred (posterior draw of X_0 | X_t), set y_t = x_t - x_0_pred.
    1. Sample C_t ~ GIG(-1, gamma^2, nu^2 t^2 + y_t^2 / sigma^2).
    2. Split C_s = R * C_t with R ~ p(R | C_t, s, t) (s = t_next < t = t_curr).
    3. X_s = x_0_pred + R y_t + sigma sqrt(C_t R (1-R)) xi.
    Exact finite-step bridge conditional on x_0_pred. Correct for
    distributional models.

Reverse step (mode='sde'):
    Drift (m^star - x)/t  -- exact for the entire clocked family.
    Diffusion sqrt(D_IG(y, t) * dt) with the moment-matched closed form
    from the note. Correct (to leading O(dt)) for mean-regression models.

Hyperparameters:
    sigma:  per-coord Brownian scale (data-scale units).
    gamma:  IG-clock relaxation parameter, > 0.
    nu:     IG-clock rate parameter (the note's eta), > 0. We use `nu` here
            to avoid clashing with the codebase-wide `eta` knob below, which
            is the noise *multiplier* on every bridge.
            E[C_t] = (nu * t) / gamma; the deterministic-clock limit is
            nu -> infty with nu/gamma fixed.

    mode:   'bridge' (exact finite-step IG bridge given x_0_pred; correct
            for distributional / energy-score models) or 'sde'
            (moment-matched diffusion approximation; correct for MSE /
            mean-regression models). Sampler-only knob.
    eta:    noise multiplier in [0, 1]. eta=0 gives the deterministic
            drift step; eta=1 gives the full sampler in the chosen mode.
            Sampler-only knob; matches the API of GaussianBridge /
            AdaptiveGaussianBridge.
    gig_n_rounds, gig_safety: GIG(-1) RoU sampler knobs (mode='bridge').
    n_quad: Gauss-Legendre nodes for D_IG quadrature (mode='sde').

Calibration to the heuristic:
    The heuristic AdaptiveGaussianBridge uses a deterministic clock c =
    sqrt(4 lam + u^2). To match the same u=0 scale, take nu = 2 sqrt(lam) / 1
    with gamma = 1: r*(0) = nu / gamma = 2 sqrt(lam) (= sqrt(4 lam) = 64 for
    lam = 1024).
"""

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch

from .gig import (
    diffusion_coeff_ig,
    sample_clock_split,
    sample_gig,
    sample_gig_pmh,
    sample_ig,
)


def pad_t_like_x(t, x):
    if isinstance(t, (float, int)):
        return t
    return t.view(-1, *([1] * (x.dim() - 1)))


class ClockedGaussianBridge:
    """Inverse-Gaussian-clocked Gaussian bridge."""

    def __init__(self,
                 sigma: float = 1.0,
                 gamma: float = 1.0,
                 nu: float = 64.0,
                 device: int = 0,
                 homogeneous_time: bool = False,
                 t_eps: float = 1e-3,
                 mode: str = "bridge",
                 eta: float = 1.0,
                 gig_n_rounds: int = 12,
                 gig_safety: float = 1.05,
                 n_quad: int = 32):
        if mode not in ("bridge", "sde"):
            raise ValueError(f"mode must be 'bridge' or 'sde', got {mode!r}")
        if sigma <= 0 or gamma <= 0 or nu <= 0:
            raise ValueError(
                f"sigma, gamma, nu must all be > 0, got "
                f"sigma={sigma}, gamma={gamma}, nu={nu}")
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.nu = float(nu)
        self.device = device
        self.homogeneous_time = homogeneous_time
        self.t_eps = float(t_eps)
        self.mode = mode
        self.eta = float(eta)
        self.gig_n_rounds = int(gig_n_rounds)
        self.gig_safety = float(gig_safety)
        self.n_quad = int(n_quad)

    # ------------------------------------------------------------------
    #  Forward corruption  (training):  X_t | X_0, X_1
    # ------------------------------------------------------------------
    def __call__(self, x_0: torch.Tensor, x_1: torch.Tensor,
                 t: Union[float, torch.Tensor, None] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward sampler. Returns (t, x_t, x_0).

        Per-coord, conditional on (x_0, x_1):
            1. C_1 ~ GIG(-1, gamma^2, nu^2 + u^2 / sigma^2),  u = x_1 - x_0.
            2. R = C_t / C_1 ~ split(t, 1, nu).
            3. X_t = x_0 + R u + sigma sqrt(C_1 R (1-R)) eps.
        """
        batch_size = x_0.shape[0]
        x_0, x_1 = x_0.float(), x_1.float()

        if t is not None:
            if isinstance(t, float):
                t = torch.full((batch_size,), t, device=x_0.device)
            else:
                t = t.to(x_0.device).reshape(batch_size)
        else:
            if self.homogeneous_time:
                t = torch.rand(1, device=x_0.device).expand(batch_size).clone()
            else:
                t = torch.rand(batch_size, device=x_0.device)

        t_clip = t.clamp(self.t_eps, 1.0 - self.t_eps)

        u = x_1 - x_0                                                       # (B, d)
        a = torch.full_like(u, self.gamma ** 2)
        b = (self.nu ** 2) + (u * u) / (self.sigma ** 2)
        C_1 = sample_gig(-1.0, a, b, n_rounds=self.gig_n_rounds,
                         safety=self.gig_safety)                            # (B, d)

        R = self._split_batched(C_1, s=t_clip, t_total=torch.ones_like(t_clip))  # (B, d)

        x_t = x_0 + R * u + self.sigma * torch.sqrt(
            (C_1 * R * (1.0 - R)).clamp_min(0.0)
        ) * torch.randn_like(x_0)

        return t.unsqueeze(1), x_t.float(), x_0.float()

    # ------------------------------------------------------------------
    #  Reverse step:  X_s | X_t, X_0_pred,  s < t
    # ------------------------------------------------------------------
    def sample_step(self, t_curr, t_next, x_t, x_0_pred, **z):
        """Single reverse step.

        Drift (always): x + dt * (x_hat_0 - x) / t_curr.
        Noise:
          mode='bridge': sample c_t ~ posterior, R ~ split(t_next, t_curr),
            then x_s = x_0_pred + R y_t + sigma sqrt(c_t R(1-R)) xi.
          mode='sde'   : noise variance = D_IG(y, t_curr) * dt per coord.
        eta scales the noise; eta=0 gives the deterministic ODE-like step.
        """
        if isinstance(t_curr, float):
            t_curr = torch.tensor(t_curr, device=x_t.device)
        if isinstance(t_next, float):
            t_next = torch.tensor(t_next, device=x_t.device)

        if float(t_next) == 0.0:
            return x_0_pred, x_0_pred

        t_eff = t_curr.clamp(max=1.0 - self.t_eps)
        dt = t_curr - t_next

        if self.mode == "bridge":
            y_t = x_t - x_0_pred                                           # (B, d)
            a = torch.full_like(y_t, self.gamma ** 2)
            b = (self.nu ** 2) * (float(t_eff) ** 2) + (y_t * y_t) / (self.sigma ** 2)
            c_t = sample_gig(-1.0, a, b, n_rounds=self.gig_n_rounds,
                             safety=self.gig_safety)                       # (B, d)
            R = sample_clock_split(c_t, float(t_next), float(t_eff), eta=self.nu)
            mean = x_0_pred + R * y_t                                      # = R x + (1-R) x_0_pred
            noise_var = (self.sigma ** 2) * c_t * R * (1.0 - R)
            noise_std = self.eta * torch.sqrt(noise_var.clamp_min(0.0))
            x_next = mean + noise_std * torch.randn_like(x_t)
            return x_next, x_0_pred

        # mode == 'sde'
        mean = x_t + dt * (x_0_pred - x_t) / t_curr
        y_t = x_t - x_0_pred
        D = diffusion_coeff_ig(y_t, t_eff, self.sigma, self.gamma, self.nu,
                               n_quad=self.n_quad)
        noise_var = D * dt
        noise_std = self.eta * torch.sqrt(noise_var.clamp_min(0.0))
        x_next = mean + noise_std * torch.randn_like(x_t)
        return x_next, x_0_pred

    # ------------------------------------------------------------------
    #  Per-sample-time clock split (forward).
    # ------------------------------------------------------------------
    def _split_batched(self, C_total: torch.Tensor, s: torch.Tensor,
                       t_total: torch.Tensor) -> torch.Tensor:
        """Sample R = C_s / C_total per (B, d), with per-sample (s, t_total).

        Forward training has heterogeneous t_i across the batch. Vectorises
        the IG-clock split kernel along (B, d).
        """
        device, dtype = C_total.device, C_total.dtype
        view_shape = (-1,) + (1,) * (C_total.ndim - 1)
        s_b = s.reshape(view_shape).to(dtype)
        t_b = t_total.reshape(view_shape).to(dtype)
        ts_b = (t_b - s_b)
        nu2 = self.nu ** 2
        a = nu2 * ts_b * ts_b / C_total.clamp_min(1e-30)                   # (B, d)
        b = nu2 * s_b * s_b / C_total.clamp_min(1e-30)

        pw = (s_b / t_b).expand_as(C_total)
        u = torch.rand(C_total.shape, device=device, dtype=dtype)
        pick_pos = u < pw
        w_pos = sample_gig_pmh(+1, a, b)
        w_neg = sample_gig_pmh(-1, a, b)
        w = torch.where(pick_pos, w_pos, w_neg)
        r = w / (1.0 + w)
        return r.clamp(min=1e-8, max=1.0 - 1e-8)

    # ------------------------------------------------------------------
    #  Reverse sampler (multi-step).
    # ------------------------------------------------------------------
    def sampler(
        self,
        x_1: torch.Tensor,
        z: Dict[str, Any],
        model,
        return_trajectory: bool = False,
        return_x_hat: bool = False,
        n_steps: int = 10,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        b = x_1.shape[0]
        x_t = x_1.float()

        time_points = torch.linspace(0.0, 1.0 - self.t_eps, n_steps + 1).to(x_t.device)

        traj = [x_t]
        xhat_traj = []

        for k in range(n_steps, 0, -1):
            t_curr = time_points[k]
            t_next = time_points[k - 1]
            t = t_curr.expand(b, 1)

            with torch.no_grad():
                x_0_pred = model.sample(x_t=x_t, t=t, **z)
                x_t, x_0_pred = self.sample_step(t_curr, t_next, x_t, x_0_pred, **z)

            if return_trajectory:
                traj.append(x_t)
            if return_x_hat:
                xhat_traj.append(x_0_pred)

        outs = [x_t]
        if return_trajectory:
            outs.append(torch.stack(traj))
        if return_x_hat:
            outs.append(torch.stack(xhat_traj))

        return tuple(outs) if len(outs) > 1 else x_t

    def __repr__(self):
        return (f"ClockedGaussianBridge(sigma={self.sigma}, gamma={self.gamma}, "
                f"nu={self.nu}, mode={self.mode}, eta={self.eta})")
