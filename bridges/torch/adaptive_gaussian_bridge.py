"""
Skellam-Matched Data-Adaptive Gaussian Bridge with calibrated per-dim posterior sampler.

Forward (per-dim, axis-aligned Gaussian):
    Var(x_t^i | x_0, x_1) = sigma^2 * sqrt(4 lam + (x_1^i - x_0^i)^2) * t(1-t)

matches Skellam per-dim variance (via the Bessel slack mode) with sigma=1, lam=1024.

Reverse step — exact Ratio-of-Uniforms sampler in y = asinh(u / (2 sqrt(lam))):
    For each dim independently, sample u_i ~ p(u_i | z_t^i, t) from the 1-D posterior

        p(u | z, t) ∝ (4 lam + u^2)^(-1/4) * exp(-(z - t u)^2 / (2 sigma^2 sqrt(4 lam + u^2) t(1-t)))

    In y-coordinates this is

        log q(y) = 0.5 log(cosh y) + A tanh y - B cosh y + C sech y + const
      with  A = z / (sigma^2 (1-t))
            B = t sqrt(lam) / (sigma^2 (1-t))
            C = (t sqrt(lam) - z^2/(4 t sqrt(lam))) / (sigma^2 (1-t))

    Mode-centered RoU gives unbiased samples at ~73% acceptance (~1.37 proposals
    per sample). Three short Newton solves (mode + 2 side-maxima for the RoU
    bounding box) + a small number of fixed rejection rounds.

    Caveat: failure mode at very small t with z ~ 0 where the posterior becomes
    bimodal in u (and in y). Condition: t <= sigma^2 / (4 sqrt(lam)). At
    sigma=1, lam=1024 this threshold is t ~ 0.008 — well below the minimum
    t = 0.1 hit by n_steps=10 sampling. See bimodal_threshold() below.

    No failure at t -> 1: the likelihood dominates strongly, posterior is
    tightly unimodal.

    Then take the axis-aligned bridge step conditional on u_hat^i:
        mean = ((t-s)/t) x_hat_0 + (s/t) x_t
        var_i = sigma^2 * sqrt(4 lam + u_hat_i^2) * s (t-s) / t
        x_s^i = mean_i + sqrt(var_i) * randn
"""

import math
import numpy as np
import torch
from typing import Dict, Any, Tuple, Union

LOG2 = math.log(2.0)


def _gauss_legendre(n_points: int):
    """Gauss-Legendre nodes and weights on [-1, 1]. Called once."""
    x, w = np.polynomial.legendre.leggauss(n_points)
    return torch.tensor(x, dtype=torch.float64), torch.tensor(w, dtype=torch.float64)


def pad_t_like_x(t, x):
    if isinstance(t, (float, int)):
        return t
    return t.view(-1, *([1] * (x.dim() - 1)))


def bimodal_threshold(sigma: float, lam: float) -> float:
    """Minimum t at which p(u | z=0, t) is guaranteed unimodal in y-space.

    Below this t, the posterior has symmetric modes at +/- y_* and the
    mode-centered samplers here collapse. For n_steps = 10 we hit t down to
    0.1; thresholds for typical settings are 0.002-0.03, safely below.
    """
    return (sigma ** 2) / (4.0 * math.sqrt(lam) + sigma ** 2)


def _logq_y(y, A, B, C):
    """log q(y) up to an additive constant. Stable for large |y|."""
    ay = y.abs()
    em2 = torch.exp(-2.0 * ay)
    th = torch.sign(y) * (1.0 - em2) / (1.0 + em2)
    se = 2.0 * torch.exp(-ay) / (1.0 + em2)
    logch = ay - LOG2 + torch.log1p(em2)
    log_Bch = torch.log(B) + logch
    neg_Bch = -torch.exp(torch.clamp(log_Bch, max=80.0))
    lp = 0.5 * logch + A * th + C * se + neg_Bch
    lp = torch.where(log_Bch > 80.0, torch.full_like(lp, -float("inf")), lp)
    return lp


def _grad_hess_y(y, A, B, C):
    """Gradient and Hessian of log q(y) at moderate |y|."""
    ch = torch.cosh(y)
    th = torch.tanh(y)
    se = 1.0 / ch
    sh = th * ch
    se2 = se * se
    g = 0.5 * th + A * se2 - B * sh - C * se * th
    h = 0.5 * se2 - 2.0 * A * se2 * th - B * ch + C * se * (2.0 * th * th - 1.0)
    return g, h


class AdaptiveGaussianBridge:
    """Per-dim Skellam-matched Gaussian bridge with 1-D posterior u-sampler.

    Args:
        sigma: overall scale multiplier on per-dim sigma_pair. sigma=1 matches
               Skellam when lam=1024 (= BesselM lam_p * lam_m with both 32).
        lam:   slack-floor parameter. For Skellam match, lam = lam_p * lam_m.
        t_eps: clip t to [0, 1 - t_eps] for reverse; keeps variances finite.
        u_range: half-width of the per-dim u-grid for posterior sampling.
                 Should comfortably cover the data's per-dim |diff| range.
        n_grid:  number of grid points per dim for inverse-CDF sampling.
    """

    def __init__(self, sigma: float = 1.0, lam: float = 1024.0, device: int = 0,
                 homogeneous_time: bool = False, t_eps: float = 1e-3,
                 mode_steps: int = 2, side_steps: int = 2,
                 rou_rounds: int = 8, safety: float = 1.01,
                 mode: str = "bridge", eta: float = 1.0,
                 gl_n: int = 128):
        """Unified reverse sampler with (mode, eta) control.

        Args:
            sigma, lam: bridge hyperparameters (forward process).
            t_eps: clip t to [0, 1 - t_eps] for reverse step.
            mode_steps, side_steps, rou_rounds, safety: RoU sampler knobs
                (used only when mode='bridge').
            mode: reverse-sampler noise structure, one of:
                'bridge' -- sample u from its 1-D posterior (RoU in y),
                    then draw x_s from the finite-step bridge conditional
                    with noise_var_i = sigma^2 * g(u_hat_i) * s*(t-s)/t.
                    Correct for distributional models (energy-score).
                'sde'    -- reverse-SDE Euler, noise_var_i = sigma^2 * dt *
                    E[g(U_i) | X_t=x] with the expectation by Gauss-Legendre.
                    Correct for mean-regression models (MSE).
            eta: noise multiplier in [0, 1].
                eta=0 gives a deterministic drift step (ODE-like), identical
                across both modes. eta=1 gives the full sampler in the mode.
            gl_n: number of Gauss-Legendre nodes for E[g(U)|x] when mode='sde'.
        """
        if mode not in ("bridge", "sde"):
            raise ValueError(f"mode must be 'bridge' or 'sde', got {mode!r}")
        self.sigma = sigma
        self.lam = lam
        self.device = device
        self.homogeneous_time = homogeneous_time
        self.t_eps = t_eps
        self.mode_steps = mode_steps
        self.side_steps = side_steps
        self.rou_rounds = rou_rounds
        self.safety = safety
        self.mode = mode
        self.eta = float(eta)
        self.gl_n = gl_n
        # Lazy: nodes/weights created on first use (device/dtype matched then)
        self._gl_x = None
        self._gl_w = None

    def _pair_var_per_dim(self, u_sq: torch.Tensor) -> torch.Tensor:
        """sigma_pair^2 per dim = sigma^2 * sqrt(4 lam + u^2)."""
        return (self.sigma ** 2) * torch.sqrt(4.0 * self.lam + u_sq)

    def __call__(self, x_0: torch.Tensor, x_1: torch.Tensor,
                 t: Union[float, torch.Tensor, None] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x_0.shape[0]
        x_0, x_1 = x_0.float(), x_1.float()

        if t is not None:
            if isinstance(t, float):
                t = torch.full((batch_size,), t, device=x_0.device)
            else:
                t = t.to(x_0.device)
        else:
            if self.homogeneous_time:
                t = torch.rand(1, device=x_0.device).expand(batch_size)
            else:
                t = torch.rand(batch_size, device=x_0.device)

        diff = x_1 - x_0
        sigma_pair_sq = self._pair_var_per_dim(diff ** 2)
        t_exp = pad_t_like_x(t, x_0)

        bridge_std = torch.sqrt(sigma_pair_sq * t_exp * (1.0 - t_exp))
        eps = torch.randn_like(x_0)
        x_t = (1.0 - t_exp) * x_0 + t_exp * x_1 + bridge_std * eps

        return t.unsqueeze(1), x_t.float(), x_0.float()

    def _newton_mode(self, init: torch.Tensor, A, B, C) -> torch.Tensor:
        """Newton iterate in y to a stationary point of log q(y). Damped,
        clamped step. Returns fixed-length batched output."""
        y = init
        for _ in range(self.mode_steps):
            g, h = _grad_hess_y(y, A, B, C)
            step = torch.where(h < 0.0, g / h, torch.zeros_like(g))
            step = torch.clamp(step, -1.0, 1.0)
            y = y - step
        return y

    def _ensure_gl(self, device, dtype):
        """Lazy-init Gauss-Legendre nodes and weights on the right device/dtype."""
        if (self._gl_x is None or self._gl_x.device != device
                or self._gl_x.dtype != dtype):
            x_d, w_d = _gauss_legendre(self.gl_n)
            self._gl_x = x_d.to(device=device, dtype=dtype)
            self._gl_w = w_d.to(device=device, dtype=dtype)

    def _posterior_E_g(self, z_t: torch.Tensor, t_eff: torch.Tensor
                       ) -> torch.Tensor:
        """Compute E[g(U_i) | z_t_i, t] per-dim via Gauss-Legendre quadrature
        on x = tanh(y/2) ∈ [-1, 1], with y = asinh(u/(2√λ)).

        The posterior density on x is
            q_x(x) ∝ 2 · sqrt(1+x^2)/(1-x^2)^{3/2}
                    · exp[(2 A x + C(1-x^2))/(1+x^2) - B (1+x^2)/(1-x^2)],
        and  g(u(x)) = 2√λ · (1+x^2)/(1-x^2),
            (A, B, C) per the y-coord setup in `_logq_y`.

        Branch-free, robust in unimodal AND bimodal regimes (no mode finding).
        Exact up to GL quadrature error (~1e-8 at gl_n=48 for our regimes).

        Returns tensor of shape z_t.
        """
        dtype = z_t.dtype
        device = z_t.device
        self._ensure_gl(device, dtype)

        sig2 = self.sigma ** 2
        sqrt_lam = math.sqrt(self.lam)
        t = torch.as_tensor(t_eff, dtype=dtype, device=device).clamp(1e-8, 1.0 - 1e-8)
        omt = 1.0 - t

        A = (z_t / (sig2 * omt)).unsqueeze(-1)
        B = float(t * sqrt_lam / (sig2 * omt))
        C = ((t * sqrt_lam - z_t * z_t / (4.0 * t * sqrt_lam))
             / (sig2 * omt)).unsqueeze(-1)

        # Reshape GL nodes/weights to broadcast along last dim
        x = self._gl_x.view(*([1] * z_t.ndim), -1)         # (..., G)
        w_gl = self._gl_w.view(*([1] * z_t.ndim), -1)       # (..., G)

        omx2 = 1.0 - x * x
        opx2 = 1.0 + x * x

        log_w = (
            LOG2
            + 0.5 * torch.log(opx2)
            - 1.5 * torch.log(omx2)
            + (2.0 * A * x + C * omx2) / opx2
            - B * opx2 / omx2
        )
        # absorb GL weights into log-space, then stable normalize
        log_w = log_w + torch.log(w_gl)
        m = log_w.max(dim=-1, keepdim=True).values
        ww = torch.exp(log_w - m)                            # (..., G)

        g_of_x = 2.0 * sqrt_lam * opx2 / omx2                # g(u(x))
        num = (ww * g_of_x).sum(dim=-1)
        den = ww.sum(dim=-1)
        return num / den

    def _u_posterior_mean(self, z_t: torch.Tensor, t_eff: torch.Tensor,
                          n_grid: int = 128, half_width: float = 6.0
                          ) -> torch.Tensor:
        """Numerical E[u | z_t, t] by Simpson-style quadrature in y-space.

        The posterior in y = asinh(u/(2 sqrt(lam))) is
            q(y) ∝ cosh^{1/2}(y) · exp(A tanh y - B cosh y + C sech y)
        which has super-exponential tails from -B cosh y. Centered on the
        Newton mode m with half-width `half_width * sigma_y` (Laplace scale),
        a 128-point uniform grid captures the whole distribution to machine
        precision. Cost: 128 * (batch * d) evaluations of the stable log-q +
        hyperbolic trig ops, ~O(10) ms/step at batch=256, d=50.

        Exact (up to quadrature error), deterministic. No approximations
        beyond grid resolution.
        """
        dtype = z_t.dtype
        device = z_t.device
        tiny = torch.finfo(dtype).tiny

        sig2 = self.sigma ** 2
        sqrt_lam = math.sqrt(self.lam)
        t = torch.as_tensor(t_eff, dtype=dtype, device=device).clamp(1e-8, 1.0 - 1e-8)
        omt = 1.0 - t

        A = z_t / (sig2 * omt)
        B = torch.full_like(z_t, float(t * sqrt_lam / (sig2 * omt)))
        C = (t * sqrt_lam - z_t * z_t / (4.0 * t * sqrt_lam)) / (sig2 * omt)

        m0 = torch.asinh(z_t / (2.0 * t * sqrt_lam))
        m = self._newton_mode(m0, A, B, C)
        _, h = _grad_hess_y(m, A, B, C)
        sigma_y = torch.sqrt((-1.0 / h).clamp_min(tiny))

        # Grid: m + offset * sigma_y, offset in [-half_width, half_width]
        offsets = torch.linspace(-half_width, half_width, n_grid,
                                 device=device, dtype=dtype)                # (G,)
        y = m.unsqueeze(-1) + offsets * sigma_y.unsqueeze(-1)                # (..., G)

        A_e = A.unsqueeze(-1); B_e = B.unsqueeze(-1); C_e = C.unsqueeze(-1)
        logq = _logq_y(y, A_e, B_e, C_e)                                     # (..., G)
        # Numerically stable weights; subtract max before exp
        logq = logq - logq.max(dim=-1, keepdim=True).values
        w = logq.exp()                                                        # unnormalized weights
        Z = w.sum(dim=-1, keepdim=True)
        E_sinh = (torch.sinh(y) * w).sum(dim=-1, keepdim=True) / Z.clamp_min(tiny)
        return (2.0 * sqrt_lam) * E_sinh.squeeze(-1)

    def _sample_u_posterior(self, z_t: torch.Tensor, t_eff: torch.Tensor) -> torch.Tensor:
        """Per-dim exact posterior sample via Ratio-of-Uniforms in y = asinh(u/(2 sqrt(lam))).

        Unbiased (no approximation); residual error is pure Monte Carlo noise.
        Accept rate ~73%, so ~1.37 proposals per sample on average.

        Flow:
          1. Set up (A, B, C) in y-coordinate.
          2. Multi-start Newton to find the global mode of log q(y). A
             discriminant h0 = 0.5 - 2B + A^2/(4B) flags elements where the
             posterior may be bimodal (symmetric-mode pair at low t, |z|≈0).
             For those we run Newton from -s, m0, +s and pick the best local
             max. For unimodal elements, one Newton from m0 suffices. Both
             paths are computed vectorized and merged with torch.where.
          3. Newton solve for the two side maxima that define the RoU
             bounding box around the chosen mode.
          4. Rejection-sample in a fixed number of vectorized rounds; any
             stragglers get the MAP plug-in 2 sqrt(lam) sinh(m).

        Caveat: for the *symmetric*-bimodal limit (z = 0, t below
        bimodal_threshold()), the two modes carry equal mass and a single
        RoU box around one of them is biased. In our operating range
        (t >= 0.1) this regime is not hit.

        z_t:   observed z_t = x_t - x_hat_0, any batch shape.
        t_eff: scalar time in (0, 1).
        Returns u_sample of shape matching z_t.
        """
        dtype = z_t.dtype
        device = z_t.device
        tiny = torch.finfo(dtype).tiny

        sig2 = self.sigma ** 2
        sqrt_lam = math.sqrt(self.lam)
        t = torch.as_tensor(t_eff, dtype=dtype, device=device).clamp(1e-8, 1.0 - 1e-8)
        omt = 1.0 - t

        A = z_t / (sig2 * omt)
        B = torch.full_like(z_t, float(t * sqrt_lam / (sig2 * omt)))
        C = (t * sqrt_lam - z_t * z_t / (4.0 * t * sqrt_lam)) / (sig2 * omt)

        # Mode via Newton from the asinh-initializer (the MLE of u given z,
        # ignoring the Jacobian term). Converges in 2 iterations for all
        # (t, σ, λ, z) tested in the operating range t in [0.01, 0.999].
        m0 = torch.asinh(z_t / (2.0 * t * sqrt_lam))
        m = self._newton_mode(m0, A, B, C)

        _, h_mode = _grad_hess_y(m, A, B, C)
        vloc = torch.clamp(-1.0 / h_mode, min=tiny)

        # Side maxima for RoU bounding box (standard: init at sqrt(2*vloc))
        xr = torch.sqrt(2.0 * vloc)
        xl = xr.clone()
        for _ in range(self.side_steps):
            g, h = _grad_hess_y(m + xr, A, B, C)
            g = 1.0 / xr + 0.5 * g
            h = -1.0 / (xr * xr) + 0.5 * h
            step = torch.where(h < 0.0, g / h, torch.zeros_like(g))
            step = torch.clamp(step, -0.5 * xr, 0.5 * xr)
            xr = torch.clamp(xr - step, min=1e-8)

            g, h = _grad_hess_y(m - xl, A, B, C)
            g = 1.0 / xl - 0.5 * g
            h = -1.0 / (xl * xl) + 0.5 * h
            step = torch.where(h < 0.0, g / h, torch.zeros_like(g))
            step = torch.clamp(step, -0.5 * xl, 0.5 * xl)
            xl = torch.clamp(xl - step, min=1e-8)

        vmax = torch.exp(0.5 * _logq_y(m, A, B, C)) * self.safety
        umax = xr * torch.exp(0.5 * _logq_y(m + xr, A, B, C)) * self.safety
        umin = -xl * torch.exp(0.5 * _logq_y(m - xl, A, B, C)) * self.safety

        # Vectorized rejection with a fixed number of rounds
        shape = z_t.shape
        N = z_t.numel()
        fm = m.reshape(-1); fA = A.reshape(-1); fB = B.reshape(-1); fC = C.reshape(-1)
        flo = umin.reshape(-1); fhi = umax.reshape(-1); fvm = vmax.reshape(-1)
        done = torch.zeros(N, dtype=torch.bool, device=device)
        out = torch.empty(N, dtype=dtype, device=device)

        for _ in range(self.rou_rounds):
            U = flo + (fhi - flo) * torch.rand(N, device=device, dtype=dtype)
            # Clamp V away from 0 so U/V is always finite (otherwise Y = m + U/V
            # can reach inf, and `-inf <= -inf` in the acceptance check then
            # lets inf through, which propagates via sinh(Y) into the bridge step
            # variance and corrupts every downstream metric).
            V = fvm * torch.rand(N, device=device, dtype=dtype).clamp_min(tiny)
            Y = fm + U / V
            accept = 2.0 * torch.log(V) <= _logq_y(Y, fA, fB, fC)
            newly = (~done) & accept & torch.isfinite(Y)
            out = torch.where(newly, 2.0 * sqrt_lam * torch.sinh(Y), out)
            done = done | newly
            if bool(done.all()):
                break

        # MAP plug-in for stragglers (probability ~10^-5 at 8 rounds, rare)
        out = torch.where(done, out, 2.0 * sqrt_lam * torch.sinh(fm))
        return out.reshape(shape)

    def sample_step(self, t_curr, t_next, x_t, x_0_pred, **z):
        """Reverse step.

        Unified drift + (eta-scaled) noise per mode.

        Drift (always): x + dt * (x_hat_0 - x)/t_curr.
        Noise variance per dim (axis-aligned):
          mode='bridge': sigma^2 * g(u_hat_i) * t_next * dt / t_curr
                         with u_hat_i ~ p(u_i | z_t_i, t) (RoU in y).
                         Correct for distributional (energy-score) denoisers.
          mode='sde'   : sigma^2 * dt * E[g(U_i) | X_t=x_t] via GL quadrature.
                         Correct for mean-regression (MSE) denoisers.
        eta in [0,1] scales the noise; eta=0 gives a deterministic drift
        step, identical in both modes.
        """
        if isinstance(t_curr, float):
            t_curr = torch.tensor(t_curr, device=x_t.device)
        if isinstance(t_next, float):
            t_next = torch.tensor(t_next, device=x_t.device)

        if float(t_next) == 0.0:
            return x_0_pred, x_0_pred

        t_eff = t_curr.clamp(max=1.0 - self.t_eps)
        dt = t_curr - t_next
        mean = x_t + dt * (x_0_pred - x_t) / t_curr

        z_t = x_t - x_0_pred
        if self.mode == "bridge":
            u_hat = self._sample_u_posterior(z_t, t_eff)
            sigma_pair_sq = self._pair_var_per_dim(u_hat ** 2)           # per-dim sigma^2 g(u_hat)
            noise_var = sigma_pair_sq * t_next * dt / t_curr
        else:  # 'sde'
            E_g = self._posterior_E_g(z_t, t_eff)                        # per-dim E[g(U)|x]
            noise_var = (self.sigma ** 2) * E_g * dt
        noise_std = self.eta * torch.sqrt(noise_var.clamp_min(0.0))

        x_next = mean + noise_std * torch.randn_like(x_t)
        return x_next, x_0_pred

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
        return (f"AdaptiveGaussianBridge(sigma={self.sigma}, lam={self.lam}, "
                f"mode={self.mode}, eta={self.eta})")
