"""
Gaussian Brownian-bridge interpolant with x_0 target and a unified (mode, eta)
reverse sampler. Both samplers share the same drift; only the noise covariance
differs.

Forward:
    x_t = (1-t) x_0 + t x_1 + sigma * sqrt(t * (1-t)) * eps,   eps ~ N(0, I).

Reverse step (common drift; noise scaled by eta):
    drift   = (x_hat_0 - x_t) / t_curr
    noise_var = (see below, per mode)
    x_next  = x_t + dt * drift + eta * sqrt(noise_var) * xi

  mode='bridge'  -- finite-step bridge conditional (x_1 marginalised out):
      noise_var = sigma^2 * s * (t-s) / t.
      Exact at any finite dt; appropriate for distributional (energy-score)
      denoisers whose .sample() returns a predictive draw of x_0.
  mode='sde'     -- reverse-SDE Euler (infinitesimal limit):
      noise_var = sigma^2 * dt.
      Appropriate for mean-regression (MSE) denoisers whose .sample() returns
      E[X_0 | X_t]. O(dt) drift error, consistent as max_k dt -> 0.

  eta = 0        -- deterministic drift step (ODE-like), identical in both modes.
  eta = 1        -- full stochastic sampler in the chosen mode.
  0 < eta < 1    -- noise-scaled hybrid.

The two noise variances coincide at dt -> 0 and differ at finite dt by a factor
(1 - dt/t): the bridge noise is strictly smaller, because it accounts for the
exact conditional structure rather than Euler.
"""

import torch
from typing import Dict, Any, Optional, Tuple, Union


def pad_t_like_x(t, x):
    if isinstance(t, (float, int)):
        return t
    return t.view(-1, *([1] * (x.dim() - 1)))


class GaussianBridge:
    def __init__(self, sigma: float = 1.0, device: int = 0,
                 homogeneous_time: bool = False,
                 mode: str = "bridge", eta: float = 1.0):
        """
        Args:
            sigma, homogeneous_time: forward-process hyperparameters.
            mode: reverse-sampler noise structure, one of:
                'bridge' -- finite-step bridge conditional (energy-score).
                'sde'    -- reverse-SDE Euler (MSE / mean regression).
            eta: noise multiplier in [0, 1].
                eta=0 gives a deterministic drift step (ODE-like), identical
                across both modes. eta=1 gives the full sampler in the chosen
                mode.
        """
        if mode not in ("bridge", "sde"):
            raise ValueError(f"mode must be 'bridge' or 'sde', got {mode!r}")
        self.sigma = sigma
        self.device = device
        self.homogeneous_time = homogeneous_time
        self.mode = mode
        self.eta = float(eta)

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

        t_exp = pad_t_like_x(t, x_0)
        bridge_std = self.sigma * torch.sqrt(t_exp * (1.0 - t_exp))
        eps = torch.randn_like(x_0)
        x_t = (1.0 - t_exp) * x_0 + t_exp * x_1 + bridge_std * eps

        return t.unsqueeze(1), x_t.float(), x_0.float()

    def sample_step(self, t_curr, t_next, x_t, x_0_pred, **z):
        if isinstance(t_curr, float):
            t_curr = torch.tensor(t_curr, device=x_t.device)
        if isinstance(t_next, float):
            t_next = torch.tensor(t_next, device=x_t.device)

        if float(t_next) == 0.0:
            return x_0_pred, x_0_pred

        # Common drift: x_next_mean = x_t + dt * (x_hat_0 - x_t)/t_curr,
        # which is (t-s)/t x_hat_0 + s/t x_t with s = t_curr - dt.
        dt = t_curr - t_next                                                # positive
        mean = x_t + dt * (x_0_pred - x_t) / t_curr

        # Noise variance per mode (in the fixed-sigma bridge this is scalar):
        #   bridge: sigma^2 * s * (t-s) / t
        #   sde:    sigma^2 * dt
        if self.mode == "bridge":
            noise_var = (self.sigma ** 2) * t_next * dt / t_curr
        else:  # 'sde'
            noise_var = (self.sigma ** 2) * dt
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

        time_points = torch.linspace(0, 1, n_steps + 1).to(x_t.device)

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
        return f"GaussianBridge(sigma={self.sigma}, mode={self.mode}, eta={self.eta})"
