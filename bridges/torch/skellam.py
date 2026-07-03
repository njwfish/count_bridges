"""Torch-native Skellam birth-death bridge (zero-slack).

A pure-PyTorch reimplementation of ``counting_flows.bridges.cupy.skellam.SkellamBridge``
for the ZERO-SLACK (lambda = 0, ``ZeroM``) case used in production. Zero slack
collapses the Skellam bridge to a single binomial per step -- no hypergeometric,
no Bessel -- so BOTH the forward corruption (training) and the reverse sampler
(inference) are exact with only ``torch.binomial``:

    forward  (training):  x_t = x_0 + sign(x_1 - x_0) * Binomial(|x_1 - x_0|, w_t)
    reverse  (sampling):  x_s = x0_hat + sign(x_t - x0_hat) * Binomial(|x_t - x0_hat|, rho)
                          with rho = w(t_next) / w(t_curr)

Keeping everything in torch (no cupy) means a single CUDA memory pool -- so large
cell sets fit GPU-resident alongside the denoiser's token stream -- and no dlpack
round-trips, which is faster at both training and inference.

Drop-in for the cupy ``SkellamBridge``: same ``__call__`` / ``sample_step`` /
``sampler`` interface and the same ``(t, x_t, target)`` / ``x_s`` contracts. Valid
ONLY for zero slack; a non-zero ``slack_sampler`` raises.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch


def weight_schedule(schedule_type: str, x, **kwargs):
    """Torch/scalar-safe copy of ``counting_flows.bridges.cupy.scheduling.weight_functions``.

    ``x`` may be a python float or a torch tensor; returns the same kind. Monotone 0->1.
    """
    if schedule_type == "linear":
        return x
    if schedule_type == "cosine":
        if isinstance(x, torch.Tensor):
            return 0.5 * (1.0 - torch.cos(math.pi * x))
        return 0.5 * (1.0 - math.cos(math.pi * x))
    if schedule_type == "early_dense":
        return x ** (1.0 / kwargs.get("concentration", 2.0))
    if schedule_type == "late_dense":
        return x ** kwargs.get("concentration", 2.0)
    if schedule_type == "sigmoid":
        s = kwargs.get("steepness", 10.0)
        if isinstance(x, torch.Tensor):
            return 1.0 / (1.0 + torch.exp(-s * (x - 0.5)))
        return 1.0 / (1.0 + math.exp(-s * (x - 0.5)))
    raise ValueError(f"Unknown schedule_type: {schedule_type!r}")


class SkellamBridge:
    """Zero-slack Skellam birth-death bridge, torch-native. See module docstring."""

    def __init__(
        self,
        slack_sampler: Optional[Callable] = None,
        delta: bool = False,
        schedule_type: str = "linear",
        homogeneous_time: bool = False,
        device: int = 0,
        backend: str = "torch",
        **schedule_kwargs,
    ):
        # ``slack_sampler`` is accepted for config/interface parity with the cupy
        # bridge, but this bridge implements ONLY the zero-slack reduction. Fail
        # loudly on a non-zero slack sampler rather than silently ignore it.
        if slack_sampler is not None:
            from counting_flows.bridges.cupy.slack_samplers import ZeroM
            if not isinstance(slack_sampler, ZeroM):
                raise ValueError(
                    "torch SkellamBridge implements only zero-slack (ZeroM); got "
                    f"slack_sampler={type(slack_sampler).__name__}"
                )
        self.slack_sampler = slack_sampler
        self.delta = delta
        self.schedule_type = schedule_type
        self.homogeneous_time = homogeneous_time
        self.device = device
        self.backend = backend
        self.schedule_kwargs = dict(schedule_kwargs)

    def _w(self, x):
        return weight_schedule(self.schedule_type, x, **self.schedule_kwargs)

    # ---- forward corruption (training collate) ----
    def __call__(self, x_0: torch.Tensor, x_1: torch.Tensor, t=None):
        """Draw x_t on the bridge from x_0 (target) to x_1 (source).

        Returns ``(t, x_t, target)`` with ``target = x_0 - x_t`` if ``delta`` else
        ``x_0`` -- the cupy bridge's return contract.
        """
        x_0 = torch.round(x_0.to(torch.float32))
        x_1 = torch.round(x_1.to(torch.float32))
        b = x_0.shape[0]
        if t is None:
            if self.homogeneous_time:
                t = torch.rand((), device=x_0.device).expand(b, 1)
            else:
                t = torch.rand(b, 1, device=x_0.device)
        elif not isinstance(t, torch.Tensor):
            t = torch.full((b, 1), float(t), device=x_0.device)
        w_t = self._w(t)                                    # [b, 1]
        diff = x_1 - x_0
        n = torch.abs(diff)
        # broadcast the per-sample weight over the remaining (gene) dims of n
        prob = w_t.reshape(b, *([1] * (n.dim() - 1))).expand_as(n).to(n.dtype)
        n_t = torch.binomial(n, prob)
        x_t = x_0 + torch.sign(diff) * n_t
        target = (x_0 - x_t) if self.delta else x_0
        return t, x_t, target

    # ---- reverse step (sampling) ----
    def sample_step(self, t_curr, t_next, x_t, model_out, return_in_backend: bool = True, **z):
        """One reverse step. ``model_out`` is the RAW model output (delta-add
        happens here, matching the cupy ``sample_step``). Returns
        ``(x_s, x0_hat)`` as int-valued float tensors. ``**z`` is accepted and
        ignored (parity with the cupy signature)."""
        x0_hat = (model_out + x_t) if self.delta else model_out
        x0_hat = torch.round(x0_hat)
        diff = torch.round(x_t) - x0_hat
        n = torch.abs(diff)
        rho = float(self._w(t_next)) / float(self._w(t_curr))
        rho = min(max(rho, 0.0), 1.0)
        n_s = torch.binomial(n.clamp_min(0.0), torch.full_like(n, rho))
        x_s = x0_hat + torch.sign(diff) * n_s
        return x_s, x0_hat

    def sampler(
        self,
        x_1: torch.Tensor,
        z: dict,
        model,
        return_trajectory: bool = False,
        return_x_hat: bool = False,
        guidance_x_0: Optional[torch.Tensor] = None,
        guidance_schedule=None,
        n_steps: int = 10,
    ):
        """Reverse-time generation. Same interface as the cupy sampler; keeps
        everything as torch tensors on the GPU. ``z`` (D_source/D_target, gene ids,
        target_sum, target_mean, forward_chunk_size, ...) is passed straight to
        ``model.sample``; extra keys are ignored by ``sample_step``."""
        x_t = torch.round(x_1.to(torch.float32))
        b = x_t.shape[0]
        device = x_t.device
        time_points = torch.linspace(0, 1, n_steps + 1, device=device)

        traj, xhat = [x_t], []
        for k in range(n_steps, 0, -1):
            t_curr = float(time_points[k].item())
            t_next = float(time_points[k - 1].item())
            t = torch.full((b, 1), t_curr, dtype=torch.float32, device=device)

            x0_hat = model.sample(x_t=x_t, t=t, **z)
            if guidance_x_0 is not None:
                x0_hat = guidance_schedule[k] * guidance_x_0 + (1 - guidance_schedule[k]) * x0_hat

            x_t, x0_hat_int = self.sample_step(t_curr, t_next, x_t, x0_hat)
            if return_trajectory:
                traj.append(x_t)
            if return_x_hat:
                xhat.append(x0_hat_int)

        # Match the cupy sampler's return contract: with no trajectory/x_hat it
        # returns the BARE x_t tensor [B, n, G] (callers do out[0] to strip the
        # batch dim -> [n, G]); otherwise a tuple (x_t, traj?, xhat?).
        if not (return_trajectory or return_x_hat):
            return x_t
        outs = [x_t]
        if return_trajectory:
            outs.append(torch.stack(traj))
        if return_x_hat:
            outs.append(torch.stack(xhat))
        return tuple(outs)
