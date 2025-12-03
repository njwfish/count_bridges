import cupy as cp
import numpy as np
import torch
from typing import Callable
from .scheduling import make_weight_schedule
from .utils import dlpack_backend
from cupyx.scipy.special import logsumexp
from counting_flows.models.proj import rescale, randomized_round_groups_exact

class SkellamBridge:
    """
    Collate for Skellam birth–death bridge with time‑varying λ₊, λ₋.
    Produces x_t drawn via Hypergeometric(N,B₁,N_t).
    """

    def __init__(
        self,
        slack_sampler: Callable,
        delta: bool = False,
        schedule_type: str = "linear",
        homogeneous_time: bool = False,
        backend: str = "torch",
        device: int = 0,
        **schedule_kwargs,
    ):
        self.device           = device
        self.slack_sampler    = slack_sampler
        self.schedule_type    = schedule_type
        self.delta            = delta
        # this is really only for comparison with the constrained bridge
        # there is no need to sample homogeneous time points in this bridge
        self.homogeneous_time = homogeneous_time
        self.backend          = backend
        self.weight_schedule  = make_weight_schedule(
            schedule_type, **schedule_kwargs
        )

    def __call__(self, x_0, x_1, t=None):
        with cp.cuda.Device(self.device):
            x_0, x_1 = dlpack_backend(x_0, x_1, backend='cupy', dtype="int32")

            b = x_0.shape[0]

            diff = (x_1 - x_0)
            w_1 = self.weight_schedule(1)
            M = self.slack_sampler(cp.abs(diff), w_1)
            N_1 = cp.abs(diff) + 2 * M
            B_1 = (N_1 + diff) // 2
            # assert cp.all((N_1 + diff) % 2 == 0), "N_1 + diff should be even"


            if t is not None:
                if isinstance(t, float):
                    t = cp.full((b, 1), t, dtype=cp.float32)
                else:
                    t = cp.from_dlpack(t)
            else:
                if self.homogeneous_time:
                    t = cp.random.rand().expand(b, 1)
                else:
                    t = cp.random.rand(b, 1)
                
            w_t   = self.weight_schedule(t)
            
            N_t = cp.random.binomial(N_1, w_t, dtype=cp.int32)
            
            # need to gaurd against zero nsample
            non_zero = N_t > 0
            B_t = cp.zeros_like(B_1)
            B_t[non_zero] = cp.random.hypergeometric(
                ngood=B_1[non_zero],
                nbad=N_1[non_zero] - B_1[non_zero],
                nsample=N_t[non_zero], 
                dtype=cp.int32
            )

            x_t = x_1 - 2 * (B_1 - B_t) + (N_1 - N_t)

            diff_t = cp.abs(x_t - x_0)
            M_t    = (N_t - diff_t) // 2
            # assert cp.all((N_t - diff_t) % 2 == 0), "N_t - diff_t should be even"

            x_t, M_t, t, x_0 = dlpack_backend(x_t, M_t, t, x_0, backend=self.backend, dtype="float32")
            return t, x_t, x_0 - x_t if self.delta else x_0


    def sample_step(self, t_curr, t_next, x_t, model_out, return_in_backend: bool = True, **z):
        x0_hat_t = cp.from_dlpack(model_out) + x_t if self.delta else cp.from_dlpack(model_out)
        
        x0_hat_t = x0_hat_t.round().astype(cp.int32)
        diff = x_t - x0_hat_t

        M_t = self.slack_sampler(cp.abs(diff), self.weight_schedule(t_curr))

        N_t = cp.abs(diff) + 2 * M_t
        B_t = (N_t + diff) // 2
        # post correction should be even
        # assert cp.all((N_t + diff) % 2 == 0), "N_t + diff should be even"

        ρ = (self.weight_schedule(t_next) / self.weight_schedule(t_curr))
        non_zero = N_t > 0
        N_s = cp.zeros_like(N_t)
        N_s[non_zero] = cp.random.binomial(N_t[non_zero], ρ)

        non_zero = N_s > 0
        B_s = cp.zeros_like(B_t)
        B_s[non_zero] = cp.random.hypergeometric(
            ngood   = B_t[non_zero],
            nbad    = N_t[non_zero] - B_t[non_zero],
            nsample = N_s[non_zero]
        )

        x_s = x_t - 2 * (B_t - B_s) + (N_t - N_s)
        diff_s = cp.abs(x_s - x0_hat_t)
        # since we corrected above this should be fine
        M_s = (N_s - diff_s) // 2 
        # assert cp.all((N_s - diff_s) % 2 == 0), "N_s - diff_s should be even"

        if return_in_backend:
            return dlpack_backend(x_s, x0_hat_t, backend=self.backend, dtype="float32")
        return x_s, x0_hat_t
    
    def sampler(
        self,
        x_1:  cp.ndarray,
        z:   dict,
        model,
        return_trajectory: bool = False,
        return_x_hat:      bool = False,
        guidance_x_0:      cp.ndarray = None,
        guidance_constant: float = None,
        n_steps: int = 10,
    ):
        
        with cp.cuda.Device(self.device):
            b = x_1.shape[0]
            x_1 = x_t = cp.from_dlpack(x_1).round().astype(cp.int32)

            time_points = cp.linspace(0, 1, n_steps + 1)
            
            traj, xhat_traj = [x_t], []

            if guidance_x_0 is not None:
                guidance_schedule = np.linspace(guidance_constant, 1, n_steps + 1)
                target_sum = z['target_sum']
                agg = torch.ones((1, guidance_x_0.shape[1]), device=self.device)
                guidance_x_0_rescaled = rescale(guidance_x_0.squeeze(0), target_sum, agg)
                guidance_x_0_proj = randomized_round_groups_exact(guidance_x_0_rescaled, target_sum, agg)

            for k in range(n_steps, 0, -1):
                t_curr = time_points[k]
                t_next = time_points[k-1]
                t = cp.broadcast_to(time_points[k], (b, 1))
    
                x_t_dl, t_dl = dlpack_backend(x_t, t, backend=self.backend, dtype="float32")

                x0_hat_t = model.sample(x_t=x_t_dl, t=t_dl, **z)

                if guidance_x_0 is not None:
                    x0_hat_t =  guidance_schedule[k] * guidance_x_0_proj + (1 - guidance_schedule[k]) * x0_hat_t

                x_t, x0_hat_t = self.sample_step(t_curr, t_next, x_t, x0_hat_t, **z, return_in_backend=False)

                if return_trajectory: traj.append(x_t)
                if return_x_hat:     xhat_traj.append(x0_hat_t)

            outs = [x_t]
            if return_trajectory: outs.append(cp.stack(traj))
            if return_x_hat:      outs.append(cp.stack(xhat_traj))
            return dlpack_backend(*outs, backend=self.backend, dtype="float32") if len(outs) > 1 else dlpack_backend(x_t, backend=self.backend, dtype="float32")[0]
