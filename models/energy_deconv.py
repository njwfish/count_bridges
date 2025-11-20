import torch
import torch.nn as nn

import torch
import torch.nn.functional as F

from .energy import EnergyScoreLoss
from .proj import rescale, randomized_round, randomized_round_groups_exact


class DeconvolutionEnergyScoreLoss(EnergyScoreLoss):
    """
    Distributional-diffusion energy score loss (eq.14) with m-sample approximation.
    Works with arbitrary architectures and clean input interface.
    """
    def __init__(
        self,
        architecture,
        noise_dim: int = 16,
        m_samples: int = 16,
        lambda_energy: float = 1.0,
        use_arch_projection: bool = True,
        project_round_method: str = 'exact_randomized',
    ):
        super().__init__(
            architecture=architecture,
            noise_dim=noise_dim,
            m_samples=m_samples,
            lambda_energy=lambda_energy,
            use_arch_projection=use_arch_projection,
        )
        self.project_round_method = project_round_method
        assert self.project_round_method in ['exact_randomized', 'randomized', 'round']

    def forward(self, inputs, eps=None):
        """
        Forward pass through architecture.
        
        Args:
            inputs: Dict of input tensors
            
        Returns:
            Prediction tensor
        """
        # Add noise (energy score specific requirement)
        inputs_with_noise = inputs.copy()
        base_input = list(inputs.values())[0]
        batch_size = base_input.shape[0]
        if eps is None:
            inputs_with_noise['noise'] = torch.randn(batch_size, self.noise_dim, device=base_input.device)
        else:
            inputs_with_noise['noise'] = eps
        
        if 'A' in inputs:
            del inputs_with_noise['A']

        for key, value in inputs_with_noise.items():
            print(key, value.shape)
        
        return self.architecture(**inputs_with_noise)

    def loss(self, target, inputs, agg):
        """
        Empirical energy-score (Distrib. Diffusion Models eq.14):
          L = mean_i [
            (1/m) ∑_j ||target_i - pred_ij||
            - (λ/(2(m-1))) ∑_{j≠j'} ||pred_ij - pred_ij'||
          ]
        """
        base_input = list(inputs.values())[0]
        n, λ = base_input.shape[0], self.lambda_energy
        num_groups = target.shape[0]

        # Replicate all inputs m times by iterating over the dict
        if 'A' in inputs:
            del inputs['A']
        replicated_inputs = {}
        for key, value in inputs.items():
            replicated_inputs[key] = value.unsqueeze(1).expand(-1, self.m, *[-1] * (value.dim() - 1)).reshape(n * self.m, *value.shape[1:])

        # Add noise (energy score specific requirement)  
        noise = torch.randn(n * self.m, self.noise_dim, device=target.device)
        replicated_inputs['noise'] = noise

        # Get all predictions: [n*m, x_dim] → view [n, m, x_dim]
        predictions = self.architecture(**replicated_inputs).reshape(n, self.m, -1)
        
        # Apply sparse aggregation: [G, B] @ [B, m*D] → [G, m*D] → [G, m, D]
        B, m, d = predictions.shape
        predictions_flat = predictions.view(B, -1)  # [B, m*D] - flatten last two dims
        agg_predictions_flat = agg @ predictions_flat  # [G, m*D]
        predictions = agg_predictions_flat.view(agg.shape[0], m, d)  # [G, m, D]

        return self._compute_energy_score(predictions, target, λ)


    def sample(self, **kwargs):
        """
        Sample prediction using arbitrary kwargs inputs.
        """
        if 'target_sum' in kwargs:
            S = kwargs['target_sum']
            del kwargs['target_sum']
            return self.conditional_sample(kwargs, S)
        else:
            prediction = self.forward(kwargs)
            return randomized_round(prediction)

    @torch.no_grad()
    def conditional_sample(
        self,
        inputs: dict,              # tensors shaped [B*G, ...]; no 'noise' key required
        target_sum: torch.Tensor,  # [B, D] aggregates across G (per batch item)
        return_float: bool = False # if True: return non-integer y with exact columns
    ):
        """
        Allocation-based conditional sampler (no latent optimization):
        1) Sample prior counts x_pred from the model (with fresh noise).
        2) KL/I-projection (IPF) onto {sum_g y = target_sum} (and row totals if keep_rows=True).
        3) Exact integerization across G per (B,D).

        • Multiplicative/IPF updates ⇒ small deltas when targets are close (anchored to x_pred).
        • keep_rows=True preserves each individual's total mass from x_pred (strong anchoring).
        """

        # ----- shapes -----
        base = next(iter(inputs.values()))
        agg = inputs['A']
        device = base.device
        BG = base.shape[0]
        B, D = target_sum.shape
        E = self.noise_dim

        # ----- 1) sample prior x_pred from architecture with noise -----
        flat_inputs = {k: v for k, v in inputs.items()}
        noise = torch.randn(BG, E, device=device)
        flat_inputs['noise'] = noise
        if 'A' in flat_inputs:
            del flat_inputs['A']
        x_pred = self.architecture(**flat_inputs)          # [B*G, D] (logits or reals)

        # Optional: normalize tiny columns to avoid degenerate zeros with positive targets
        # (ipf_robust_anchor also smooths)
        C = target_sum.to(device)

        # ----- 2) KL/I-projection (IPF) anchored to prior -----
        y_float = rescale(
            x=x_pred,
            C=C,
            A=agg
        )

        # print(y_float.float().var(dim=0))

        if return_float:
            return y_float

        # ----- 3) exact integerization across G per (B,D) -----
        if self.project_round_method == 'exact_randomized':
            y_int = randomized_round_groups_exact(y_float, C, agg)
        elif self.project_round_method == 'randomized':
            y_int = randomized_round(y_float)
        else:
            y_int = y_float.round().long()
        # print(y_int.float().var(dim=0))
        return y_int


