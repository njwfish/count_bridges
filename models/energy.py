import torch
import torch.nn as nn

from .proj import rescale, randomized_round_groups_exact

class EnergyScoreLoss(nn.Module):
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
    ):
        super().__init__()
        self.architecture = architecture
        self.noise_dim = noise_dim
        self.m = m_samples
        self.lambda_energy = lambda_energy
        self.use_arch_projection = use_arch_projection

    def _compute_energy_score(self, predictions, target, λ):
        """
        Compute energy score for a single modality.
        
        Args:
            predictions: [n, m, d] - m predictions per sample
            target: [n, d] - target values
            λ: energy score regularization parameter
            
        Returns:
            energy_score: scalar loss value
        """
        n, m = predictions.shape[:2]  # Handle arbitrary dimensions after m
        
        # Flatten spatial dimensions for distance computation
        predictions_flat = predictions.view(n, m, -1)  # [n, m, d_flat]
        target_flat = target.view(n, -1)  # [n, d_flat]
        
        # Confinement term: distance to target
        target_expanded = target_flat.unsqueeze(1).expand(-1, m, -1)  # [n, m, d_flat]
        term_conf = (predictions_flat - target_expanded).norm(dim=2).mean(dim=1)  # [n]
        
        # Interaction term (efficient batched computation)
        # Using ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩ identity
        sq = predictions_flat.pow(2).sum(dim=2)  # [n, m] - squared norms
        inn = torch.bmm(predictions_flat, predictions_flat.transpose(1,2))  # [n, m, m] - inner products
        sqd = sq.unsqueeze(2) + sq.unsqueeze(1) - 2*inn  # [n, m, m] - squared distances
        sqd = torch.clamp(sqd, min=1e-6)  # avoid sqrt(0)
        d = sqd.sqrt()  # [n, m, m] - distances
        
        # Mean of off-diagonal pairwise distances
        m_mask = torch.ones(m, m, device=predictions.device) - torch.eye(m, device=predictions.device)
        mean_pd = (d * m_mask).sum(dim=(1,2)) / (m * (m - 1))  # [n]
        term_int = (λ / 2.0) * mean_pd  # [n]
        
        return (term_conf - term_int).mean()

    def forward(self, inputs):
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
        inputs_with_noise['noise'] = torch.randn(batch_size, self.noise_dim, device=base_input.device)
        
        return self.architecture(**inputs_with_noise)

    def loss(self, target, inputs):
        """
        Empirical energy-score (Distrib. Diffusion Models eq.14):
          L = mean_i [
            (1/m) ∑_j ||target_i - pred_ij||
            - (λ/(2(m-1))) ∑_{j≠j'} ||pred_ij - pred_ij'||
          ]
        """
        n, λ = target.shape[0], self.lambda_energy

        # Replicate all inputs m times by iterating over the dict
        replicated_inputs = {}
        for key, value in inputs.items():
            if key == 'seq':
                replicated_inputs[key] = value
                continue
            if key == 'target_sum':
                replicated_inputs[key] = value.expand(self.m, -1)
                continue
            replicated_inputs[key] = value.unsqueeze(1).expand(-1, self.m, *[-1] * (value.dim() - 1)).reshape(n * self.m, *value.shape[1:])

        # Add noise (energy score specific requirement)  
        noise = torch.randn(n * self.m, self.noise_dim, device=target.device)
        replicated_inputs['noise'] = noise

        # Get all predictions: [n*m, x_dim] → view [n, m, x_dim]
        predictions = self.architecture(**replicated_inputs)   
        predictions = predictions.view(n, self.m, *predictions.shape[1:])  # [n, m, d]

        return self._compute_energy_score(predictions, target, λ)
        
    @torch.no_grad()
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
            return prediction.round().long()

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
        if self.use_arch_projection:
            flat_inputs['target_sum'] = target_sum
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
        y_int = randomized_round_groups_exact(y_float, C, agg)
        # print(y_int.float().var(dim=0))
        return y_int
