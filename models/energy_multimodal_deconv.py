import torch
import torch.nn as nn
import torch.nn.functional as F
from .energy_multimodal import MultimodalEnergyScoreLoss
from .proj import rescale, randomized_round, randomized_round_groups_exact


class MultimodalDeconvolutionEnergyScoreLoss(MultimodalEnergyScoreLoss):
    """
    Multimodal deconvolution energy score loss that handles aggregation.
    
    Key insight: Only the counts get aggregated, images remain at individual level.
    
    For images: Standard energy score at individual level
    For counts: Energy score at aggregate level (group sums)
    
    This allows us to learn:
    - Image generation for each individual 
    - Count allocation that sums to group targets
    """
    
    def __init__(
        self,
        architecture,
        noise_dim: int = 16,
        m_samples: int = 16,
        lambda_energy: float = 1.0,
        weights: dict = None,
    ):
        super().__init__(
            architecture=architecture,
            noise_dim=noise_dim,
            m_samples=m_samples,
            lambda_energy=lambda_energy,
            weights=weights,
        )

    def loss(self, target, inputs, agg):
        """
        Compute multimodal deconvolution energy score loss.
        
        Args:
            target: Dict with targets at different levels:
                   - target['img']: [B*G, C, H, W] individual image targets  
                   - target['counts']: [G, count_dim] aggregated count targets
            inputs: Dict of input tensors for model
            agg: [G, B*G] aggregation matrix (sparse or dense)
            
        Returns:
            Combined energy score loss
        """
        # Get base input to determine batch size
        def find_tensor(value):
            if isinstance(value, torch.Tensor):
                return value
            elif isinstance(value, dict):
                for v in value.values():
                    result = find_tensor(v)
                    if result is not None:
                        return result
            return None
        
        base_input = None
        for value in inputs.values():
            base_input = find_tensor(value)
            if base_input is not None:
                break
                
        n = base_input.shape[0]  # This is B*G (total individuals)
        λ = self.lambda_energy

        # Replicate all inputs m times (reuse parent logic)
        def replicate_tensor_or_dict(value):
            if isinstance(value, torch.Tensor):
                # Expand to [n, m, ...] then reshape to [n*m, ...]
                expanded = value.unsqueeze(1).expand(-1, self.m, *[-1] * (value.dim() - 1))
                return expanded.reshape(n * self.m, *value.shape[1:])
            elif isinstance(value, dict):
                # Recursively handle nested dictionaries
                return {k: replicate_tensor_or_dict(v) for k, v in value.items()}
            else:
                return value
        
        replicated_inputs = {}
        for key, value in inputs.items():
            if key == 'A':
                continue
            replicated_inputs[key] = replicate_tensor_or_dict(value)

        # Add noise (energy score specific requirement)  
        noise = torch.randn(n * self.m, self.noise_dim, device=base_input.device)
        replicated_inputs['noise'] = noise

        # Get all predictions: [n*m, ...] for each modality
        predictions = self.architecture(**replicated_inputs)
        
        total_loss = 0.0
        
        # Process each modality with appropriate aggregation
        for k in predictions:
            predictions_k = predictions[k].reshape(n, self.m, *predictions[k].shape[1:])
            
            if k in target['X_0']:
                # Apply aggregation for counts: [G, B*G] @ [B*G, m*D] → [G, m, D]
                B_times_G, m, d = predictions_k.shape
                predictions_flat = predictions_k.view(B_times_G, -1)  # [B*G, m*D]
                agg_predictions_flat = agg @ predictions_flat  # [G, m*D]
                aggregated_predictions = agg_predictions_flat.view(agg.shape[0], m, d)  # [G, m, D]
                loss = self._compute_energy_score(aggregated_predictions, target['X_0'][k], λ)
            else:
                # No aggregation for other modalities (e.g., images)
                loss = self._compute_energy_score(predictions_k, target['x_0'][k], λ)
            
            total_loss += self.weights[k] * loss

        return total_loss

    @torch.no_grad()
    def conditional_sample(
        self,
        inputs: dict,              # tensors shaped [B*G, ...]; no 'noise' key required
        target_sum: dict,          # Dict with 'counts': [G, D] and other modalities as needed
        agg: torch.Tensor,         # [G, B*G] aggregation matrix
        keep_rows: bool = True,    # anchor each individual's total mass
        return_float: bool = False # if True: return non-integer y with exact columns
    ):
        """
        Multimodal allocation-based conditional sampler.
        
        1) Sample prior predictions from the model (with fresh noise).
        2) For images: return as-is (no aggregation constraints)
        3) For counts: KL/I-projection onto {sum_g y = target_sum['counts']}
        4) Exact integerization for counts across G per (G,D).

        Args:
            inputs: Dict of input tensors [B*G, ...]
            target_sum: Dict with target sums, e.g. {'counts': [G, D]}
            agg: [G, B*G] aggregation matrix 
            keep_rows: Whether to preserve individual count totals (unused for now)
            return_float: Whether to return float counts before integerization
            
        Returns:
            Dict with predictions for each modality
        """
        # Get shapes
        base = next(iter(inputs.values()))
        if isinstance(base, dict):
            base = next(iter(base.values()))
        device = base.device
        BG = base.shape[0]  # B*G total individuals
        E = self.noise_dim

        if 'x_0' in inputs:
            x_0 = inputs['x_0']
            del inputs['x_0']
        else:
            x_0 = {}

        # Remove aggregation matrix from model inputs
        if 'A' in inputs:
            del inputs['A']

        # Sample prior predictions from architecture with noise
        flat_inputs = {k: v for k, v in inputs.items()}
        noise = torch.randn(BG, E, device=device)
        flat_inputs['noise'] = noise
        
        # Get predictions for each modality
        predictions = self.forward(flat_inputs)
        
        result = {}
    
        # Process each modality with appropriate constraints
        for k in predictions:
            if k in target_sum:
                # Apply deconvolution constraints for counts
                
                # KL/I-projection onto aggregation constraints
                C = target_sum[k].to(device)  # [G, D]
                y = rescale(
                    x=predictions[k],
                    C=C,
                    A=agg
                )
                
                if k == 'counts':
                    # Exact integerization 
                    y_post = randomized_round_groups_exact(
                        y, C, agg
                    )
                    y = y_post.long()
                
                result[k] = y
            else:
                # If we have x_0, use it instead of predictions
                # this lets us use "conditional" sampling where we condition on one modality
                # while the rest are sampled from the model
                result[k] = x_0[k] if k in x_0 else predictions[k]
                if k == 'counts':
                    result[k] = randomized_round(result[k])
        
        return result

    def sample(self, **kwargs):
        """
        Sample prediction using arbitrary kwargs inputs.
        
        If target_sum is provided, performs conditional sampling with deconvolution.
        Otherwise, performs unconstrained sampling.
        """
        if 'target_sum' in kwargs:
            target_sum = kwargs['target_sum']
            agg = kwargs['A']  # Aggregation matrix
            del kwargs['target_sum']
            del kwargs['A']
            return self.conditional_sample(kwargs, target_sum, agg)
        else:
            # call super to get predictions
            return super().sample(**kwargs)
