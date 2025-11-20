import torch
import torch.nn as nn
import torch.nn.functional as F

from .energy import EnergyScoreLoss
from .proj import randomized_round

class MultimodalEnergyScoreLoss(EnergyScoreLoss):
    """
    Multimodal energy score loss that computes energy scores for both images and counts.
    
    This is the basic version without deconvolution aggregation - it treats each modality
    independently and combines their energy scores.
    
    Energy score formula (per modality):
    L = mean_i [
        (1/m) ∑_j ||target_i - pred_ij||
        - (λ/(2(m-1))) ∑_{j≠j'} ||pred_ij - pred_ij'||
    ]
    
    Total loss is weighted combination of image and count energy scores.
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
        )
        self.weights = weights
        
        if self.weights is None:
            from collections import defaultdict
            self.weights = defaultdict(lambda: 1.0)

    def forward(self, inputs, eps=None):
        """
        Forward pass through architecture.
        
        Args:
            inputs: Dict of input tensors
            eps: Optional noise tensor (if None, generate fresh noise)
            
        Returns:
            Dict with 'img' and 'counts' predictions
        """
        # Add noise (energy score specific requirement)
        inputs_with_noise = inputs.copy()
        
        # Get batch size from any tensor in inputs
        base_input = None
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                base_input = value
                break
            elif isinstance(value, dict):
                for v in value.values():
                    if isinstance(v, torch.Tensor):
                        base_input = v
                        break
                break
        
        if base_input is None:
            raise ValueError("No tensor found in inputs")
            
        batch_size = base_input.shape[0]
        
        if eps is None:
            inputs_with_noise['noise'] = torch.randn(
                batch_size, self.noise_dim, device=base_input.device
            )
        else:
            inputs_with_noise['noise'] = eps
            
        return self.architecture(**inputs_with_noise)

    def loss(self, target, inputs):
        """
        Compute multimodal energy score loss.
        
        Args:
            target: Dict with target values
                   - target[k]: [n, ...] target values for modality k
            inputs: Dict of input tensors for model
            
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
                
        n = base_input.shape[0]
        λ = self.lambda_energy

        # Replicate all inputs m times
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
            replicated_inputs[key] = replicate_tensor_or_dict(value)

        # Add noise (energy score specific requirement)  
        noise = torch.randn(n * self.m, self.noise_dim, device=base_input.device)
        replicated_inputs['noise'] = noise

        # Get all predictions: [n*m, ...] for each modality
        predictions = self.architecture(**replicated_inputs)
        
        total_loss = 0.0
        
        for k in predictions:
            predictions_k = predictions[k].reshape(n, self.m, *predictions[k].shape[1:])
            loss = self._compute_energy_score(predictions_k, target[k], λ)
            total_loss += self.weights[k] * loss

        return total_loss

    def sample(self, **kwargs):
        """
        Sample prediction using arbitrary kwargs inputs.
        
        Returns:
            Dict with predictions for each modality
        """
        # this lets us condition on x_0 if x_0 is provided
        if 'x_0' in kwargs:
            x_0 = kwargs['x_0']
            del kwargs['x_0']
        else:
            x_0 = {}

        predictions = self.forward(kwargs)
        
        # Post-process predictions
        result = {}
        for k in predictions:
            # if x_0 is provided, use it instead of predictions
            # this lets us use "conditional" sampling where we condition on one modality
            # while the rest are sampled from the model
            result[k] = x_0[k] if k in x_0 else predictions[k]
            if k == 'counts':
                result[k] = randomized_round(result[k])
            
        return result
