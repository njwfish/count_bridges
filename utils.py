import torch
import os   
import logging
import hashlib
import json
from typing import Optional, List
from omegaconf import DictConfig, OmegaConf

# Enable JIT compilation for better performance
# TODO: We are using DDP 
torch._dynamo.config.optimize_ddp = False
torch._dynamo.config.suppress_errors = True

USE_COMPILE = False
try:

    if os.environ.get("USE_COMPILE") == "1":
        USE_COMPILE = True
 
except Exception as e:
    USE_COMPILE = False

logger = logging.getLogger(__name__)

logger.info(f"Using compile: {USE_COMPILE}")

# Try to use torch.compile if available (PyTorch 2.0+)
def maybe_compile(func):

    """Decorator to optionally compile functions for better performance."""
    if USE_COMPILE and hasattr(torch, 'compile'):
        try:
            return torch.compile(func, mode='max-autotune')
        except:
            pass
    return func


def get_model_hash(cfg: DictConfig, excluded_params: Optional[List[str]] = None) -> str:
    """
    Get model hash excluding training and sampling parameters.
    Only includes model-relevant configuration for consistent model identification.
    """
    # Create config hash (excluding training and sampling params)
    config_copy = OmegaConf.to_container(cfg, resolve=True)
    
    # Remove entire training section (not model-relevant)
    config_copy.pop('training', None)
    
    # Remove other non-model-relevant sections and parameters
    if excluded_params is None:
        excluded_params = []
    excluded_params = [
        # Hydra/experiment management
        'hydra', 'defaults', 'logging',
        # Runtime parameters
        'device', 'create_plots',
        # Sampling parameters
        'n_steps', 'n_samples', 'sum_conditioned'
    ] + excluded_params

    for param in excluded_params:
        config_copy.pop(param, None)

    # Drop nested sampler-only params that don't affect training or the
    # trained weights, so the same trained checkpoint can be re-used under
    # different reverse samplers. Add more entries here as needed.
    sampler_only_nested = [
        # Unified (mode, eta) sampler controls -- sampler-only, not model.
        ('bridge', 'mode'),
        ('bridge', 'eta'),
        # Legacy: kept for backward compat with older checkpoints.
        ('bridge', 'stochastic'),
        # AdaptiveGaussianBridge RoU / quadrature knobs -- sampler-only.
        ('bridge', 'mode_steps'),
        ('bridge', 'side_steps'),
        ('bridge', 'rou_rounds'),
        ('bridge', 'safety'),
        ('bridge', 'gl_n'),
        ('bridge', 't_eps'),
        # ClockedGaussianBridge sampler-only knobs.
        ('bridge', 'gig_n_rounds'),
        ('bridge', 'gig_safety'),
        ('bridge', 'n_quad'),
    ]
    for section, key in sampler_only_nested:
        if section in config_copy and isinstance(config_copy[section], dict):
            config_copy[section].pop(key, None)
    
    config_str = json.dumps(config_copy, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]