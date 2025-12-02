"""
Evaluation functions for counting flows with smart data type detection and caching
"""

import torch
import numpy as np
import logging
import pickle
import yaml
import hashlib
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Callable  

def detect_data_type(dataset: Any) -> str:
    """Detect if dataset contains vector, image, or multimodal data"""
    try:
        sample = dataset[0]
        
        # Check x_0 shape to determine data type
        x_0 = sample['x_0']
        
        if len(x_0.shape) == 1:
            return 'vector'
        elif len(x_0.shape) == 3:  # [C, H, W] or [H, W, C]
            return 'image'
        elif len(x_0.shape) == 2:
            # Could be [batch, features] or [H, W] - check size
            if x_0.shape[0] > 1 and x_0.shape[1] > 1 and max(x_0.shape) > 50:
                return 'image'  # Likely image data
            else:
                return 'vector'  # Likely vector features
        else:
            return 'multimodal'  # Complex shape, handle specially
            
    except Exception as e:
        logging.warning(f"Could not detect data type: {e}, defaulting to vector")
        return 'vector'


def hash_config(config_dict: Dict[str, Any]) -> str:
    """Create hash of config for caching"""
    config_str = str(sorted(config_dict.items()))
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def load_cached_evaluation(config_path: Path) -> Optional[Dict[str, Any]]:
    """Load cached evaluation results if they exist"""
    cache_path = config_path.parent / 'evaluation_cache.pkl'
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logging.warning(f"Could not load cached evaluation: {e}")
    return None


def save_cached_evaluation(eval_data: Dict[str, Any], true_data: Dict[str, Any], metrics: Dict[str, Any], config_path: Path):
    """Save evaluation results and metrics to cache"""
    cache_data = {
        'eval_data': eval_data,
        'true_data': true_data,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save pickle cache
    cache_path = config_path.parent / 'evaluation_cache.pkl'
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    # Save metrics as YAML for easy reading
    metrics_path = config_path.parent / 'metrics.yaml'
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
        
    logging.info(f"Saved evaluation cache to {cache_path}")
    logging.info(f"Saved metrics to {metrics_path}")


def evaluate_model(model: torch.nn.Module, bridge: Any, dataset: Any, 
                  config_path: Optional[Path] = None, n_samples: int = 1000,
                  force_regenerate: bool = False, n_steps: int = 10,
                  sum_conditioned: bool = False,
                  condition_on_end_time: bool = False,
                  collate_fn: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Smart evaluation dispatcher with caching and data type detection
    
    Args:
        model: Trained model
        bridge: Bridge object  
        dataset: Dataset to evaluate on
        config_path: Path to config for caching (optional)
        n_samples: Number of samples to generate
        force_regenerate: Force regeneration even if cache exists
        n_steps: Optional override for number of sampling steps
        
    Returns:
        Dictionary containing evaluation data and metrics
    """
    # Check cache first (skip cache if using custom n_steps)
    if config_path and not force_regenerate:
        cached_results = load_cached_evaluation(config_path)
        if cached_results:
            logging.info("Using cached evaluation results")
            return cached_results, True
    
    # Detect data type
    data_type = detect_data_type(dataset)
    logging.info(f"Detected data type: {data_type}")
    
    # Generate evaluation data 
    eval_data = generate_evaluation_data(model, bridge, dataset, n_samples, n_steps, sum_conditioned=sum_conditioned, condition_on_end_time=condition_on_end_time, collate_fn=collate_fn)
    
    # Generate true distribution data by sampling bridge directly
    true_data = generate_true_trajectory_data(bridge, dataset, n_samples, n_steps, condition_on_end_time=condition_on_end_time, collate_fn=collate_fn)
    
    # Compute evaluation metrics
    metrics = compute_evaluation_metrics(eval_data)
    
    # Add sampling info if using custom steps
    metrics['n_sampling_steps'] = n_steps
    
    # Always save metrics to config directory if config_path provided
    if config_path:
        # Save metrics as YAML for easy reading
        metrics_path = config_path.parent / 'metrics.yaml'
        with open(metrics_path, 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False)
        logging.info(f"Saved metrics to {metrics_path}")
        
        # Cache evaluation data only if using default settings
        save_cached_evaluation(eval_data, true_data, metrics, config_path)
    
    return {
        'eval_data': eval_data,
        'true_data': true_data,
        'metrics': metrics,
        'data_type': data_type
    }, False


def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, dict):
        return {k: to_numpy(x[k]) for k in x}
    if hasattr(x, 'get'):  # CuPy array
        return x.get()
    elif hasattr(x, 'cpu'):  # PyTorch tensor
        return x.cpu().numpy()
    else:
        return np.array(x)

def generate_evaluation_data(
    model: torch.nn.Module, 
    bridge: Any, 
    dataset: Any, 
    n_samples: int = 1000, 
    n_steps: int = 10,
    sum_conditioned: bool = False,
    condition_on_end_time: bool = False,
    collate_fn: Optional[Callable] = None
) -> Dict[str, np.ndarray]:
    """Generate evaluation data from trained model"""
    model.eval()
    
    # Get data efficiently using DataLoader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=n_samples, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    
    if isinstance(batch['x_0'], dict):
        x0_target = {k: batch['x_0'][k].squeeze(0).numpy() for k in batch['x_0']}
        x1_source = {k: batch['x_1'][k].squeeze(0).numpy() for k in batch['x_1']}

        x0_torch = {k: torch.tensor(x0_target[k]).cuda() for k in x0_target}
        x1_torch = {k: torch.tensor(x1_source[k]).cuda() for k in x1_source}
    else:
        x0_target = batch['x_0'].squeeze(0).numpy()# .reshape(-1, batch['x_0'].shape[-1])
        x1_source = batch['x_1'].squeeze(0).numpy()# .reshape(-1, batch['x_1'].shape[-1])n

        # Convert to torch tensors only for bridge call
        x0_torch = torch.tensor(x0_target).cuda()
        x1_torch = torch.tensor(x1_source).cuda()
    
    
    with torch.no_grad():
        # Prepare arguments for reverse sampler
        sampler_kwargs = {
            'x_1': x1_torch,
            'z': {},
            'model': model.to('cuda'),
            'return_trajectory': True,
            'return_x_hat': True,
        }
        # if 'z' in batch:
        # z = batch['z'].squeeze(0).numpy()# .reshape(-1, batch['z'].shape[-1])
        sampler_kwargs['z'] = {k: batch[k].squeeze(0).cuda() for k in batch if k not in ['x_0', 'x_1', 'X_0', 'group_sizes', 'A']}
        
        # Add target mean for mean constrained bridges
        if sum_conditioned:
            batch_X_0 = batch['X_0']
            if isinstance(batch_X_0, dict):
                batch_X_0 = {k: batch_X_0[k].squeeze(0).cuda() for k in batch_X_0}
            else:
                batch_X_0 = batch_X_0.squeeze(0).cuda()
            sampler_kwargs['z']['target_sum'] = batch_X_0
            sampler_kwargs['z']['A'] = batch['A'].cuda()
            if isinstance(batch['x_0'], dict) and 'X_0' in batch:
                # this lets us condition on x_0 if it is provided (and not in X_0)
                # using the fact that the model "samples" from x_0 if x_0 is provided, e.g. "conditional" sampling
                # across modalities 
                context_x_0 = {}
                for k in batch['x_0']:
                    if k not in batch['X_0']:
                        context_x_0[k] = batch['x_0'][k].cuda()

                sampler_kwargs['z']['x_0'] = context_x_0

                if condition_on_end_time:
                    sampler_kwargs['start_times'] = {k: 0.0 for k in context_x_0}
                    sampler_kwargs['x_start_time'] = {k: context_x_0[k] for k in context_x_0}
            logging.info("Using mean constrained bridge with target mean")
        
        # Add n_steps to sampler
        sampler_kwargs['n_steps'] = n_steps
            
        # Generate samples with trajectories
        result = bridge.sampler(**sampler_kwargs)
        
        # Our bridge sampler always returns (x0_generated, x_trajectory, x_hat_trajectory)
        x0_generated, x_trajectory, x_hat_trajectory = result
        
        x0_generated = to_numpy(x0_generated)
        x_trajectory = to_numpy(x_trajectory)
        x_hat_trajectory = to_numpy(x_hat_trajectory)
    
    result_dict = {
        'x0_target': x0_target,
        'x1_batch': x1_source,
        'x0_generated': x0_generated,
    }

    if sum_conditioned:
        result_dict['target_sum'] = batch['X_0'].numpy()
        result_dict['A'] = batch['A'].numpy()
    
    # Only include trajectories if they exist
    if x_trajectory is not None:
        result_dict['x_trajectory'] = x_trajectory
    else:
        # Create minimal trajectory from start and end points
        result_dict['x_trajectory'] = np.array([x1_source, x0_generated])
        
    if x_hat_trajectory is not None:
        result_dict['x_hat_trajectory'] = x_hat_trajectory
    else:
        # Create minimal x_hat trajectory (just final prediction)
        result_dict['x_hat_trajectory'] = np.array([x0_generated])
        

    
    return result_dict


def compute_evaluation_metrics(eval_data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics using new metrics module"""
    from metrics import compute_comprehensive_metrics
    
    
    # Add basic compatibility metrics
    x0_generated = eval_data['x0_generated']
    x0_target = eval_data['x0_target']

    if isinstance(x0_generated, dict):
        metrics = {}
        gen_mean = {}
        target_mean = {}
        for k in x0_generated:
            eval_data_k = {
                data_key: eval_data[data_key][k] for data_key in eval_data.keys()
            }
            metrics[k] = compute_comprehensive_metrics(eval_data_k)
            gen_mean[k] = np.mean(x0_generated[k], axis=0)
            target_mean[k] = np.mean(x0_target[k], axis=0)
    else:
        metrics = compute_comprehensive_metrics(eval_data)
        gen_mean = np.mean(x0_generated, axis=0)
        target_mean = np.mean(x0_target, axis=0)

    if isinstance(x0_generated, dict):
        metrics['target_mean'] = {k: target_mean[k].tolist() for k in target_mean}
        metrics['gen_mean'] = {k: gen_mean[k].tolist() for k in gen_mean}
    else:
        metrics['target_mean'] = target_mean.tolist()
        metrics['gen_mean'] = gen_mean.tolist()
    
    return metrics


def generate_true_trajectory_data(bridge: Any, dataset: Any, n_samples: int = 1000, n_steps: int = 10, condition_on_end_time: bool = False, collate_fn: Optional[Callable] = None) -> Dict[str, np.ndarray]:
    """
    Generate true trajectory data by sampling bridge directly at each time step.
    This is completely bridge-agnostic and doesn't require any dummy models.
    """
    from torch.utils.data import DataLoader
    
    # Get data from dataset
    dataloader = DataLoader(dataset, batch_size=n_samples, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    
    if isinstance(batch['x_0'], dict):
        x0_true = {k: batch['x_0'][k].squeeze(0).numpy() for k in batch['x_0']}
        x1_true = {k: batch['x_1'][k].squeeze(0).numpy() for k in batch['x_1']}

        x0_torch = {k: torch.tensor(x0_true[k]).cuda() for k in x0_true}
        x1_torch = {k: torch.tensor(x1_true[k]).cuda() for k in x1_true}

        x_trajectory = {k: [] for k in x0_true}
        x_hat_trajectory = {k: [] for k in x0_true}
    else:
        x0_true = batch['x_0'].squeeze(0).numpy()# .reshape(-1, batch['x_0'].shape[-1])
        x1_true = batch['x_1'].squeeze(0).numpy()# .reshape(-1, batch['x_1'].shape[-1])
    
        # Convert to torch tensors
        x0_torch = torch.tensor(x0_true).cuda() if torch.cuda.is_available() else torch.tensor(x0_true)
        x1_torch = torch.tensor(x1_true).cuda() if torch.cuda.is_available() else torch.tensor(x1_true)

        x_trajectory = []
        x_hat_trajectory = []
    
    
    # Create time points from 1 to 0
    times = np.linspace(0.0, 1.0, n_steps + 1)

    with torch.no_grad():
        for i, t in enumerate(times):
            # Sample x_t from bridge at time t
            t, x_t, _ = bridge(x1_torch, x0_torch, t)
            if isinstance(x_t, dict):
                for k in x_t:
                    x_trajectory[k].append(to_numpy(x_t[k]))
                    x_hat_trajectory[k].append(to_numpy(x0_torch[k]))
            else:
                x_trajectory.append(to_numpy(x_t))
                x_hat_trajectory.append(to_numpy(x0_torch))

    
    # Convert to numpy arrays
    if isinstance(x_trajectory, dict):
        x_trajectory = {k: np.array(x_trajectory[k]) for k in x_trajectory}
        x_hat_trajectory = {k: np.array(x_hat_trajectory[k]) for k in x_hat_trajectory}
    else:
        x_trajectory = np.array(x_trajectory)  # [T, B, d]
        x_hat_trajectory = np.array(x_hat_trajectory)  # [T, B, d]
    
    return {
        'x0_target': x0_true,
        'x1_batch': x1_true,
        'x0_generated': x0_true,  # True data is the target
        'x_trajectory': x_trajectory,
        'x_hat_trajectory': x_hat_trajectory,
    }

# Sampling sweep functionality moved to run_sampling_sweep.py

def log_evaluation_summary(eval_data: Dict[str, np.ndarray], metrics: Dict[str, float]) -> None:
    """Log evaluation summary - dynamically iterate over available metrics"""
    logging.info("=== Evaluation Summary ===")
    
    # Log basic info first
    if 'data_type' in metrics:
        logging.info(f"Data type: {metrics['data_type']}")
    if 'n_samples' in metrics:
        logging.info(f"Samples: {metrics['n_samples']}")
    if 'n_dimensions' in metrics:
        logging.info(f"Dimensions: {metrics['n_dimensions']}")
    if 'image_shape' in metrics:
        logging.info(f"Image shape: {metrics['image_shape']}")
    
    # Log all other metrics dynamically
    skip_keys = {'data_type', 'n_samples', 'n_dimensions', 'image_shape', 'target_mean', 'gen_mean'}
    for metric_name, value in metrics.items():
        if metric_name not in skip_keys:
            if isinstance(value, float) and not np.isnan(value):
                logging.info(f"{metric_name}: {value:.4f}")
            elif isinstance(value, int):
                logging.info(f"{metric_name}: {value}")
    
    logging.info(f"Generated {len(eval_data['x0_generated'])} samples") 