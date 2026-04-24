"""
Sampling, evaluation, and plotting script for counting flows.
Clean Hydra-based CLI for generating samples with specified n_steps.
"""

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
import numpy as np
import pickle
import cupy as cp
from torch.utils.data import random_split
from typing import Dict, Any, Optional, Callable
import os

# Capture original working directory before Hydra changes it
ORIGINAL_CWD = Path.cwd().resolve()

from visualization import plot_loss_curve, plot_model_samples, save_plots
from evaluate import evaluate_model

def run_sampling_evaluation(
    trained_model: torch.nn.Module,
    bridge: Any,
    dataset: Any,
    output_dir: Path,
    n_steps: int = 10,
    n_samples: int = 10000,
    n_epochs: Optional[int] = None,
    sum_conditioned: bool = False,
    condition_on_end_time: bool = False,
    collate_fn: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Run sampling evaluation with specified parameters.
    
    Args:
        trained_model: Trained model
        bridge: Bridge object
        dataset: Evaluation dataset
        output_dir: Directory to save results
        n_steps: Number of sampling steps
        n_samples: Number of samples to generate
        n_epochs: Number of epochs to load
    Returns:
        Dictionary with evaluation results
    """
    logging.info(f"Running sampling evaluation with {n_steps} steps...")
    results_dir = output_dir
    if sum_conditioned:
        results_dir = results_dir / "sum_conditioned"
        results_dir.mkdir(exist_ok=True)

    # Include bridge sampler settings in subdir name so different (mode, eta)
    # evaluations of the same trained checkpoint don't clobber each other.
    suffix_parts = [f"n_steps={n_steps},n_epochs={n_epochs}"]
    sampler_mode = getattr(bridge, "mode", None)
    sampler_eta = getattr(bridge, "eta", None)
    if sampler_mode is not None:
        suffix_parts.append(f"mode={sampler_mode}")
    if sampler_eta is not None:
        suffix_parts.append(f"eta={sampler_eta}")
    results_dir = results_dir / ",".join(suffix_parts)
    results_dir.mkdir(exist_ok=True)
    
    # Run evaluation
    eval_result, cached = evaluate_model(
        trained_model, 
        bridge, 
        dataset,
        config_path=results_dir / "config.yaml",
        n_samples=n_samples,
        n_steps=n_steps,
        sum_conditioned=sum_conditioned,
        condition_on_end_time=condition_on_end_time,
        collate_fn=collate_fn
    )
    
    eval_data = eval_result['eval_data']
    true_data = eval_result['true_data']
    metrics = eval_result['metrics']
    
    # Log results
    logging.info("=== Sampling Evaluation Summary ===")
    logging.info(f"Steps: {n_steps}")
    logging.info(f"Samples: {n_samples}")
    logging.info(f"Epochs: {n_epochs}")
    for metric_name, value in metrics.items():
        if metric_name not in {'data_type', 'n_samples', 'n_dimensions', 'image_shape', 'target_mean', 'gen_mean'}:
            if isinstance(value, float) and not np.isnan(value):
                logging.info(f"{metric_name}: {value:.4f}")
    
    if not cached:
        # Save evaluation data
        eval_data_path = results_dir / "eval_data.pkl"
        with open(eval_data_path, 'wb') as f:
            pickle.dump(eval_data, f)
    
    # Generate and save plots
    plots_dir = results_dir / "plots"
    if plots_dir.exists() and any(plots_dir.glob("*.png")):
        logging.info(f"Plots already exist at {plots_dir}, skipping plot generation")
    else:
        # Generate model evaluation plots with true distributions
        if isinstance(eval_data['x0_generated'], dict):
            
            for k in eval_data['x0_generated']:
                plots = {}
                plots_dir = results_dir / f"plots/{k}"
                os.makedirs(plots_dir, exist_ok=True)
                eval_data_k = {
                    data_key: eval_data[data_key][k] for data_key in eval_data.keys()
                }
                true_data_k = {
                    data_key: true_data[data_key][k] for data_key in true_data.keys()
                }
                sample_figs = plot_model_samples(eval_data_k, title=f"Samples (n_steps={n_steps})", true_data=true_data_k)
                plots.update(sample_figs)
                save_plots(plots, str(plots_dir))
                logging.info(f"Plots saved to: {plots_dir}")
        else:
            plots = {}
            sample_figs = plot_model_samples(eval_data, title=f"Samples (n_steps={n_steps})", true_data=true_data)
            plots.update(sample_figs)
        
            # Save plots
            save_plots(plots, str(plots_dir))
            logging.info(f"Plots saved to: {plots_dir}")
    
    return eval_result
