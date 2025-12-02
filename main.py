"""
Clean Hydra-based Main Script for Count-based Flow Matching

Simple orchestration of training, evaluation, and visualization.
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
from typing import Optional

# Capture original working directory before Hydra changes it
import os
ORIGINAL_CWD = Path(os.path.join(os.path.dirname(__file__))).resolve()
print(ORIGINAL_CWD)

from visualization import plot_loss_curve, plot_model_samples, save_plots
from sample import run_sampling_evaluation


def setup_environment(cfg: DictConfig) -> str:
    """Setup logging, seeds, and device"""
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Set random seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    cp.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    # Setup device
    device = "cuda" if cfg.device == "auto" and torch.cuda.is_available() else cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logging.warning("CUDA requested but not available, falling back to CPU")
    
    logging.info(f"Using device: {device}")
    return device


def get_checkpoint_info(cfg: DictConfig, num_epochs: Optional[int] = None) -> tuple:
    """Get checkpoint path and load if exists"""
    from pathlib import Path
    from utils import get_model_hash
    
    # Get model hash excluding training and sampling params
    config_hash = get_model_hash(cfg)
    
    # Setup output directory
    output_dir = ORIGINAL_CWD / "outputs" / config_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = output_dir / "config.yaml"
    if not config_path.exists():
        with open(config_path, 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))

    if num_epochs is not None:
        epoch_checkpoint_path =  output_dir / f"model_epoch={num_epochs}.pt"
        if epoch_checkpoint_path.exists():
            checkpoint = torch.load(epoch_checkpoint_path, map_location='cpu')
            logging.info(f"Found epoch checkpoint: {epoch_checkpoint_path}")
            return output_dir, checkpoint
    
    # Check for any checkpoint
    checkpoint_path = output_dir / "model.pt"
    checkpoint = None
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logging.info(f"Found checkpoint: {checkpoint_path}")
    
    return output_dir, checkpoint


def is_training_complete(checkpoint: dict, total_epochs: int) -> bool:
    """Check if training is complete based on saved epoch info"""
    if not checkpoint:
        return False
    
    current_epoch = checkpoint.get('current_epoch', 0)
    if current_epoch > total_epochs:
        logging.warning(f"Current epoch {current_epoch} is greater than total epochs {total_epochs}")
    return current_epoch >= total_epochs


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   losses: list, current_epoch: int, cfg: DictConfig, output_dir: Path) -> None:
    """Save training checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'current_epoch': current_epoch,
        'total_epochs': cfg.training.num_epochs,
        'config': OmegaConf.to_container(cfg, resolve=True)
    }
    
    checkpoint_path = output_dir / "model.pt"
    torch.save(checkpoint, checkpoint_path)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function - simple orchestration"""
    
    # Setup environment
    device = setup_environment(cfg)
    
    # Get checkpoint info
    output_dir, checkpoint = get_checkpoint_info(cfg, num_epochs=cfg.training.num_epochs)

    logging.info(f"Output directory: {output_dir}")
    
    # Check if training is complete
    training_complete = is_training_complete(checkpoint, cfg.training.num_epochs)
    
    # Instantiate everything
    bridge = hydra.utils.instantiate(cfg.bridge)
    dataset = hydra.utils.instantiate(cfg.dataset)
    model = hydra.utils.instantiate(cfg.model)

    print(model)
    
    # Create train/eval split
    train_split = cfg.get('train_split', 0.8)  # Default 80% train, 20% eval
    train_size = int(train_split * len(dataset))
    eval_size = len(dataset) - train_size
    
    train_dataset, eval_dataset = random_split(
        dataset, [train_size, eval_size], 
        generator=torch.Generator().manual_seed(cfg.seed)  # Reproducible split
    )

    # Instantiate EMA model
    avg_model = hydra.utils.instantiate(cfg.averaging, model=model) if 'averaging' in cfg else None
    
    logging.info(f"Experiment: {cfg.experiment.name}")
    logging.info(f"Model: {type(model).__name__} ({sum(p.numel() for p in model.parameters()):,} params)")
    logging.info(f"Dataset: {len(dataset)} samples ({len(train_dataset)} train, {len(eval_dataset)} eval), {dataset.data_dim}D")
    
    
    # Training
    if not training_complete:
        logging.info("Starting training...")
        collate_fn = hydra.utils.instantiate(cfg.coupling) if 'coupling' in cfg else None
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
        scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer) if 'scheduler' in cfg else None
        trainer = hydra.utils.instantiate(
            cfg.training, output_dir=str(output_dir), collate_fn=collate_fn, optimizer=optimizer, scheduler=scheduler
        )
        
        # Load from checkpoint if available
        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if avg_model is not None and 'avg_model_state_dict' in checkpoint:
                avg_model.load_state_dict(checkpoint['avg_model_state_dict'])
            trainer.losses = checkpoint['losses']
            trainer.start_epoch = checkpoint['current_epoch']
            
            # Setup optimizer and load its state
            model = model.to(device)
            trainer.optimizer = optimizer
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None:
                trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logging.info(f"Resuming from epoch {trainer.start_epoch}")
        
        trained_model, trained_avg_model, losses = trainer.train(model=model, bridge=bridge, dataset=train_dataset, avg_model=avg_model)
    else:
        logging.info("Training already complete, loading model...")
        model.load_state_dict(checkpoint['model_state_dict'])
        if avg_model is not None:
            avg_model.load_state_dict(checkpoint['avg_model_state_dict']) 
        trained_model, trained_avg_model = model, avg_model
        losses = checkpoint['losses']
    
    # Save final model
    final_model_path = Path("final_model.pt")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'avg_model_state_dict': trained_avg_model.state_dict() if trained_avg_model else None,
        'config': OmegaConf.to_container(cfg, resolve=True),
        'losses': losses
    }, final_model_path)
    
    # Evaluation and visualization
    if cfg.get('create_plots', True):
        logging.info("Generating evaluation and plots...")
        
        # Get n_steps from config or use default
        n_steps = cfg.get('n_steps', 10)
        sum_conditioned = cfg.get('sum_conditioned', False)
        group_size = cfg['dataset'].get('group_size', None)
        n_samples = cfg.get('n_samples', 10000)
        dataset_size = len(eval_dataset)
        if group_size is not None:
            n_samples = min(n_samples // group_size, dataset_size // group_size)
        else:
            n_samples = min(n_samples, dataset_size)
        
        # Create training loss plot
        plots = {}
        plots['training_loss'] = plot_loss_curve(losses, title="Training Loss")
        
        # Save training plot immediately
        training_plots_dir = output_dir / "training_plots"
        save_plots(plots, str(training_plots_dir))
        logging.info(f"Training plots saved to: {training_plots_dir}")

        # if deconv pass collate_fn to evaluation
        if 'deconv' in cfg.model._target_:
            from training_deconv import sparse_aggregation_collate_fn
            collate_fn = sparse_aggregation_collate_fn
        else:
            collate_fn = None
        
        # Run sampling evaluation
        eval_result = run_sampling_evaluation(
            trained_model=avg_model.module.to(device) if avg_model is not None else trained_model,
            bridge=bridge,
            dataset=eval_dataset,
            output_dir=output_dir,
            n_steps=n_steps,
            n_samples=n_samples,
            n_epochs=cfg.training.num_epochs,
            sum_conditioned=sum_conditioned,
            condition_on_end_time=cfg.training.get('condition_on_end_time', False),
            collate_fn=collate_fn
        )
    
    # Final summary
    final_loss = losses[-1] if losses else float('inf')
    logging.info(f"Training completed! Final loss: {final_loss:.4f}")


if __name__ == "__main__":
    main() 