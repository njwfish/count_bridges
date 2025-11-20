#!/usr/bin/env python3
"""
Simple analysis script for counting flows experiments.
Replaces the overly complex analyze_results.py with clean functionality.
"""

import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import argparse
import logging
import re
from utils import get_model_hash

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_metrics_files(base_dir: str = "outputs") -> List[Path]:
    """Find all metrics.yaml files in the output directory"""
    base_path = Path(base_dir)
    metrics_files = list(base_path.glob("**/metrics.yaml"))
    logger.info(f"Found {len(metrics_files)} metrics files")
    return metrics_files


def extract_training_params(metrics_path: Path) -> Dict[str, Any]:
    """Extract n_epochs and n_steps from directory structure"""
    # Look for pattern like n_steps=10,n_epochs=600
    dir_name = metrics_path.parent.name
    params = {}
    
    # Extract n_steps
    n_steps_match = re.search(r'n_steps=(\d+)', dir_name)
    if n_steps_match:
        params['n_steps'] = int(n_steps_match.group(1))
    
    # Extract n_epochs  
    n_epochs_match = re.search(r'n_epochs=(\d+)', dir_name)
    if n_epochs_match:
        params['n_epochs'] = int(n_epochs_match.group(1))
        
    return params


def get_dataset_hash(config: Dict[str, Any]) -> str:
    """Get a hash for the dataset configuration to group similar datasets"""
    from omegaconf import DictConfig, OmegaConf
    
    if not config:
        return "unknown"
    
    # Convert to DictConfig for hashing
    cfg = OmegaConf.create(config)
    
    # Use only dataset-relevant parameters for hashing
    excluded_params = [
        'bridge', 'model', 'architecture', 'training', 'slack_sampler',
        'hydra', 'defaults', 'logging', 'device', 'create_plots', 
        'experiment', 'n_steps', 'n_samples', 'seed', 'scheduler', 'averaging', 'optimizer',
    ]
    
    return get_model_hash(cfg, excluded_params)


def load_experiment_data(metrics_path: Path) -> Dict[str, Any]:
    """Load metrics and config data for a single experiment"""
    try:
        # Load metrics
        with open(metrics_path) as f:
            metrics = yaml.safe_load(f)
        
        # Try to load config from parent directory (since metrics are in subdirs like n_steps=10,n_epochs=5000)
        config_path = metrics_path.parent.parent / "config.yaml"
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
        else:
            # Fallback: try same directory in case structure is different
            config_path_alt = metrics_path.parent / "config.yaml"
            if config_path_alt.exists():
                with open(config_path_alt) as f:
                    config = yaml.safe_load(f)
        
        # Extract training parameters from directory structure
        training_params = extract_training_params(metrics_path)
        
        # Only extract numeric metrics, not config strings
        numeric_metrics = {}
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    numeric_metrics[key] = value
                elif isinstance(value, dict):
                    # Handle nested metrics (like from counts or img subdicts)
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float)):
                            numeric_metrics[f"{key}_{subkey}"] = subvalue
        
        # Extract key experiment parameters safely
        def safe_extract_target(target_dict):
            if isinstance(target_dict, dict) and '_target_' in target_dict:
                target_str = target_dict['_target_']
                if target_str:
                    return target_str.split('.')[-1]
            elif isinstance(target_dict, str):
                return target_dict.split('.')[-1]
            return 'unknown'
        
        experiment_data = {
            'experiment_path': str(metrics_path.parent),
            'dataset_type': safe_extract_target(config.get('dataset', {})),
            'bridge_type': safe_extract_target(config.get('bridge', {})),
            'model_type': safe_extract_target(config.get('model', {})),
            'seed': config.get('seed', 'unknown'),
            'dataset_hash': get_dataset_hash(config),
            'optimizer_type': safe_extract_target(config.get('optimizer', {})) + '=' + str(config.get('optimizer', {}).get('lr', 'unknown')),
            **numeric_metrics   # Include only numeric computed metrics
        }

        if 'group_size' in config['dataset']:
            experiment_data['group_size'] = config['dataset']['group_size']
        
        if 'uniform_lambda' in config['dataset']:
            experiment_data['uniform_lambda'] = config['dataset']['uniform_lambda']
        
        if 'dirichlet_concentration' in config['dataset']:
            experiment_data['dirichlet_concentration'] = config['dataset']['dirichlet_concentration']

        if 'fraction_unit_level' in config['dataset']:
            experiment_data['fraction_unit_level'] = config['dataset']['fraction_unit_level']

        if 'project_round_method' in config['model']:
            experiment_data['project_round_method'] = config['model']['project_round_method']

        if 'slack_sampler' in config and 'lam_p' in config['slack_sampler']:
            experiment_data['lam_p'] = config['slack_sampler']['lam_p']

        return experiment_data
        
    except Exception as e:
        logger.warning(f"Could not load experiment data from {metrics_path}: {e}")
        print(config)
        return None


def aggregate_results_by_dataset(base_dir: str = "outputs") -> Dict[str, pd.DataFrame]:
    """Aggregate experiment results grouped by dataset configuration"""
    metrics_files = find_metrics_files(base_dir)
    
    if not metrics_files:
        logger.error("No metrics files found!")
        return {}
    
    all_results = []
    for metrics_path in metrics_files:
        experiment_data = load_experiment_data(metrics_path)
        if experiment_data:
            all_results.append(experiment_data)
    
    if not all_results:
        logger.error("No valid experiment data found!")
        return {}
    
    df = pd.DataFrame(all_results)
    logger.info(f"Aggregated {len(df)} experiments")
    
    # Group by dataset hash
    dataset_dfs = {}
    for dataset_hash, group_df in df.groupby('dataset_hash'):
        dataset_dfs[dataset_hash] = group_df.copy()
        logger.info(f"Dataset {dataset_hash}: {len(group_df)} experiments")
    
    return dataset_dfs


def aggregate_results(base_dir: str = "outputs") -> pd.DataFrame:
    """Aggregate all experiment results into a single DataFrame (for backward compatibility)"""
    dataset_dfs = aggregate_results_by_dataset(base_dir)
    if not dataset_dfs:
        return pd.DataFrame()
    
    # Combine all datasets into one DataFrame
    all_df = pd.concat(dataset_dfs.values(), ignore_index=True)
    return all_df


def create_summary_table(df: pd.DataFrame, output_path: str = "experiment_summary.csv"):
    """Create a clean summary table"""
    if df.empty:
        logger.warning("No data to summarize")
        return
    
    # Select key columns for summary
    print(df.columns)
    summary_cols = ['dataset_type', 'bridge_type', 'model_type', 'seed']
    metric_cols = [col for col in df.columns if col not in summary_cols + ['experiment_path']]
    
    # Round numeric columns
    for col in metric_cols:
        if df[col].dtype in ['float64', 'float32']:
            df[col] = df[col].round(4)
    
    # Save full results
    df.to_csv(output_path, index=False)
    logger.info(f"Saved summary table to {output_path}")
    
    # Print top performers by primary metric
    primary_metrics = ['mean_error', 'mmd_rbf', 'wasserstein_distance']
    primary_metric = None
    for metric in primary_metrics:
        if metric in df.columns:
            primary_metric = metric
            break
    
    if primary_metric:
        print(f"\nTOP 10 EXPERIMENTS BY {primary_metric.upper()}:")
        print("=" * 80)
        top_experiments = df.nsmallest(10, primary_metric)
        display_cols = summary_cols + [primary_metric]
        available_cols = [col for col in display_cols if col in df.columns]
        print(top_experiments[available_cols].to_string(index=False))
        print()


def compare_methods_for_dataset(df: pd.DataFrame, output_path: str = "method_comparison.csv"):
    """Compare different methods within a single dataset, averaging over seeds"""
    if df.empty:
        logger.warning("Cannot compare methods - no data")
        return
    
    # Define columns that identify unique model configurations (excluding seed)
    config_cols = ['bridge_type', 'model_type', 'dataset_type', 'optimizer_type']
    if 'n_epochs' in df.columns:
        config_cols.append('n_epochs')
    if 'n_steps' in df.columns:
        config_cols.append('n_steps')
    if 'project_round_method' in df.columns and not df['project_round_method'].isna().all():
        config_cols.append('project_round_method')
    if 'fraction_unit_level' in df.columns and not df['fraction_unit_level'].isna().all():
        config_cols.append('fraction_unit_level')
    if 'lam_p' in df.columns and not df['lam_p'].isna().all():
        config_cols.append('lam_p')

    # Remove missing config columns
    config_cols = [col for col in config_cols if col in df.columns]
    
    # Identify metric columns (numeric columns that aren't config or metadata)
    metadata_cols = config_cols + ['seed', 'experiment_path', 'dataset_hash', 'optimizer_type']
    metric_cols = [col for col in df.columns 
                   if col not in metadata_cols and df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    if not metric_cols:
        logger.warning("No numeric metric columns found")
        return
    
    # Group by unique model configurations (averaging over seeds)
    df[config_cols] = df[config_cols].fillna('default')
    comparison_data = []
    for config_values, group_df in df.groupby(config_cols):
        result = {}
        
        # Store configuration
        if len(config_cols) == 1:
            result[config_cols[0]] = config_values
        else:
            for i, col in enumerate(config_cols):
                result[col] = config_values[i]
        
        # Create method name for easier identification
        method_parts = []
        if 'bridge_type' in result:
            method_parts.append(str(result['bridge_type']))
        if 'model_type' in result:
            method_parts.append(str(result['model_type']))
        result['method'] = '_'.join(method_parts) if method_parts else 'unknown'
        
        result['n_seeds'] = len(group_df['seed'].unique()) if 'seed' in group_df.columns else len(group_df)
        result['n_experiments'] = len(group_df)
        
        # Compute aggregated statistics for each metric
        for metric in metric_cols:
            if metric in group_df.columns and group_df[metric].notna().any():
                values = group_df[metric].dropna()
                if len(values) > 0:
                    result[f'{metric}_mean'] = float(np.mean(values))
                    result[f'{metric}_std'] = float(np.std(values))
                    result[f'{metric}_min'] = float(np.min(values))
                    result[f'{metric}_max'] = float(np.max(values))
        
        comparison_data.append(result)
    
    if not comparison_data:
        logger.warning("No valid method comparisons found")
        return
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Round numeric columns
    numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
    comparison_df[numeric_cols] = comparison_df[numeric_cols].round(4)
    
    comparison_df.to_csv(output_path, index=False)
    logger.info(f"Saved method comparison to {output_path}")
    
    return comparison_df


def main():
    parser = argparse.ArgumentParser(description="Simple analysis of counting flows experiments")
    parser.add_argument("--base_dir", default="outputs", help="Base directory to search for results")
    parser.add_argument("--output_dir", default="results", help="Output directory for analysis")
    
    args = parser.parse_args()
    
    # Create output directory structure
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    summary_dir = output_dir / "summary"
    comparison_dir = output_dir / "comparison"
    summary_dir.mkdir(exist_ok=True)
    comparison_dir.mkdir(exist_ok=True)
    
    print(f"Analyzing results from: {args.base_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Aggregate results by dataset
    dataset_dfs = aggregate_results_by_dataset(args.base_dir)
    
    if not dataset_dfs:
        print("No experiment results found!")
        return
    
    total_experiments = sum(len(df) for df in dataset_dfs.values())
    print(f"\nFound {total_experiments} experiments across {len(dataset_dfs)} unique dataset configurations")
    
    # Process each dataset separatel
    all_summary_files = []
    all_comparison_files = []
    
    for dataset_hash, df in dataset_dfs.items():
        print(f"\nProcessing dataset {dataset_hash} ({len(df)} experiments)...")
        
        # Get a descriptive name for this dataset
        if not df.empty and 'dataset_type' in df.columns:
            dataset_type = df['dataset_type'].iloc[0]
            # Add dimension info if available (prefer n_dimensions from metrics over data_dim from config)
            dim_info = ""
            if 'n_dimensions' in df.columns and not df['n_dimensions'].isna().all():
                dim_val = int(df['n_dimensions'].iloc[0])
                dim_info = f"_dim{dim_val}"
            elif any('data_dim' in col for col in df.columns):
                for col in df.columns:
                    if 'data_dim' in col and not df[col].isna().all():
                        dim_val = int(df[col].iloc[0])
                        dim_info = f"_dim{dim_val}"
                        break
            dataset_name = f"{dataset_type}{dim_info}_{dataset_hash[:8]}"
        else:
            dataset_name = f"dataset_{dataset_hash[:8]}"
        
        # Create summary table for this dataset
        summary_path = summary_dir / f"{dataset_name}.csv"
        create_summary_table(df, str(summary_path))
        all_summary_files.append(summary_path)
        
        # Compare methods for this dataset (averaging over seeds)
        comparison_path = comparison_dir / f"{dataset_name}.csv"
        comparison_df = compare_methods_for_dataset(df, str(comparison_path))
        if comparison_df is not None:
            all_comparison_files.append(comparison_path)
            
            # Print top methods for this dataset
            if not comparison_df.empty:
                primary_metrics = ['mean_error_mean', 'mmd_rbf_mean', 'wasserstein_distance_mean']
                primary_metric = None
                for metric in primary_metrics:
                    if metric in comparison_df.columns:
                        primary_metric = metric
                        break
                
                if primary_metric:
                    print(f"\nTOP 5 METHODS FOR {dataset_name.upper()} BY {primary_metric.upper()}:")
                    print("=" * 60)
                    top_methods = comparison_df.nsmallest(5, primary_metric)
                    display_cols = ['method', primary_metric, 'n_seeds']
                    available_cols = [col for col in display_cols if col in comparison_df.columns]
                    print(top_methods[available_cols].to_string(index=False))
    
    print(f"\nAnalysis complete!")
    print(f"- Found {total_experiments} experiments across {len(dataset_dfs)} datasets")
    print(f"- Results saved to {args.output_dir}/")
    print(f"- Summary files: {len(all_summary_files)} files in {args.output_dir}/summary/")
    print(f"- Method comparison files: {len(all_comparison_files)} files in {args.output_dir}/comparison/")
    
    # List all output files by directory
    print(f"\nOutput files created:")
    print(f"Summary files ({args.output_dir}/summary/):")
    for file_path in sorted(all_summary_files):
        print(f"  - {file_path.name}")
    print(f"Comparison files ({args.output_dir}/comparison/):")
    for file_path in sorted(all_comparison_files):
        print(f"  - {file_path.name}")


if __name__ == "__main__":
    main()


