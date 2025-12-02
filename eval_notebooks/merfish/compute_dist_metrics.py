"""
Compute metrics directly from original objects without saving intermediates.
This script replicates the evaluation from dist_metrics.ipynb but works directly
with the original S1R1.npz and predicted counts.
"""
import argparse
import numpy as np
import pickle as pkl
import torch
import pandas as pd
import sys
import os

# Add counting_flows_ root directory to path to import metrics
script_dir = os.path.dirname(os.path.abspath(__file__))
# From analysis/merfish/ to counting_flows_ root: go up 2 levels
counting_flows_root = os.path.join(script_dir, '../..')
counting_flows_root = os.path.abspath(counting_flows_root)
sys.path.insert(0, counting_flows_root)

from metrics import compute_comprehensive_metrics


def convert_predicted_to_s1_format(s1_data, predicted_counts_list, max_img_size=256):
    """
    Convert predicted counts to S1 format in memory.
    
    Args:
        s1_data: Loaded S1R1.npz data
        predicted_counts_list: List of torch tensors (or single tensor) with predicted counts
        max_img_size: Maximum image size to filter cells
    
    Returns:
        dict with 'counts' and 'spots' keys matching S1 format
    """
    # Stack predicted counts if it's a list
    if isinstance(predicted_counts_list, list):
        combined_predicted_counts = torch.vstack(predicted_counts_list)
    else:
        combined_predicted_counts = predicted_counts_list
    
    # Convert to numpy if needed
    if isinstance(combined_predicted_counts, torch.Tensor):
        combined_predicted_counts_np = combined_predicted_counts.cpu().numpy()
    else:
        combined_predicted_counts_np = combined_predicted_counts
    
    # Get original spots structure
    imgs = s1_data['imgs'][:]
    original_spots = s1_data['spots']
    n_spots = len(original_spots)
    
    # Create new spots array with consecutive indices, filtering by image size
    new_spots = []
    current_index = 0
    cells_to_remove = []
    total_cells = 0
    
    for i in range(n_spots):
        # Get number of cells in this spot from original data (filtering by image size)
        num_cells_in_spot = 0
        for cell in original_spots[i]:
            if max(imgs[cell].shape) > max_img_size:
                cells_to_remove.append(cell)
            else:
                num_cells_in_spot += 1
        total_cells += num_cells_in_spot
        
        # Create array of consecutive indices for this spot
        spot_indices = np.arange(current_index, current_index + num_cells_in_spot)
        new_spots.append(spot_indices)
        
        # Update current index for next spot
        current_index += num_cells_in_spot
    
    # Trim predicted counts to match filtered cells
    combined_predicted_counts_np = combined_predicted_counts_np[:total_cells]
    
    return {
        'counts': combined_predicted_counts_np,
        'spots': new_spots
    }


def evaluate_predictions(true_data, predicted_data_dict, max_img_size=256):
    """
    Evaluate predicted counts against true counts and spot means.
    
    Args:
        true_data: npz file with true data
        predicted_data_dict: dict with 'counts' and 'spots' keys
        max_img_size: maximum image size to consider (default 256)
    
    Returns:
        tuple of (metrics_pred_vs_true, metrics_pred_vs_spot_mean, metrics_true_vs_spot_mean)
    """
    true_counts = true_data['counts']
    predicted_counts = predicted_data_dict['counts']
    true_spots = true_data['spots']
    predicted_spots = predicted_data_dict['spots']
    imgs = true_data['imgs'][:]
    
    n_spots = len(true_spots)
    
    # Store MSE results for each spot
    metrics_pred_vs_true = {}
    metrics_pred_vs_spot_mean = {}
    metrics_true_vs_spot_mean = {}
    metrics = ['energy_distance', 'wasserstein_distance', 'mmd_rbf', 'mse']
    for metric in metrics:
        metrics_pred_vs_true[metric] = []
        metrics_pred_vs_spot_mean[metric] = []
        metrics_true_vs_spot_mean[metric] = []
    valid_spots = 0
    n_cells = []
    
    print(f"Evaluating {n_spots} spots...")
    
    for spot_idx in range(n_spots):
        if spot_idx % 500 == 0:
            print(f"Processing spot {spot_idx}/{n_spots}")
        
        # Get cell indices for this spot
        true_cell_indices = np.array(true_spots[spot_idx]).astype(int)
        predicted_cell_indices = np.array(predicted_spots[spot_idx]).astype(int)
        
        # Get counts for this spot
        true_spot_counts = true_counts[true_cell_indices]
        predicted_spot_counts = predicted_counts[predicted_cell_indices]
        
        # Filter cells based on image size (only for true data since predicted is already filtered)
        valid_cells = []
        for i, cell_idx in enumerate(true_cell_indices):
            cell_img = imgs[cell_idx]
            if max(cell_img.shape) <= max_img_size:
                valid_cells.append(i)
        if len(valid_cells) == 0:
            continue  # Skip spots with no valid cells
        n_cells.append(len(valid_cells))
        
        # Filter true counts to only valid cells (predicted is already filtered)
        true_spot_counts_valid = true_spot_counts[valid_cells]
        predicted_spot_counts_valid = predicted_spot_counts  # Already filtered

        # Calculate spot mean from true counts (filtered)
        spot_mean = np.mean(true_spot_counts_valid, axis=0)
        
        # Calculate MSE: predicted vs true
        mse_pred_true = np.mean((predicted_spot_counts_valid - true_spot_counts_valid) ** 2)
        metrics_pred_vs_true['mse'].append(mse_pred_true)
        
        # Calculate MSE: predicted vs spot mean
        mse_pred_mean = np.mean((predicted_spot_counts_valid - spot_mean) ** 2)
        metrics_pred_vs_spot_mean['mse'].append(mse_pred_mean)
        
        # Calculate MSE: true vs spot mean
        mse_true_mean = np.mean((true_spot_counts_valid - spot_mean) ** 2)
        metrics_true_vs_spot_mean['mse'].append(mse_true_mean)
        
        valid_spots += 1

        # Compute comprehensive metrics: predicted vs true
        d = {}
        d['x0_target'] = true_spot_counts_valid
        d['x0_generated'] = predicted_spot_counts_valid
        res_true_vs_predicted = compute_comprehensive_metrics(d)
        for metric in metrics:
            if metric == 'mse':
                continue
            metrics_pred_vs_true[metric].append(res_true_vs_predicted[metric])

        # Compute comprehensive metrics: predicted vs spot mean
        d = {}
        d['x0_target'] = np.tile(spot_mean, (len(predicted_spot_counts_valid), 1))
        d['x0_generated'] = predicted_spot_counts_valid
        res_pred_vs_spot_mean = compute_comprehensive_metrics(d)
        for metric in metrics:
            if metric == 'mse':
                continue
            metrics_pred_vs_spot_mean[metric].append(res_pred_vs_spot_mean[metric])

        # Compute comprehensive metrics: true vs spot mean
        d = {}
        d['x0_target'] = true_spot_counts_valid
        d['x0_generated'] = np.tile(spot_mean, (len(true_spot_counts_valid), 1))
        res_true_vs_spot_mean = compute_comprehensive_metrics(d)
        for metric in metrics:
            if metric == 'mse':
                continue
            metrics_true_vs_spot_mean[metric].append(res_true_vs_spot_mean[metric])
    
    return metrics_pred_vs_true, metrics_pred_vs_spot_mean, metrics_true_vs_spot_mean


def main():
    parser = argparse.ArgumentParser(description="Compute metrics directly from original objects")
    parser.add_argument("--reference", required=True, help="Path to reference S1R1.npz")
    parser.add_argument("--predicted", required=True, help="Path to predicted counts (pkl file with list of tensors)")
    parser.add_argument("--max-img-size", type=int, default=256, help="Maximum image size to filter cells")
    parser.add_argument("--output", help="Output CSV path for metrics summary (optional)")
    args = parser.parse_args()
    
    # Load original data
    print("Loading reference data...")
    s1_data = np.load(args.reference, allow_pickle=True)
    
    # Load predicted counts
    print("Loading predicted counts...")
    with open(args.predicted, 'rb') as f:
        predicted_counts = pkl.load(f)
    
    # Convert predicted counts to S1 format (in memory)
    print("Converting predicted counts to S1 format...")
    predicted_data = convert_predicted_to_s1_format(s1_data, predicted_counts, args.max_img_size)
    
    print(f"Converted to {len(predicted_data['spots'])} spots")
    print(f"Total cells: {predicted_data['counts'].shape[0]}")
    print(f"Counts shape: {predicted_data['counts'].shape}")
    
    # Compute metrics
    print("Starting evaluation...")
    metrics_pred_vs_true, metrics_pred_vs_spot_mean, metrics_true_vs_spot_mean = evaluate_predictions(
        s1_data, predicted_data, args.max_img_size
    )
    
    # Print results
    metrics_to_show = ['energy_distance', 'wasserstein_distance', 'mmd_rbf']
    rows = []
    for metric in metrics_to_show:
        rows.append({
            "Metric": metric,
            "Pred vs True": np.mean(metrics_pred_vs_true[metric]),
            "Pred vs Spot Mean": np.mean(metrics_pred_vs_spot_mean[metric]),
            "True vs Spot Mean": np.mean(metrics_true_vs_spot_mean[metric]),
        })
    
    df = pd.DataFrame(rows)
    df.set_index('Metric', inplace=True)
    df = df.T
    print("\nResults:")
    print(df.to_string(float_format="%.3f"))
    
    # Save if output path provided
    if args.output:
        df.to_csv(args.output)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()

