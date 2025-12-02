"""
Generate CTP matrix directly from original objects without saving intermediates.
This script is similar to generate_ctp_count_bridge.py but works directly
with the original S1R1.npz and predicted counts.
"""
import argparse
import os
import numpy as np
import pickle as pkl
import torch
from collections import Counter
from sklearn.neighbors import NearestNeighbors
import scanpy as sc
import anndata as ad
import pandas as pd


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
    total_cells = 0
    
    for i in range(n_spots):
        # Get number of cells in this spot from original data (filtering by image size)
        num_cells_in_spot = 0
        for cell in original_spots[i]:
            if max(imgs[cell].shape) <= max_img_size:
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


def compute_ctp(spot_annotations, num_celltypes):
    """Compute cell type proportion matrix from spot annotations."""
    n_spots = len(spot_annotations)
    ctp = np.zeros((n_spots, num_celltypes), dtype=float)
    for i in range(n_spots):
        labels = np.asarray(spot_annotations[i])
        total = labels.size if hasattr(labels, 'size') else len(labels)
        if total == 0:
            continue
        for cid in range(num_celltypes):
            ctp[i, cid] = (labels == cid).sum() / float(total)
    return ctp


def majority_vote_from_indices(indices, reference_annotations):
    """Get majority vote predictions from neighbor indices."""
    neighbor_anns = reference_annotations[indices]
    preds = []
    for row in neighbor_anns:
        cnt = Counter(row)
        preds.append(cnt.most_common(1)[0][0])
    return np.array(preds)


def main():
    parser = argparse.ArgumentParser(description="Generate CTP matrix from original objects")
    parser.add_argument("--reference", required=True, help="Path to reference S1R1.npz (with counts, annotations, spots)")
    parser.add_argument("--predicted", required=True, help="Path to predicted counts (pkl file with list of tensors)")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--max-img-size", type=int, default=256, help="Maximum image size to filter cells")
    parser.add_argument("--output", help="Output CSV path for single run")
    parser.add_argument("--output-dir", help="Output directory when running multiple cell type counts", 
                       default="/orcd/data/omarabu/001/tanush20/counting_flows_/MERFISH/outputs/")
    parser.add_argument("--num-celltypes", type=int, help="Total number of cell types (columns) for single run")
    parser.add_argument("--num-celltypes-list", type=str, 
                       help="Comma-separated list of cell type counts to iterate over, e.g. 5,10,15")
    parser.add_argument("--count-bridge-version", type=str, default="new_1", help="suffix name to add after outputs")
    args = parser.parse_args()
    
    # Load original reference data
    print("Loading reference data...")
    true = np.load(args.reference, allow_pickle=True)
    
    # Load predicted counts
    print("Loading predicted counts...")
    with open(args.predicted, 'rb') as f:
        predicted_counts = pkl.load(f)
    
    # Convert predicted counts to S1 format (in memory)
    print("Converting predicted counts to S1 format...")
    predicted_data = convert_predicted_to_s1_format(true, predicted_counts, args.max_img_size)
    
    # Get counts
    reference_counts = true["counts"].copy()
    predicted_counts_array = predicted_data['counts'].copy()
    
    # Preprocess reference with scanpy
    print("Preprocessing reference data...")
    ref_adata = ad.AnnData(reference_counts)
    sc.pp.filter_genes(ref_adata, min_cells=3)
    ref_adata.layers["counts"] = ref_adata.X.copy()
    sc.pp.normalize_total(ref_adata)
    sc.pp.log1p(ref_adata)
    sc.tl.pca(ref_adata)
    ref_pca = ref_adata.obsm["X_pca"]
    
    # Preprocess predicted counts with the same steps (no re-fitting filtering)
    print("Preprocessing predicted data...")
    query_adata = ad.AnnData(predicted_counts_array)
    query_adata.layers["counts"] = query_adata.X.copy()
    sc.pp.normalize_total(query_adata)
    sc.pp.log1p(query_adata)
    
    # Project predicted with PCA fitted on reference
    from sklearn.decomposition import PCA
    pca = PCA(n_components=ref_pca.shape[1])
    pca.fit(ref_adata.X)
    query_pca = pca.transform(query_adata.X)
    
    print(f"Reference PCA shape: {ref_pca.shape}, Query PCA shape: {query_pca.shape}")
    
    # Fit NN on reference PCA and prepare neighbor indices once
    print("Fitting nearest neighbors model...")
    nn_model = NearestNeighbors(n_neighbors=args.n_neighbors, metric="euclidean")
    nn_model.fit(ref_pca)
    distances, indices = nn_model.kneighbors(query_pca)
    
    # Predictions will be computed below per configuration of cell types
    # Decide single vs multiple runs
    if args.num_celltypes_list:
        assert args.output_dir, "--output-dir is required when using --num-celltypes-list"
        os.makedirs(args.output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.reference))[0]
        values = [int(x) for x in args.num_celltypes_list.split(',') if x.strip()]
        for k in values:
            print(f"Processing k={k}...")
            reference_annotations = np.array(true["annotations"].item()[k])
            all_preds = majority_vote_from_indices(indices, reference_annotations)
            # Organize by spots again to keep purity per K
            spot_annotations = []
            j = 0
            for arr in predicted_data['spots']:
                num = len(arr)
                spot_annotations.append(all_preds[j:j + num])
                j += num
            ctp = compute_ctp(spot_annotations, k)
            out_path = os.path.join(args.output_dir, f"S1R1_{k}/count_bridge_{args.count_bridge_version}.csv")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            # Create DataFrame with spot indices as row index
            df = pd.DataFrame(ctp, index=range(len(ctp)))
            df.to_csv(out_path, index=True)
            print(f"Saved CTP matrix {ctp.shape} to {out_path}")
    else:
        assert args.num_celltypes is not None and args.output, \
            "Provide --num-celltypes and --output for single run"
        k = int(args.num_celltypes)
        print(f"Processing k={k}...")
        reference_annotations = np.array(true["annotations"].item()[k])
        all_preds = majority_vote_from_indices(indices, reference_annotations)
        spot_annotations = []
        j = 0
        for arr in predicted_data['spots']:
            num = len(arr)
            spot_annotations.append(all_preds[j:j + num])
            j += num
        ctp = compute_ctp(spot_annotations, k)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        # Create DataFrame with spot indices as row index
        df = pd.DataFrame(ctp, index=range(len(ctp)))
        df.to_csv(args.output, index=True)
        print(f"Saved CTP matrix {ctp.shape} to {args.output}")


if __name__ == "__main__":
    main()

