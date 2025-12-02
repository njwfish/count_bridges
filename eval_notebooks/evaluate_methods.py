#!/usr/bin/env python3
"""
Evaluate methods across multiple cell type counts and plot metric curves.

Inputs:
- --celltypes-list: comma-separated list like 5,10,15
- --outputs-dir: directory containing per-k subfolders like <basename>_5, <basename>_10, ...
- --ground-truth-dir: directory containing ground-truth CSVs named like *_<k>.csv (e.g., S1R1_5.csv)

For each k, this script:
  - loads ground truth CTP (*_<k>.csv) from ground-truth-dir
  - scans outputs-dir/<*>_<k>/ for *.csv predictions
  - computes metrics (JSD scipy, JSD, RMSE)
Aggregates metrics over k and plots line charts (x=k, y=metric) per method.
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import evaluation functions
from eval_metrics import (
    jsd_ct,
    rmse_per_celltype,
    per_celltype_pearson,
    per_celltype_spearman,
    mae_per_celltype,
    mse_per_celltype
)


def load_csv_as_array(csv_path):
    """Load CSV file and return as numpy array and row indices."""
    df = pd.read_csv(csv_path, index_col=0)
    return df.values, df.index.values


def main():
    parser = argparse.ArgumentParser(description="Evaluate methods across cell type counts and plot metrics")
    parser.add_argument("--celltypes-list", help="Comma-separated list of cell type counts, e.g. 5,10,15. If not provided, assumes no cell type suffix.")
    parser.add_argument("--outputs-dir", required=True, help="Directory containing per-k subfolders like <basename>_k with method CSVs")
    parser.add_argument("--ground-truth-dir", help="Directory containing ground-truth CSVs named like *_<k>.csv (ignored if --ground-truth-csv is provided)")
    parser.add_argument("--ground-truth-csv", help="Path to a single ground-truth CSV; when provided, runs a single evaluation without k suffix")
    parser.add_argument("--figs-out", help="Directory to write metric plots. If not provided, no plots will be generated.")
    parser.add_argument("--basename", help="Optional basename to match, e.g., S1R1 (filters *_<k> and *_<k>.csv)")
    args = parser.parse_args()

    if args.ground_truth_csv:
        # Force single evaluation with no suffix when CSV is directly provided
        ks = [None]
        if args.celltypes_list:
            print("[INFO] --ground-truth-csv provided; ignoring --celltypes-list and running single evaluation without suffix.")
    else:
        if args.celltypes_list:
            ks = [int(x) for x in args.celltypes_list.split(',') if x.strip()]
        else:
            ks = [None]  # No cell type suffix
    
    if args.figs_out:
        os.makedirs(args.figs_out, exist_ok=True)

    # metrics_by_method: method -> metric arrays aligned with ks
    metrics_by_method = {}

    for k in ks:
        # locate ground truth file
        if args.ground_truth_csv:
            true_csv_path = args.ground_truth_csv
        else:
            if k is None:
                # No cell type suffix
                pattern = f"{args.basename}.csv" if args.basename else "*.csv"
            else:
                pattern = f"{args.basename}_{k}.csv" if args.basename else f"*_{k}.csv"
            gt_candidates = glob.glob(os.path.join(args.ground_truth_dir, pattern))
            if not gt_candidates:
                print(f"[WARN] No ground truth file found for k={k}; skipping")
                continue
            true_csv_path = gt_candidates[0]
        true_array, true_row_indices = load_csv_as_array(true_csv_path)

        print(f"Loaded ground truth file: {true_csv_path}")

        # locate outputs subdir for this k
        if k is None:
            # No cell type suffix - use outputs-dir/basename
            if args.basename:
                k_dir = os.path.join(args.outputs_dir, args.basename)
            else:
                k_dir = args.outputs_dir
        else:
            kdir_pattern = f"{args.basename}_{k}" if args.basename else f"*_{k}"
            k_dir_candidates = [d for d in glob.glob(os.path.join(args.outputs_dir, kdir_pattern)) if os.path.isdir(d)]
            if not k_dir_candidates:
                print(f"[WARN] No outputs subdir found for k={k}; skipping")
                continue
            k_dir = k_dir_candidates[0]
        predicted_files = glob.glob(os.path.join(k_dir, "*.csv"))
        print(f"Found predicted files: {predicted_files}")
        if not predicted_files:
            print(f"[WARN] No predicted CSVs in {k_dir}; skipping")
            continue

        for pred_file in predicted_files:
            method_name = Path(pred_file).stem
            pred_array, pred_row_indices = load_csv_as_array(pred_file)

            # align rows by intersection if needed
            if pred_array.shape != true_array.shape or not np.array_equal(pred_row_indices, true_row_indices):
                common_rows = np.intersect1d(true_row_indices, pred_row_indices)
                if len(common_rows) == 0:
                    print(f"[WARN] No common rows for {method_name} at k={k}; skipping")
                    continue
                true_map = {r: i for i, r in enumerate(true_row_indices)}
                pred_map = {r: i for i, r in enumerate(pred_row_indices)}
                t_idx = [true_map[r] for r in common_rows]
                p_idx = [pred_map[r] for r in common_rows]
                eval_true = true_array[t_idx]
                eval_pred = pred_array[p_idx]
            else:
                eval_true = true_array
                eval_pred = pred_array

            # compute metrics
            _, jsd_mean = jsd_ct(eval_pred, eval_true)
            mse_val = mse_per_celltype(eval_pred, eval_true)
            rmse_mean = rmse_per_celltype(eval_pred, eval_true)
            _, mae_mean = mae_per_celltype(eval_pred, eval_true)
            _, pearson_mean = per_celltype_pearson(eval_pred, eval_true)
            _, spearman_mean = per_celltype_spearman(eval_pred, eval_true)
            print(
                f"Computed metrics for {method_name} at k={k}: "
                f"JSD={jsd_mean}, MSE={mse_val}, "
                f"RMSE={rmse_mean}, MAE={mae_mean}, "
                f"Pearson={pearson_mean}, Spearman={spearman_mean}"
            )

            store = metrics_by_method.setdefault(
                method_name,
                {
                    "k": [],
                    "jsd": [],
                    "mse": [],
                    "rmse": [],
                    "mae": [],
                    "pearson": [],
                    "spearman": [],
                },
            )
            store["k"].append(k if k is not None else " ")
            store["jsd"].append(jsd_mean)
            store["mse"].append(mse_val)
            store["rmse"].append(rmse_mean)
            store["mae"].append(mae_mean)
            store["pearson"].append(pearson_mean)
            store["spearman"].append(spearman_mean)

    # plot three graphs: one per metric
    def _plot_metric(metric_key, title, ylabel, fname):
        plt.figure(figsize=(4, 3))
        for method, vals in metrics_by_method.items():
            # order by k
            order = np.argsort(vals["k"]) if len(vals["k"]) else []
            ks_sorted = np.array(vals["k"])[order]
            ys = np.array(vals[metric_key])[order]
            if len(ks_sorted) == 0:
                continue
            if method =='count_bridge':
                method = 'Count Bridge'
            plt.plot(ks_sorted, ys, marker='o', label=method)
        plt.title(title)
        plt.xlabel('Number of cell types')
        plt.ylabel(ylabel)
        #plt.grid(True, alpha=0.3)
        plt.legend(loc = 'upper right', bbox_to_anchor=(1.5, 1), borderaxespad=0)
        out_path = os.path.join(args.figs_out, fname)
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"Saved {title} to {out_path}")
        plt.close()

    if args.figs_out:
        _plot_metric("jsd", "JSD vs Number of Cell Types", "Mean JSD", "metric_jsd.png")
        _plot_metric("mse", "MSE vs Number of Cell Types", "Mean MSE", "metric_mse.png")
        _plot_metric("rmse", "RMSE vs Number of Cell Types", "RMSE", "metric_rmse.png")
        _plot_metric("mae", "Per-Celltype MAE vs Number of Cell Types", "MAE (per CT)", "metric_mae.png")
        _plot_metric("pearson", "Pearson Correlation vs Number of Cell Types", "Mean Pearson r", "metric_pearson.png")
        _plot_metric("spearman", "Spearman Correlation vs Number of Cell Types", "Mean Spearman ρ", "metric_spearman.png")

        # build and save summary tables: rows=methods, columns=cell type number (k)
        def _save_summary_table(metric_key, fname):
            # collect all ks observed across methods
            all_ks = sorted({k for vals in metrics_by_method.values() for k in vals["k"]})
            methods = sorted(metrics_by_method.keys())
            table = pd.DataFrame(index=methods, columns=all_ks, dtype=float)
            for method in methods:
                vals = metrics_by_method[method]
                for k_val, metric_val in zip(vals["k"], vals[metric_key]):
                    table.at[method, k_val] = metric_val
            out_csv = os.path.join(args.figs_out, fname)
            table.to_csv(out_csv)
            print(f"Saved summary table to {out_csv}")

        _save_summary_table("jsd", "table_jsd_scipy.csv")
        _save_summary_table("mse", "table_mse.csv")
        _save_summary_table("rmse", "table_rmse.csv")
        _save_summary_table("mae", "table_mae.csv")
        _save_summary_table("pearson", "table_pearson.csv")
        _save_summary_table("spearman", "table_spearman.csv")

    # Print summary table to stdout
    def _print_summary_table(metric_key, title):
        print(f"\n{title}")
        print("=" * 60)
        
        # collect all ks observed across methods
        all_ks = sorted({k for vals in metrics_by_method.values() for k in vals["k"]})
        methods = sorted(metrics_by_method.keys())
        
        if not methods:
            print("No methods found.")
            return
            
        # Create table data
        table_data = []
        for method in methods:
            vals = metrics_by_method[method]
            row = [method]
            for k_val in all_ks:
                if k_val in vals["k"]:
                    idx = vals["k"].index(k_val)
                    row.append(f"{vals[metric_key][idx]:.3f}")
                else:
                    row.append("N/A")
            table_data.append(row)
        
        # Print header
        header = ["Method"] + [str(k) for k in all_ks]
        col_widths = [max(len(str(cell)) for cell in col) for col in zip(*([header] + table_data))]
        
        # Print table
        print(" | ".join(h.ljust(w) for h, w in zip(header, col_widths)))
        print("-" * sum(col_widths + [3 * (len(col_widths) - 1)]))
        for row in table_data:
            print(" | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))

    _print_summary_table("jsd", "JSD (Scipy) Summary")
    _print_summary_table("mse", "MSE Summary")
    _print_summary_table("rmse", "RMSE Summary")
    _print_summary_table("mae", "Per-Celltype MAE Summary")
    _print_summary_table("pearson", "Pearson Correlation Summary")
    _print_summary_table("spearman", "Spearman Correlation Summary")

    # Per-celltype tables: rows=methods, columns=[JSD (paper), RMSE]
    def _print_method_summary_per_k():
        # collect all ks observed across methods, keep order stable by sorting string/int uniformly
        all_ks = sorted({k for vals in metrics_by_method.values() for k in vals["k"]})
        methods = sorted(metrics_by_method.keys())
        if not methods:
            print("\nNo methods found for per-k summaries.")
            return
        for k_val in all_ks:
            print(f"\nSummary for cell types: {k_val}")
            print("=" * 60)
            header = ["Method", "JSD", "MSE", "RMSE", "MAE", "Pearson", "Spearman"]
            rows = []
            for method in methods:
                vals = metrics_by_method[method]
                if method =='count_bridge':
                    method = 'Count Bridge'
                if k_val in vals["k"]:
                    idx = vals["k"].index(k_val)
                    jsd_v = vals["jsd"][idx]
                    rmse_v = vals["rmse"][idx]
                    mse_v = vals["mse"][idx]
                    mae_v = vals["mae"][idx]
                    pearson_v = vals["pearson"][idx]
                    spearman_v = vals["spearman"][idx]
                    rows.append(
                        [
                            method,
                            f"{jsd_v:.3f}",
                            f"{mse_v:.3f}",
                            f"{rmse_v:.3f}",
                            f"{mae_v:.3f}",
                            f"{pearson_v:.3f}",
                            f"{spearman_v:.3f}",
                        ]
                    )
                else:
                    rows.append([method, "N/A", "N/A", "N/A", "N/A", "N/A"])
            col_widths = [max(len(str(c)) for c in col) for col in zip(*([header] + rows))]
            print(" & ".join(h.ljust(w) for h, w in zip(header, col_widths)))
            print("-" * sum(col_widths + [3 * (len(col_widths) - 1)]))
            for r in rows:
                print(" & ".join(str(c).ljust(w) for c, w in zip(r, col_widths)))
            

    _print_method_summary_per_k()


if __name__ == "__main__":
    main()
