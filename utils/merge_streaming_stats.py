#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from pathlib import Path
from typing import List, Dict
from scipy.stats import gaussian_kde

from compute_streaming_stats import StreamingStats

def weighted_gaussian_kde(x, weights, bandwidth="scott"):
    """
    Weighted Gaussian KDE using scipy's gaussian_kde formula with weights.
    """
    # Normalize weights
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()

    # Define weighted KDE
    kde = gaussian_kde(x, bw_method=bandwidth, weights=weights)
    return kde

def merge_stats_files(pickle_files: List[str], output_prefix: str = 'merged_stats', 
                     plot_histogram: bool = True, plot_kde: bool = True):
    """
    Merge multiple pickle files containing StreamingStats and generate visualizations.
    
    Args:
        pickle_files: List of paths to pickle files containing StreamingStats
        output_prefix: Prefix for output files
        plot_histogram: Whether to generate histogram plot
        plot_kde: Whether to add kernel density estimate to plot
    """
    
    if not pickle_files:
        raise ValueError("No pickle files provided")
    
    print(f"Loading and merging {len(pickle_files)} statistics files...")
    
    # Load first file
    merged_stats = StreamingStats.load_state(pickle_files[0])
    print(f"Loaded: {pickle_files[0]}")
    
    # Merge remaining files
    for pickle_file in pickle_files[1:]:
        other_stats = StreamingStats.load_state(pickle_file)
        merged_stats.merge(other_stats)
        print(f"Merged: {pickle_file}")
    
    # Print summary
    summary = merged_stats.get_summary()
    print_summary(summary)
    
    # Save merged state for future use
    state_file = f"{output_prefix}_state.pkl"
    merged_stats.save_state(state_file)
    print(f"Merged state saved to: {state_file}")
    
    # Generate visualization
    if plot_histogram:
        generate_histogram_plot(merged_stats, output_prefix, plot_kde)
    
    return merged_stats


def generate_histogram_plot(stats: StreamingStats, output_prefix: str, plot_kde: bool = True):
    """Generate histogram plot with optional kernel density estimate."""
    
    hist_counts, hist_edges = stats.histogram.get_histogram()
    
    # Calculate bin centers
    bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
    bin_width = hist_edges[1] - hist_edges[0]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot histogram
    plt.bar(bin_centers, hist_counts, width=bin_width * 0.8, alpha=0.7, 
            color='skyblue', edgecolor='black', linewidth=0.5, label='Histogram')
    
    # Add KDE if requested and data is available
    if plot_kde and stats.count > 1:
        try:
            # Build weighted KDE using scipy
            kde = weighted_gaussian_kde(bin_centers, weights=hist_counts, bandwidth="scott")

            x_range = np.linspace(stats.min_val, stats.max_val, 1000)
            kde_values = kde(x_range)
                
                # Scale KDE to match histogram scale
            kde_scale = np.max(hist_counts) / np.max(kde_values)
            kde_values *= kde_scale
            
            plt.plot(x_range, kde_values, 'red', linewidth=2, label='Kernel Density Estimate')
        except Exception as e:
            print(f"Warning: Could not generate KDE: {e}")
    
    # Formatting
    plt.xlabel('Similarity Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Similarity Scores', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f"""Statistics:
Count: {stats.count:,}
Mean: {stats.mean:.4f}
Std: {stats.std:.4f}
Min: {stats.min_val:.4f}
Max: {stats.max_val:.4f}
Median: {stats.tdigest.quantile(0.5):.4f}
95th percentile: {stats.tdigest.quantile(0.95):.4f}"""
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save plot
    plot_file = f"{output_prefix}_histogram.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Histogram plot saved to: {plot_file}")
    plt.close()


def print_summary(summary: Dict):
    """Print formatted summary statistics."""
    print("\n" + "="*50)
    print("MERGED STATISTICS SUMMARY")
    print("="*50)
    print(f"Count:           {summary['count']:,}")
    print(f"Mean:            {summary['mean']:.6f}")
    print(f"Standard Dev:    {summary['std']:.6f}")
    print(f"Variance:        {summary['variance']:.6f}")
    print(f"Minimum:         {summary['min']:.6f}")
    print(f"Maximum:         {summary['max']:.6f}")
    print(f"Median:          {summary['median']:.6f}")
    print(f"25th percentile: {summary['q25']:.6f}")
    print(f"75th percentile: {summary['q75']:.6f}")
    print(f"95th percentile: {summary['q95']:.6f}")
    print(f"99th percentile: {summary['q99']:.6f}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(
        description='Merge multiple pickle files containing StreamingStats and generate visualizations'
    )
    parser.add_argument('pickle_files', nargs='+', 
                       help='List of pickle files containing StreamingStats to merge')
    parser.add_argument('--output-prefix', '-o', default='merged_stats',
                       help='Prefix for output files (default: merged_stats)')
    parser.add_argument('--no-histogram', action='store_true',
                       help='Skip generating histogram plot')
    parser.add_argument('--kde', action='store_true',
                       help='Generate kernel density estimate in plot')
    
    args = parser.parse_args()
    
    # Validate input files
    for pickle_file in args.pickle_files:
        if not Path(pickle_file).exists():
            raise FileNotFoundError(f"Pickle file not found: {pickle_file}")
    
    # Merge statistics
    merged_stats = merge_stats_files(
        args.pickle_files,
        args.output_prefix,
        plot_histogram=not args.no_histogram,
        plot_kde=args.kde
    )
    
    return merged_stats


if __name__ == "__main__":
    main()
