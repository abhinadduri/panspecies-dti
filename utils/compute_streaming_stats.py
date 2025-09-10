#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd
import torch
import pickle
import json
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

from torch.nn import CosineSimilarity
from torch.utils.data import DataLoader
from ultrafast.datamodules import EmbeddedDataset


class StreamingHistogram:
    def __init__(self, bins: int = 1000, range_min: float = None, range_max: float = None):
        self.bins = bins
        self.range_min = range_min
        self.range_max = range_max
        self.counts = np.zeros(bins, dtype=np.int64)
        self.bin_edges = None
        self.total_count = 0
        self.overflow_count = 0
        self.underflow_count = 0
        
    def _initialize_bins(self, min_val: float, max_val: float):
        if self.range_min is None:
            self.range_min = min_val
        if self.range_max is None:
            self.range_max = max_val
        self.bin_edges = np.linspace(self.range_min, self.range_max, self.bins + 1)
        
    def update(self, values: np.ndarray):
        values = np.asarray(values)
        if self.bin_edges is None:
            self._initialize_bins(values.min(), values.max())
        
        # Count overflow and underflow
        overflow_mask = values > self.range_max
        underflow_mask = values < self.range_min
        self.overflow_count += np.sum(overflow_mask)
        self.underflow_count += np.sum(underflow_mask)
        
        # Only histogram values within range
        valid_values = values[(~overflow_mask) & (~underflow_mask)]
        if len(valid_values) > 0:
            hist, _ = np.histogram(valid_values, bins=self.bin_edges)
            self.counts += hist
        
        self.total_count += len(values)
    
    def merge(self, other: 'StreamingHistogram'):
        if self.bin_edges is None and other.bin_edges is not None:
            self.bin_edges = other.bin_edges.copy()
            self.range_min = other.range_min
            self.range_max = other.range_max
        elif other.bin_edges is not None:
            if not np.allclose(self.bin_edges, other.bin_edges):
                raise ValueError("Cannot merge histograms with different bin edges")
        
        self.counts += other.counts
        self.total_count += other.total_count
        self.overflow_count += other.overflow_count
        self.underflow_count += other.underflow_count
    
    def get_histogram(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.counts.copy(), self.bin_edges.copy() if self.bin_edges is not None else None


class TDigest:
    def __init__(self, compression: int = 100):
        self.compression = compression
        self.centroids = []  # List of (mean, weight) tuples
        self.total_weight = 0
        self.min_val = float('inf')
        self.max_val = float('-inf')
    
    def update(self, values: Union[np.ndarray, List[float]], weights: Optional[np.ndarray] = None):
        values = np.asarray(values)
        if weights is None:
            weights = np.ones(len(values))
        else:
            weights = np.asarray(weights)
        
        self.min_val = min(self.min_val, values.min())
        self.max_val = max(self.max_val, values.max())
        
        for val, weight in zip(values, weights):
            self.centroids.append((float(val), float(weight)))
            self.total_weight += weight
        
        if len(self.centroids) > self.compression * 2:
            self._compress()
    
    def _compress(self):
        if len(self.centroids) <= self.compression:
            return
        
        # Sort centroids by mean
        self.centroids.sort(key=lambda x: x[0])
        
        # Merge adjacent centroids
        new_centroids = []
        current_mean, current_weight = self.centroids[0]
        
        for mean, weight in self.centroids[1:]:
            if len(new_centroids) < self.compression - 1:
                # Can merge this centroid
                combined_weight = current_weight + weight
                combined_mean = (current_mean * current_weight + mean * weight) / combined_weight
                current_mean, current_weight = combined_mean, combined_weight
            else:
                # Start new centroid
                new_centroids.append((current_mean, current_weight))
                current_mean, current_weight = mean, weight
        
        new_centroids.append((current_mean, current_weight))
        self.centroids = new_centroids
    
    def merge(self, other: 'TDigest'):
        self.centroids.extend(other.centroids)
        self.total_weight += other.total_weight
        self.min_val = min(self.min_val, other.min_val)
        self.max_val = max(self.max_val, other.max_val)
        self._compress()
    
    def quantile(self, q: float) -> float:
        if not self.centroids:
            return 0.0
        
        if q <= 0.0:
            return self.min_val
        if q >= 1.0:
            return self.max_val
        
        self.centroids.sort(key=lambda x: x[0])
        target_weight = q * self.total_weight
        cumulative_weight = 0.0
        
        for i, (mean, weight) in enumerate(self.centroids):
            if cumulative_weight + weight >= target_weight:
                if i == 0:
                    return mean
                
                # Interpolate between this centroid and the previous one
                prev_mean, prev_weight = self.centroids[i-1]
                ratio = (target_weight - cumulative_weight) / weight
                return prev_mean + ratio * (mean - prev_mean)
            
            cumulative_weight += weight
        
        return self.centroids[-1][0]


class StreamingStats:
    def __init__(self, histogram_bins: int = 1000, tdigest_compression: int = 100):
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.histogram = StreamingHistogram(bins=histogram_bins, range_min=-1, range_max=1)
        self.tdigest = TDigest(compression=tdigest_compression)
    
    def update(self, values: np.ndarray):
        values = np.asarray(values).flatten()
        
        # Basic statistics
        self.count += len(values)
        self.sum += np.sum(values)
        self.sum_sq += np.sum(values ** 2)
        self.min_val = min(self.min_val, float(np.min(values)))
        self.max_val = max(self.max_val, float(np.max(values)))
        
        # Update histogram and t-digest
        self.histogram.update(values)
        self.tdigest.update(values)
    
    def merge(self, other: 'StreamingStats'):
        self.count += other.count
        self.sum += other.sum
        self.sum_sq += other.sum_sq
        self.min_val = min(self.min_val, other.min_val)
        self.max_val = max(self.max_val, other.max_val)
        self.histogram.merge(other.histogram)
        self.tdigest.merge(other.tdigest)
    
    @property
    def mean(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0
    
    @property
    def variance(self) -> float:
        if self.count <= 1:
            return 0.0
        return (self.sum_sq - self.sum ** 2 / self.count) / (self.count - 1)
    
    @property
    def std(self) -> float:
        return np.sqrt(self.variance)
    
    def get_summary(self) -> Dict:
        return {
            'count': self.count,
            'mean': self.mean,
            'variance': self.variance,
            'std': self.std,
            'min': self.min_val,
            'max': self.max_val,
            'median': self.tdigest.quantile(0.5),
            'q95': self.tdigest.quantile(0.95),
            'q99': self.tdigest.quantile(0.99),
            'q9999': self.tdigest.quantile(0.9999),
            'q99999': self.tdigest.quantile(0.99999)
        }
    
    def save_state(self, filepath: str):
        state = {
            'count': self.count,
            'sum': self.sum,
            'sum_sq': self.sum_sq,
            'min_val': self.min_val,
            'max_val': self.max_val,
            'histogram_counts': self.histogram.counts,
            'histogram_bin_edges': self.histogram.bin_edges,
            'histogram_total_count': self.histogram.total_count,
            'histogram_overflow': self.histogram.overflow_count,
            'histogram_underflow': self.histogram.underflow_count,
            'tdigest_centroids': self.tdigest.centroids,
            'tdigest_total_weight': self.tdigest.total_weight,
            'tdigest_min': self.tdigest.min_val,
            'tdigest_max': self.tdigest.max_val,
            'tdigest_compression': self.tdigest.compression
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load_state(cls, filepath: str) -> 'StreamingStats':
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        stats = cls()
        stats.count = state['count']
        stats.sum = state['sum']
        stats.sum_sq = state['sum_sq']
        stats.min_val = state['min_val']
        stats.max_val = state['max_val']
        
        # Restore histogram
        stats.histogram.counts = state['histogram_counts']
        stats.histogram.bin_edges = state['histogram_bin_edges']
        stats.histogram.total_count = state['histogram_total_count']
        stats.histogram.overflow_count = state['histogram_overflow']
        stats.histogram.underflow_count = state['histogram_underflow']
        if stats.histogram.bin_edges is not None:
            stats.histogram.range_min = stats.histogram.bin_edges[0]
            stats.histogram.range_max = stats.histogram.bin_edges[-1]
        
        # Restore t-digest
        stats.tdigest.centroids = state['tdigest_centroids']
        stats.tdigest.total_weight = state['tdigest_total_weight']
        stats.tdigest.min_val = state['tdigest_min']
        stats.tdigest.max_val = state['tdigest_max']
        stats.tdigest.compression = state['tdigest_compression']
        
        return stats


def compute_streaming_stats_cli():
    parser = argparse.ArgumentParser(description='Compute streaming distribution statistics for billion-scale molecule libraries')
    parser.add_argument('--library-embeddings', type=str, required=True, help='Path to the library embeddings')
    parser.add_argument('--library-data', type=str, required=True, help='Path to the library data (csv)')
    parser.add_argument('--query-embeddings', type=str, required=True, help='Path to the query embeddings')
    parser.add_argument('--query-data', type=str, required=True, help='Path to the query data (csv)')
    parser.add_argument('--batch-size', type=int, default=2048, help='Batch size for the dataloader')
    parser.add_argument('--histogram-bins', type=int, default=20000, help='Number of histogram bins')
    parser.add_argument('--tdigest-compression', type=int, default=100, help='T-digest compression parameter')
    parser.add_argument('--output-prefix', type=str, default='streaming_stats', help='Prefix for output files')
    parser.add_argument('--load-states', type=str, nargs='*', help='Load and merge states from these files')
    parser.add_argument('--delimiter', type=str, default=',', help='Delimiter for the csv files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed progress')
    args = parser.parse_args()
    compute_streaming_stats(**vars(args))


def compute_streaming_stats(library_embeddings, library_data, query_embeddings, query_data, 
                          batch_size=2048, histogram_bins=1000, tdigest_compression=100,
                          output_prefix='streaming_stats', save_state=False, load_states=None,
                          delimiter=',', verbose=False):
    
        
    # Load data
    library_embeddings = EmbeddedDataset(library_embeddings)
    query_embeddings = np.load(query_embeddings)
    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings[np.newaxis, :]
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    query_embeddings = torch.tensor(query_embeddings).to(device)
    cosine_sim = CosineSimilarity(dim=1)
    
    # Initialize streaming statistics for each query
    query_stats = [StreamingStats(histogram_bins, tdigest_compression) 
                   for _ in range(query_embeddings.shape[0])]
    
    # Create dataloader
    dataloader = DataLoader(library_embeddings, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for emb_mols, idxs in tqdm(dataloader, desc="Computing similarities", total=len(dataloader)):
            emb_mols = emb_mols.to(device)
            
            if query_embeddings.shape[0] == 1:
                similarities = cosine_sim(query_embeddings, emb_mols)
                similarities = similarities.unsqueeze(0)
            else:
                # Calculate similarities for multiple queries
                emb_mols = emb_mols.unsqueeze(-1).repeat(1, 1, query_embeddings.shape[0])
                similarities = cosine_sim(query_embeddings.T.unsqueeze(0), emb_mols).T
            
            # Update statistics for each query
            for i, similarity in enumerate(similarities):
                query_stats[i].update(similarity.cpu().numpy())
            
            torch.cuda.empty_cache()
    
    # Load query data for output naming
    query_data = pd.read_csv(query_data, delimiter=delimiter)
    query_ids = query_data['uniprot_id'].values if 'uniprot_id' in query_data.columns else query_data['id'].values

    merged_stats = dict()
    # Handle state loading and merging
    if load_states:
        print(f"Loading {len(load_states)} state files...")
        for state_file in load_states:
            other_stats = StreamingStats.load_state(state_file)
            query_id = state_file.split(output_prefix)[1].split('_state.pkl')[0].strip('_')
            if query_id in merged_stats:
                merged_stats[query_id].merge(other_stats)
            else:
                merged_stats[query_id] = other_stats
    
    # Save results for each query
    for i, (stats, query_id) in enumerate(zip(query_stats, query_ids)):
        # Save state for potential merging
        if query_id in merged_stats:
            print(f"Merging new {query_id} stats with loaded stats")
            stats.merge(merged_stats[query_id])
        stats.save_state(f"{output_prefix}_{query_id}_state.pkl")
        
        if verbose:
            summary = stats.get_summary()
            print(f"\nQuery {query_id} statistics:")
            print(f"  Count: {summary['count']:,}")
            print(f"  Mean: {summary['mean']:.6f}")
            print(f"  Std: {summary['std']:.6f}")
            print(f"  Min: {summary['min']:.6f}")
            print(f"  Max: {summary['max']:.6f}")
            print(f"  Median: {summary['median']:.6f}")
            print(f"  95th percentile: {summary['q95']:.6f}")
    
    return query_stats


if __name__ == "__main__":
    compute_streaming_stats_cli()
