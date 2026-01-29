#!/usr/bin/env python3
"""
Benchmark script for ChromaDB query timing evaluation.

This script measures ChromaDB query performance across different database sizes
and top-k values by:
1. Loading SMILES from id_to_smiles.npy
2. Loading protein sequences from id_to_saprot_sequence.npy
3. Building databases of increasing sizes (10 to 1E8 molecules)
4. Timing top-k queries (k=1, 10, 100, 1000) for each database size
5. Generating a plot showing query time vs database size

Modes:
- 'full': Generate embeddings and run timing benchmarks (default)
- 'embed-only': Only generate and save embeddings for later use
- 'timing-only': Load existing embeddings and run timing benchmarks

Example usage:
  # Generate embeddings only
  python benchmark_chromadb_timing.py --mode embed-only --checkpoint model.ckpt --output-dir ./embeddings
  
  # Run timing benchmarks using existing embeddings
  python benchmark_chromadb_timing.py --mode timing-only --output-dir ./embeddings --db-dir ./dbs
  
  # Do both in one run
  python benchmark_chromadb_timing.py --mode full --checkpoint model.ckpt
"""

import os
import argparse
import time
import tempfile
import shutil
import random
import json
import numpy as np
import pandas as pd
import chromadb
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from ultrafast.embed import embed
from ultrafast.store import store


def load_data(smiles_path: str, sequences_path: str) -> Tuple[Dict, Dict]:
    """
    Load SMILES and protein sequences from numpy files.
    
    Args:
        smiles_path: Path to id_to_smiles.npy
        sequences_path: Path to id_to_saprot_sequence.npy
        
    Returns:
        Tuple of (id_to_smiles dict, id_to_sequences dict)
    """
    if not os.path.exists(smiles_path):
        raise FileNotFoundError(f"SMILES file not found: {smiles_path}")
    if not os.path.exists(sequences_path):
        raise FileNotFoundError(f"Sequences file not found: {sequences_path}")
    
    print(f"Loading SMILES from {smiles_path}...")
    id_to_smiles = np.load(smiles_path, allow_pickle=True).item()
    print(f"Loaded {len(id_to_smiles)} SMILES")
    
    print(f"Loading protein sequences from {sequences_path}...")
    id_to_sequences = np.load(sequences_path, allow_pickle=True).item()
    print(f"Loaded {len(id_to_sequences)} protein sequences")
    
    return id_to_smiles, id_to_sequences


def select_random_protein(id_to_sequences: Dict) -> Tuple[str, str]:
    """
    Select a random protein sequence from the dictionary.
    
    Args:
        id_to_sequences: Dictionary mapping protein IDs to sequences
        
    Returns:
        Tuple of (protein_id, protein_sequence)
    """
    protein_id = random.choice(list(id_to_sequences.keys()))
    protein_sequence = id_to_sequences[protein_id]
    print(f"Selected random protein: {protein_id}")
    return protein_id, protein_sequence


def create_temp_csv(data: List[str], column_name: str, output_path: str, delimiter: str = '\t') -> str:
    """
    Create a temporary TSV file with the given data.
    Uses TSV format to avoid issues with spaces in column names.
    
    For single-column files, pd.read_table with sep=None may not detect tabs correctly.
    We add a dummy second column to help delimiter detection, then EmbedDataset will
    only use the column it needs.
    
    Args:
        data: List of strings to write
        column_name: Name of the column
        output_path: Path to save the file
        delimiter: Delimiter to use (default: tab for TSV format)
        
    Returns:
        Path to the created file
    """
    # Filter out NaN, None, and invalid values
    filtered_data = []
    for item in data:
        if item is None:
            continue
        if pd.isna(item):
            continue
        if not isinstance(item, str):
            # Convert to string if possible
            try:
                item = str(item)
            except:
                continue
        if len(item.strip()) == 0:
            continue
        filtered_data.append(item)
    
    if len(filtered_data) == 0:
        raise ValueError(f"No valid data to write after filtering NaN/None values")
    
    if len(filtered_data) < len(data):
        print(f"Warning: Filtered out {len(data) - len(filtered_data)} invalid values (NaN/None/empty)")
    
    # Create DataFrame with the actual column
    df = pd.DataFrame({column_name: filtered_data})
    
    # Ensure no NaN values remain
    if df[column_name].isna().any():
        nan_count = df[column_name].isna().sum()
        print(f"Warning: Found {nan_count} NaN values, removing them")
        df = df.dropna(subset=[column_name])
    
    # Add a dummy column to help pd.read_table with sep=None detect tabs correctly
    # This is necessary because csv.Sniffer needs multiple columns to detect delimiters
    df['_dummy'] = 'dummy'
    
    # Write TSV file
    df.to_csv(output_path, index=False, sep=delimiter, na_rep='')
    
    # Verify the file was created correctly
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Failed to create file: {output_path}")
    
    # Verify it can be read with sep=None (same as EmbedDataset uses)
    test_df = pd.read_table(output_path, header=0, sep=None)
    if column_name not in test_df.columns:
        raise ValueError(f"Column '{column_name}' not found when reading with sep=None. Columns: {test_df.columns.tolist()}")
    
    # Verify no NaN values in the column
    if test_df[column_name].isna().any():
        nan_count = test_df[column_name].isna().sum()
        print(f"Warning: Found {nan_count} NaN values after reading file, removing them")
        test_df = test_df.dropna(subset=[column_name])
        # Rewrite the file without NaN values
        test_df.to_csv(output_path, index=False, sep=delimiter, na_rep='')
    
    # Note: The dummy column will be ignored by EmbedDataset since it only uses
    # the column specified by moltype ('SMILES' or 'Target Sequence')
    
    return output_path


def embed_molecules(
    checkpoint: str,
    device: int,
    data_file: str,
    moltype: str,
    output_path: str,
    batch_size: int = 128,
    ext: str = "lmdb",
    map_size: int = 10000,
    num_workers: int = -1,
) -> str:
    """
    Wrapper function to create embeddings using the embed() function.
    
    Args:
        checkpoint: Path to model checkpoint
        device: CUDA device number
        data_file: Path to CSV file with molecules
        moltype: "drug" or "target"
        output_path: Path to save embeddings
        batch_size: Batch size for embedding
        ext: File format for features
        map_size: Map size limit for LMDB
        num_workers: Number of worker processes
        
    Returns:
        Path to saved embeddings file
    """
    embed(
        checkpoint=checkpoint,
        device=device,
        data_file=data_file,
        moltype=moltype,
        output_path=output_path,
        batch_size=batch_size,
        ext=ext,
        map_size=map_size,
        num_workers=num_workers,
    )
    return output_path


def store_database(
    data_file: str,
    embeddings: str,
    moltype: str,
    db_dir: str,
    db_name: str,
    delimiter: str = ',',
) -> None:
    """
    Wrapper function to store embeddings in ChromaDB.
    
    Args:
        data_file: Path to CSV file with molecules
        embeddings: Path to embeddings numpy file
        moltype: "drug" or "target"
        db_dir: Directory for ChromaDB databases
        db_name: Name of the database collection
        delimiter: CSV delimiter
    """
    store(
        data_file=data_file,
        embeddings=embeddings,
        moltype=moltype,
        db_dir=db_dir,
        db_name=db_name,
        delimiter=delimiter,
    )


def time_query(
    query_embedding: List[float],
    db_dir: str,
    db_name: str,
    k: int,
    num_trials: int = 5,
) -> float:
    """
    Time a ChromaDB query operation.
    
    Args:
        query_embedding: Query embedding as a list
        db_dir: Directory containing the ChromaDB database
        db_name: Name of the database collection
        k: Number of results to retrieve (top-k)
        num_trials: Number of trials to average over
        
    Returns:
        Average query time in seconds
    """
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_or_create_collection(name=db_name, metadata={"hnsw:space": "cosine"})
    
    times = []
    for _ in range(num_trials):
        start_time = time.perf_counter()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return np.mean(times)


def plot_results(
    results: Dict[int, Dict[int, float]],
    output_path: str,
) -> None:
    """
    Plot timing results.
    
    Args:
        results: Dictionary mapping database size to dictionary of k -> time
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    k_values = [1, 10, 100, 1000]
    colors = ['blue', 'green', 'orange', 'red']
    
    database_sizes = sorted(results.keys())
    
    for k, color in zip(k_values, colors):
        sizes_for_k = []
        times_for_k = []
        for size in database_sizes:
            if k in results[size]:
                sizes_for_k.append(size)
                times_for_k.append(results[size][k])
        if sizes_for_k:  # Only plot if we have data for this k
            ax.plot(sizes_for_k, times_for_k, marker='o', label=f'k={k}', color=color, linewidth=2)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Molecules in Database', fontsize=12)
    ax.set_ylabel('Query Time (seconds)', fontsize=12)
    ax.set_title('ChromaDB Query Performance vs Database Size', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


def save_metadata(
    output_dir: str,
    protein_id: str,
    protein_sequence: str,
    database_sizes: List[int],
    seed: int,
) -> str:
    """
    Save metadata about the embedding run.
    
    Args:
        output_dir: Directory to save metadata
        protein_id: ID of the protein used for querying
        protein_sequence: Sequence of the protein used for querying
        database_sizes: List of database sizes that were embedded
        seed: Random seed used
        
    Returns:
        Path to saved metadata file
    """
    metadata = {
        'protein_id': protein_id,
        'protein_sequence': protein_sequence,
        'database_sizes': database_sizes,
        'seed': seed,
    }
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    return metadata_path


def load_metadata(output_dir: str) -> Dict:
    """
    Load metadata from a previous embedding run.
    
    Args:
        output_dir: Directory containing metadata.json
        
    Returns:
        Dictionary with metadata
    """
    metadata_path = os.path.join(output_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark ChromaDB query timing across different database sizes and top-k values'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'embed-only', 'timing-only'],
        default='full',
        help='Mode: "full" (embed and time), "embed-only" (only generate embeddings), or "timing-only" (use existing embeddings)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (required for full and embed-only modes)'
    )
    parser.add_argument(
        '--smiles-path',
        type=str,
        default='data/MERGED/huge_data/id_to_smiles.npy',
        help='Path to id_to_smiles.npy file'
    )
    parser.add_argument(
        '--sequences-path',
        type=str,
        default='data/MERGED/huge_data/id_to_saprot_sequence.npy',
        help='Path to id_to_saprot_sequence.npy file'
    )
    parser.add_argument(
        '--db-dir',
        type=str,
        default='./benchmark_dbs',
        help='Directory for ChromaDB databases'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./benchmark_results',
        help='Directory for embeddings and plots'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='CUDA device number'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for embedding'
    )
    parser.add_argument(
        '--num-trials',
        type=int,
        default=5,
        help='Number of query trials per measurement'
    )
    parser.add_argument(
        '--max-size',
        type=int,
        default=int(1e8),
        help='Maximum database size'
    )
    parser.add_argument(
        '--plot-output',
        type=str,
        default=None,
        help='Output path for plot (default: {output_dir}/timing_plot.png)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=-1,
        help='Number of worker processes for embedding'
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint for modes that need it
    if args.mode in ['full', 'embed-only']:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for 'full' and 'embed-only' modes")
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    # Create output directories
    os.makedirs(args.db_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set plot output path
    if args.plot_output is None:
        args.plot_output = os.path.join(args.output_dir, 'timing_plot.png')
    
    # Handle timing-only mode: load existing embeddings and metadata
    if args.mode == 'timing-only':
        print("Mode: timing-only - Loading existing embeddings and metadata...")
        metadata = load_metadata(args.output_dir)
        protein_id = metadata['protein_id']
        protein_sequence = metadata['protein_sequence']
        database_sizes = metadata['database_sizes']
        # Use the seed from when embeddings were created (for consistency)
        random.seed(metadata['seed'])
        np.random.seed(metadata['seed'])
        
        # Load protein embedding
        protein_emb_path = os.path.join(args.output_dir, 'protein_embedding.npy')
        if not os.path.exists(protein_emb_path):
            raise FileNotFoundError(f"Protein embedding not found: {protein_emb_path}")
        protein_embedding = np.load(protein_emb_path, allow_pickle=True)
        if len(protein_embedding.shape) > 1:
            protein_embedding = protein_embedding[0]
        protein_embedding = protein_embedding.tolist()
        print(f"Loaded protein embedding for protein: {protein_id}")
        
        # Results dictionary: {database_size: {k: time}}
        results = {}
        
        # Time queries for each database size
        print(f"\nDatabase sizes to test: {database_sizes}")
        print(f"Top-k values to test: [1, 10, 100, 1000]")
        
        for db_size in database_sizes:
            print(f"\n{'='*60}")
            print(f"Timing queries for database size: {db_size}")
            print(f"{'='*60}")
            
            db_name = f'drugs_{db_size}'
            
            # Check if database exists
            client = chromadb.PersistentClient(path=args.db_dir)
            try:
                collection = client.get_collection(name=db_name)
                print(f"Found existing database: {db_name} ({collection.count()} molecules)")
            except Exception as e:
                print(f"Warning: Database {db_name} not found. Creating from saved embeddings...")
                # Load embeddings and TSV, then create database
                embeddings_path = os.path.join(args.output_dir, f'embeddings_{db_size}.npy')
                smiles_csv = os.path.join(args.output_dir, f'smiles_{db_size}.tsv')
                
                if not os.path.exists(embeddings_path):
                    raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
                if not os.path.exists(smiles_csv):
                    raise FileNotFoundError(f"TSV file not found: {smiles_csv}")
                
                store_database(
                    data_file=smiles_csv,
                    embeddings=embeddings_path,
                    moltype='drug',
                    db_dir=args.db_dir,
                    db_name=db_name,
                    delimiter='\t',
                )
            
            # Time queries for each k value
            results[db_size] = {}
            for k in [1, 10, 100, 1000]:
                if k > db_size:
                    print(f"Skipping k={k} for database size {db_size} (k > db_size)")
                    continue
                print(f"Timing query for k={k}...")
                avg_time = time_query(
                    query_embedding=protein_embedding,
                    db_dir=args.db_dir,
                    db_name=db_name,
                    k=k,
                    num_trials=args.num_trials,
                )
                results[db_size][k] = avg_time
                print(f"  Average query time: {avg_time:.6f} seconds")
        
        # Generate plot
        print(f"\n{'='*60}")
        print("Generating plot...")
        print(f"{'='*60}")
        plot_results(results, args.plot_output)
        
        # Print summary
        print("\nSummary of results:")
        print(f"{'Database Size':<15} {'k=1':<15} {'k=10':<15} {'k=100':<15} {'k=1000':<15}")
        print("-" * 75)
        for db_size in sorted(results.keys()):
            row = [f"{db_size:<15}"]
            for k in [1, 10, 100, 1000]:
                if k in results[db_size]:
                    row.append(f"{results[db_size][k]:.6f}")
                else:
                    row.append("N/A")
            print(" ".join(row))
        
        return
    
    # Handle embed-only and full modes
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load data
    id_to_smiles, id_to_sequences = load_data(args.smiles_path, args.sequences_path)
    
    # Select random protein
    protein_id, protein_sequence = select_random_protein(id_to_sequences)
    
    # Define database sizes (logarithmic progression)
    max_available = len(id_to_smiles)
    database_sizes = []
    for size in [10, 100, 1000, 10000, 100000, 1000000, 10000000, int(1e8)]:
        if size <= min(max_available, args.max_size):
            database_sizes.append(size)
    
    if not database_sizes:
        raise ValueError(f"No valid database sizes found. Available SMILES: {max_available}, Max size: {args.max_size}")
    
    print(f"\nDatabase sizes to process: {database_sizes}")
    
    # Create temporary directory for intermediate files (only used during embedding in full mode)
    temp_dir = None
    if args.mode == 'full':
        temp_dir = tempfile.mkdtemp(prefix='chromadb_benchmark_')
        print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Generate protein query embedding (once)
        print("\nGenerating protein query embedding...")
        # Save protein CSV to output_dir so it can be reused
        protein_csv = os.path.join(args.output_dir, 'protein_query.csv')
        create_temp_csv([protein_sequence], 'Target Sequence', protein_csv, delimiter=',')
        protein_emb_path = os.path.join(args.output_dir, 'protein_embedding.npy')
        embed_molecules(
            checkpoint=args.checkpoint,
            device=args.device,
            data_file=protein_csv,
            moltype='target',
            output_path=protein_emb_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        protein_embedding = np.load(protein_emb_path, allow_pickle=True)
        if len(protein_embedding.shape) > 1:
            protein_embedding = protein_embedding[0]
        protein_embedding_list = protein_embedding.tolist()
        print("Protein embedding generated and saved")
        
        # Save metadata
        save_metadata(args.output_dir, protein_id, protein_sequence, database_sizes, args.seed)
        
        # Get all SMILES IDs
        smiles_ids = list(id_to_smiles.keys())
        
        # Results dictionary: {database_size: {k: time}}
        results = {}
        
        # Process each database size
        for db_size in database_sizes:
            print(f"\n{'='*60}")
            print(f"Processing database size: {db_size}")
            print(f"{'='*60}")
            
            # Sample SMILES for this database size (use same seed for reproducibility)
            random.seed(args.seed)
            np.random.seed(args.seed)
            sampled_ids = random.sample(smiles_ids, min(db_size, len(smiles_ids)))
            sampled_smiles = [id_to_smiles[smile_id] for smile_id in sampled_ids]
            
            # Filter out any invalid SMILES before creating the file
            valid_smiles = []
            for smile in sampled_smiles:
                if smile is None or pd.isna(smile):
                    continue
                if not isinstance(smile, str):
                    continue
                if len(smile.strip()) == 0:
                    continue
                valid_smiles.append(smile)
            
            if len(valid_smiles) < len(sampled_smiles):
                print(f"Warning: Filtered out {len(sampled_smiles) - len(valid_smiles)} invalid SMILES")
            
            if len(valid_smiles) == 0:
                raise ValueError(f"No valid SMILES found for database size {db_size}")
            
            # Create TSV file (save to output_dir for reuse)
            # Using TSV format to avoid issues with spaces in column names
            smiles_csv = os.path.join(args.output_dir, f'smiles_{db_size}.tsv')
            create_temp_csv(valid_smiles, 'SMILES', smiles_csv, delimiter='\t')
            
            # Generate embeddings (save to output_dir)
            print(f"Generating embeddings for {db_size} molecules...")
            embeddings_path = os.path.join(args.output_dir, f'embeddings_{db_size}.npy')
            if os.path.exists(os.path.join(args.output_dir, f'Morgan_features.lmdb')):
                # delete the directory
                shutil.rmtree(os.path.join(args.output_dir, f'Morgan_features.lmdb'))
            embed_molecules(
                checkpoint=args.checkpoint,
                device=args.device,
                data_file=smiles_csv,
                moltype='drug',
                output_path=embeddings_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            print(f"Embeddings saved to {embeddings_path}")
            
            print(f"Storing {db_size} molecules in ChromaDB...")
            db_name = f'drugs_{db_size}'
            store_database(
                data_file=smiles_csv,
                embeddings=embeddings_path,
                moltype='drug',
                db_dir=args.db_dir,
                db_name=db_name,
                delimiter='\t',
            )

            # Store in ChromaDB (only if not embed-only mode, or always to have DB ready)
            if args.mode == 'full':
                # Time queries for each k value
                results[db_size] = {}
                for k in [1, 10, 100, 1000]:
                    if k > db_size:
                        print(f"Skipping k={k} for database size {db_size} (k > db_size)")
                        continue
                    print(f"Timing query for k={k}...")
                    avg_time = time_query(
                        query_embedding=protein_embedding_list,
                        db_dir=args.db_dir,
                        db_name=db_name,
                        k=k,
                        num_trials=args.num_trials,
                    )
                    results[db_size][k] = avg_time
                    print(f"  Average query time: {avg_time:.6f} seconds")
        
        # Generate plot and summary (only for full mode)
        if args.mode == 'full':
            print(f"\n{'='*60}")
            print("Generating plot...")
            print(f"{'='*60}")
            plot_results(results, args.plot_output)
            
            # Print summary
            print("\nSummary of results:")
            print(f"{'Database Size':<15} {'k=1':<15} {'k=10':<15} {'k=100':<15} {'k=1000':<15}")
            print("-" * 75)
            for db_size in sorted(results.keys()):
                row = [f"{db_size:<15}"]
                for k in [1, 10, 100, 1000]:
                    if k in results[db_size]:
                        row.append(f"{results[db_size][k]:.6f}")
                    else:
                        row.append("N/A")
                print(" ".join(row))
        else:
            print(f"\n{'='*60}")
            print("Embedding complete!")
            print(f"{'='*60}")
            print(f"Embeddings saved to: {args.output_dir}")
            print(f"Databases created in: {args.db_dir}")
            print(f"\nTo run timing benchmarks later, use:")
            script_name = os.path.basename(__file__)
            print(f"  python {script_name} --mode timing-only --output-dir {args.output_dir} --db-dir {args.db_dir}")
        
    finally:
        # Clean up temporary files
        if temp_dir is not None and os.path.exists(temp_dir):
            print(f"\nCleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
