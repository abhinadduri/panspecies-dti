import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ultrafast.embed import embed
from ultrafast.model import DrugTargetCoembeddingLightning


def parse_fasta(fasta_path):
    """Parse a FASTA file into a list of (protein_id, sequence) tuples."""
    sequences = []
    current_id = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_id is not None:
                    sequences.append((current_id, ''.join(current_seq)))
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

    if current_id is not None:
        sequences.append((current_id, ''.join(current_seq)))

    return sequences


def parse_smi(smi_path):
    """Parse a SMI file into a list of (smiles, drug_id) tuples."""
    drugs = []

    with open(smi_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            smiles = parts[0]
            drug_id = parts[1] if len(parts) > 1 else f"drug_{i}"
            drugs.append((smiles, drug_id))

    return drugs


def predict_cli():
    parser = argparse.ArgumentParser(
        description='Predict drug-target interactions from FASTA and SMI files'
    )
    parser.add_argument('--fasta', type=str, required=True,
                        help='Path to FASTA file with protein sequences')
    parser.add_argument('--smi', type=str, required=True,
                        help='Path to SMI file (one SMILES per line)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained SPRINT model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Directory to save all outputs (default: ./results)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for embedding (default: 128)')
    parser.add_argument('--device', type=int, default=0,
                        help='CUDA device ID (default: 0)')
    parser.add_argument('--num-workers', type=int, default=-1,
                        help='Number of data loading workers (default: -1 = auto)')
    parser.add_argument('--ext', type=str, default='h5',
                        choices=['h5', 'lmdb', 'pt'],
                        help='Intermediate featurization format (default: h5)')
    args = parser.parse_args()
    predict(**vars(args))


def predict(fasta, smi, checkpoint, output_dir='./results', batch_size=128,
            device=0, num_workers=-1, ext='h5'):
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Parse input files
    proteins = parse_fasta(fasta)
    drugs = parse_smi(smi)
    print(f"Loaded {len(proteins)} proteins and {len(drugs)} drugs")

    # Step 2: Build CSVs expected by the existing embed pipeline
    target_csv = os.path.join(output_dir, 'targets.csv')
    drug_csv = os.path.join(output_dir, 'drugs.csv')

    target_df = pd.DataFrame({
        'Target Sequence': [seq for _, seq in proteins],
        'uniprot_id': [pid for pid, _ in proteins],
    })
    target_df.to_csv(target_csv, index=False)

    drug_df = pd.DataFrame({
        'SMILES': [smi_str for smi_str, _ in drugs],
        'id': [did for _, did in drugs],
    })
    drug_df.to_csv(drug_csv, index=False)

    # Step 3: Embed targets
    target_emb_path = os.path.join(output_dir, 'target_embeddings.npy')
    print("Embedding targets...")
    embed(
        checkpoint=checkpoint,
        device=device,
        data_file=target_csv,
        moltype='target',
        output_path=target_emb_path,
        batch_size=batch_size,
        ext=ext,
        map_size=10000,
        num_workers=num_workers,
    )

    # Step 4: Embed drugs
    drug_emb_path = os.path.join(output_dir, 'drug_embeddings.npy')
    print("Embedding drugs...")
    embed(
        checkpoint=checkpoint,
        device=device,
        data_file=drug_csv,
        moltype='drug',
        output_path=drug_emb_path,
        batch_size=batch_size,
        ext=ext,
        map_size=10000,
        num_workers=num_workers,
    )

    # Step 5: Compute all-pairs DTI predictions
    print("Computing DTI predictions...")
    target_embs = torch.tensor(np.load(target_emb_path))
    drug_embs = torch.tensor(np.load(drug_emb_path))

    target_norm = F.normalize(target_embs, dim=1)
    drug_norm = F.normalize(drug_embs, dim=1)

    # Cosine similarity matrix: (n_proteins, n_drugs)
    sim_matrix = torch.mm(target_norm, drug_norm.t())

    # Load sigmoid_scalar from model checkpoint (default: 5)
    model = DrugTargetCoembeddingLightning.load_from_checkpoint(checkpoint)
    sigmoid_scalar = getattr(model.args, 'sigmoid_scalar', 5)

    # Interaction probability
    prob_matrix = torch.sigmoid(sigmoid_scalar * sim_matrix)

    # Step 6: Write output CSV
    rows = []
    for i, (pid, _) in enumerate(proteins):
        for j, (smi_str, did) in enumerate(drugs):
            rows.append({
                'protein_id': pid,
                'smiles': smi_str,
                'drug_id': did,
                'cosine_similarity': round(sim_matrix[i, j].item(), 6),
                'interaction_probability': round(prob_matrix[i, j].item(), 6),
            })

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values('interaction_probability', ascending=False)

    output_path = os.path.join(output_dir, 'dti_predictions.csv')
    result_df.to_csv(output_path, index=False)
    print(f"Saved {len(result_df)} predictions to {output_path}")


if __name__ == '__main__':
    predict_cli()
