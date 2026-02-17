import argparse
import os
import numpy as np
import pandas as pd
import chromadb

from ultrafast.embed import embed
from ultrafast.predict import parse_smi


def query_cli():
    parser = argparse.ArgumentParser(
        description='Query drugs against a pre-built proteome database'
    )
    parser.add_argument('--smi', type=str, required=True,
                        help='Path to SMI file (one SMILES per line)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained SPRINT model checkpoint')
    parser.add_argument('--db-dir', type=str, required=True,
                        help='Path to ChromaDB proteome database directory')
    parser.add_argument('--db-name', type=str, required=True,
                        help='Name of the ChromaDB collection to query')
    parser.add_argument('--topk', type=int, default=10,
                        help='Number of top hits per drug (default: 10)')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Directory to save outputs (default: ./results)')
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
    query(**vars(args))


def query(smi, checkpoint, db_dir, db_name, topk=10, output_dir='./results',
          batch_size=128, device=0, num_workers=-1, ext='h5'):
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Parse SMI and build CSV
    drugs = parse_smi(smi)
    print(f"Loaded {len(drugs)} drugs")

    drug_csv = os.path.join(output_dir, 'drugs.csv')
    drug_df = pd.DataFrame({
        'SMILES': [s for s, _ in drugs],
        'id': [d for _, d in drugs],
    })
    drug_df.to_csv(drug_csv, index=False)

    # Step 2: Embed drugs
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

    # Step 3: Query ChromaDB
    print(f"Querying {db_name} database...")
    drug_embs = np.load(drug_emb_path)
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_or_create_collection(
        name=db_name, metadata={"hnsw:space": "cosine"}
    )

    rows = []
    for i, (smi_str, drug_id) in enumerate(drugs):
        results = collection.query(
            query_embeddings=[drug_embs[i].tolist()],
            n_results=topk,
        )
        for rank, (tid, doc, dist) in enumerate(zip(
            results['ids'][0],
            results['documents'][0],
            results['distances'][0],
        ), 1):
            rows.append({
                'drug_id': drug_id,
                'smiles': smi_str,
                'rank': rank,
                'target_id': tid,
                'target_sequence': doc,
                'distance': round(dist, 6),
            })

    result_df = pd.DataFrame(rows)
    output_path = os.path.join(output_dir, 'query_results.csv')
    result_df.to_csv(output_path, index=False)
    print(f"Saved {len(result_df)} hits to {output_path}")


if __name__ == '__main__':
    query_cli()
