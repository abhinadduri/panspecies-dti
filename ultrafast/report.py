"""
# Evaluate top-k accuracy
```
ultrafast-report --data-file data/BIOSNAP/full_data/test.csv  \
    --embeddings results/BIOSNAP_test_drug_embeddings.npy \
    --moltype drug \ 
    --db_dir ./dbs \
    --db_name biosnap_test_target_embeddings \
    --topk 100
```
"""

import argparse
import pandas as pd
import numpy as np
import chromadb
from tqdm import tqdm


def report_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, required=True, help='Path to the data file')
    parser.add_argument('--embeddings', type=str, required=True, help='Path to the embeddings file')
    parser.add_argument('--moltype', type=str, required=True, help='Type of molecule (target or drug)')
    parser.add_argument('--db_dir', type=str, default='./dbs', help='Path to save the database(s)')
    parser.add_argument('--db_name', type=str, required=True, help='Name of the database under db_dir')
    parser.add_argument('--topk', type=int, default=100, help='Top-k accuracy')
    parser.add_argument('--return_val', choices=['accuracy','topk'], default='accuracy', help='What to return')
    args = parser.parse_args()
    report(**vars(args))


def report(data_file, embeddings, moltype, db_dir, db_name, topk, return_val):
    query_embeddings = np.load(embeddings, allow_pickle=True)
    if len(query_embeddings.shape) < 2:
        query_embeddings = query_embeddings[np.newaxis,:]
    query_embeddings = query_embeddings.tolist()
    df = pd.read_csv(data_file)
    if moltype == 'target':
        query_doc_col = 'Target Sequence'
        value_doc_col = 'SMILES'
    elif moltype == 'drug':
        query_doc_col = 'SMILES'
        value_doc_col = 'Target Sequence'
    query_documents = list(df[query_doc_col])
    value_documents = list(df[value_doc_col]) if return_val =='accuracy' else query_documents
    # Usually we have data with many duplicate targets and drugs, so only store unique ones
    query_labels = {}
    for i, (query_doc, value_doc, query_embedding) in enumerate(zip(query_documents, value_documents, query_embeddings)):
        if query_doc not in query_labels:
            query_labels[query_doc] = {
                'binding_partners': {value_doc} if return_val == "accuracy" else None,
                'embedding': query_embedding,
            }
        elif return_val == "accuracy":
            query_labels[query_doc]['binding_partners'].add(value_doc)
    # Query the given database for the top-k results and log hits
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_or_create_collection(name=db_name, metadata={"hnsw:space": "cosine"})
    total_topk_hits = 0
    top_hits = dict()
    for query_doc, query_data in tqdm(query_labels.items()):
        # Todo: maybe batch queries
        results = collection.query(
            query_embeddings=[query_data['embedding']],
            n_results=topk,
        )
        result_docs = results['documents'][0]
        if return_val == "accuracy":
            hits = query_data['binding_partners'].intersection(set(result_docs))
            total_topk_hits += int(len(hits) > 0)
        elif return_val == "topk":
            top_hits[query_doc] = results['ids'][0]
    if return_val == "accuracy":
        print(f"Top-{topk} accuracy: {total_topk_hits / len(query_labels)}")
        return total_topk_hits / len(query_labels)
    elif return_val == "topk":
        for query, topk in top_hits.items():
            print(f"{query}:")
            print(f"\t{topk}")
        return top_hits

