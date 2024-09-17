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
    args = parser.parse_args()
    report(**vars(args))


def report(data_file, embeddings, moltype, db_dir, db_name, topk):
    query_embeddings = np.load(embeddings, allow_pickle=True)
    query_embeddings = query_embeddings.tolist()
    df = pd.read_csv(data_file)
    if moltype == 'target':
        query_doc_col = 'Target Sequence'
        value_doc_col = 'SMILES'
    elif moltype == 'drug':
        query_doc_col = 'SMILES'
        value_doc_col = 'Target Sequence'
    query_documents = list(df[query_doc_col])
    value_documents = list(df[value_doc_col])
    # Usually we have data with many duplicate targets and drugs, so only store unique ones
    query_labels = {}
    for i, (query_doc, value_doc, query_embedding) in enumerate(zip(query_documents, value_documents, query_embeddings)):
        if query_doc not in query_labels:
            query_labels[query_doc] = {
                'binding_partners': {value_doc},
                'embedding': query_embedding,
            }
        else:
            query_labels[query_doc]['binding_partners'].add(value_doc)
    # Query the given database for the top-k results and log hits
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_or_create_collection(name=db_name, metadata={"hnsw:space": "cosine"})
    total_topk_hits = 0
    for query_doc, query_data in tqdm(query_labels.items()):
        # Todo: maybe batch queries
        results = collection.query(
            query_embeddings=[query_data['embedding']],
            n_results=topk,
        )
        result_docs = results['documents'][0]
        hits = query_data['binding_partners'].intersection(set(result_docs))
        total_topk_hits += int(len(hits) > 0)
    print(f"Top-{topk} accuracy: {total_topk_hits / len(query_labels)}")