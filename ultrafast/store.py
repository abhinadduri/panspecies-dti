import argparse
import pandas as pd
import numpy as np
import chromadb
from tqdm import tqdm


def store_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, required=True, help='Path to the data file')
    parser.add_argument('--embeddings', type=str, required=True, help='Path to the embeddings file')
    parser.add_argument('--moltype', type=str, required=True, help='Type of molecule (target or drug)')
    parser.add_argument('--db_dir', type=str, default='./dbs', help='Path to save the database(s)')
    parser.add_argument('--db_name', type=str, required=True, help='Name of the database under db_dir')
    args = parser.parse_args()
    store(**vars(args))


def store(data_file, embeddings, moltype, db_dir, db_name):
    embeddings = np.load(embeddings, allow_pickle=True)
    df = pd.read_csv(data_file)
    if moltype == 'target':
        doc_col = 'Target Sequence'
    elif moltype == 'drug':
        doc_col = 'SMILES'
    documents = list(df[doc_col])
    if 'id' in df.columns:
        ids = list(df['id'])
    else:
        ids = [str(i) for i in range(len(documents))]
    embeddings = embeddings.tolist()
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_or_create_collection(name=db_name, metadata={"hnsw:space": "cosine"})
    # Max upsert size tolerated by chromadb
    batch_size = 41660
    for i in tqdm(range(0, len(documents), batch_size)):
        collection.upsert(
            documents=documents[i:i+batch_size],
            ids=ids[i:i+batch_size],
            # metadatas=metadatas[i:i+batch_size],  # todo: add cli metadata support
            embeddings=embeddings[i:i+batch_size],
        )
    print(f"Stored {len(documents)} documents in the {db_name} collection of in the {db_dir} database")
    