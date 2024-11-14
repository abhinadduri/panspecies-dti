#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from torch.nn import CosineSimilarity
from torch.utils.data import DataLoader
from ultrafast.datamodules import EmbeddedDataset
from ultrafast.utils import TopK

def compute_topk_cli():
    parser = argparse.ArgumentParser(description='Find TopK similarities between library and query embeddings')
    parser.add_argument('--library-embeddings', type=str, required=True, help='Path to the library embeddings')
    parser.add_argument('--library-type', default="drug", choices=['drug','target'], help='Type of the library embeddings')
    parser.add_argument('--library-data', type=str, required=True, help='Path to the library data (csv)')
    parser.add_argument('--query-embeddings', type=str, required=True, help='Path to the query embeddings')
    parser.add_argument('--query-data', type=str, required=True, help='Path to the query data (csv)')
    parser.add_argument('-K', type=int, default=100, help='TopK similarities to return')
    parser.add_argument('--delimiter', type=str, default=',', help='Delimiter for the csv files')
    parser.add_argument('--batch-size', type=int, default=2048, help='Batch size for the dataloader')
    parser.add_argument('--verbose','-v',action='store_true',help='Print similarities')
    args = parser.parse_args()
    compute_topk(**vars(args))

def compute_topk(library_embeddings, library_data, query_embeddings, query_data, K=100, batch_size=2048, verbose=False, delimiter=',', library_type='drug'):
    assert library_type in ['drug','target'], "Library type should be either 'drug' or 'target'"
    library_embeddings = EmbeddedDataset(library_embeddings)
    query_embeddings = np.load(query_embeddings)
    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings[np.newaxis, :]
    query_data = pd.read_csv(query_data)
    query_ids = query_data['uniprot_id'].values if library_type == 'drug' else query_data['id'].values

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    query_embeddings = torch.tensor(query_embeddings).to(device)
    cosine_sim = CosineSimilarity(dim=1)
    # create a dataloader for the library embeddings
    # create a TopK object for each query
    topks = [TopK(K) for _ in range(query_embeddings.shape[0])]
    dataloader = DataLoader(library_embeddings, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for emb_mols, idxs in tqdm(dataloader, desc="Similarity", total=len(dataloader)):
            emb_mols = emb_mols.to(device)
            if query_embeddings.shape[0] == 1:
                similarities = cosine_sim(query_embeddings, emb_mols)
            else:
                # calculate the cosine similarity between the query and the library embeddings
                # the output shape should be (query_size, library_size)
                # tile the query embeddings to match the library embeddings
                emb_mols = emb_mols.unsqueeze(-1).repeat(1,1,query_embeddings.shape[0])
                similarities = cosine_sim(query_embeddings.T.unsqueeze(0), emb_mols).T

            if len(similarities.shape) == 1:
                similarities = similarities.unsqueeze(0)
            for i, similarity in enumerate(similarities):
                topks[i].push_list(similarity.cpu().numpy(), idxs)

            torch.cuda.empty_cache()

    # print the topk similarities and IDs
    lib_df = pd.read_csv(library_data, sep=delimiter)
    for i, topk in enumerate(topks):
        rec_id = query_ids[i]
        simi_and_idx = topk.get()
        idx = [int(x[1]) for x in simi_and_idx]
        simi = [x[0] for x in simi_and_idx]
        top_mols = lib_df[lib_df.index.isin(idx)]
        # sort the top_mols by the order of the idx
        top_mols = top_mols.reindex(idx)
        top_mols['CosineSimi'] = simi
        top_mols.to_csv(f"topk_mol_data_{rec_id}.csv", index=False)
        # use library_embeddings to get the embeddings of the topk molecules
        top_mol_emb = np.zeros((len(top_mols), emb_mols.shape[1]))
        for j, idx in enumerate(idx):
            top_mol_emb[j,:] = library_embeddings[idx][0].numpy()
        np.save(f"topk_mol_embeddings_{rec_id}.npy", top_mol_emb)
        if verbose:
            print(f"{rec_id}:{i}")
            for similarity, idx in simi_and_idx:
                print(f"Similarity: {similarity} ID: {lib_df.iloc[idx]['id']}")
