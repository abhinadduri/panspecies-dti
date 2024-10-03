#!/usr/bin/env python
import argparse
import heapq
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from torch.nn import CosineSimilarity
from torch.utils.data import DataLoader
from ultrafast.datamodules import EmbeddedDataset, embedded_collate_fn

# create a class that keeps track of the topk cosine_similarities and their IDs
class TopK:
    def __init__(self, topk):
        self.topk = topk
        self.data = []
    def push(self, similarity, id):
        if len(self.data) < self.topk:
            heapq.heappush(self.data, (similarity, id))
        elif similarity > self.data[0][0]:
            heapq.heappushpop(self.data,(similarity, id))
    def get(self):
        return sorted(self.data, reverse=True)
    # write a method that pushes a list of similarities and IDs
    def push_list(self, similarities, ids):
        for similarity, id in zip(similarities, ids):
            self.push(similarity, id)

def argparse_topk():
    parser = argparse.ArgumentParser(description='Topk similarity search')
    parser.add_argument('--library_embeddings', type=str, required=True, help='Path to the library embeddings')
    parser.add_argument('--library_type', default="drug", choices=['drug','target'], help='Type of the library embeddings')
    parser.add_argument('--library_data', type=str, required=True, help='Path to the library data (csv)')
    parser.add_argument('--delimiter', type=str, default=',', help='Delimiter for the csv files')
    parser.add_argument('--query_embeddings', type=str, required=True, help='Path to the query embeddings')
    parser.add_argument('--query_data', type=str, required=True, help='Path to the query data (csv)')
    parser.add_argument('--topk', type=int, default=100, help='Topk to consider for similarity')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size for the dataloader')
    return parser.parse_args()

def topk(args):
    library_embeddings = EmbeddedDataset(args.library_embeddings)
    query_embeddings = np.load(args.query_embeddings)
    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings[np.newaxis, :]
    query_data = pd.read_csv(args.query_data)
    query_ids = query_data['uniprot_id'].values if args.library_type == 'drug' else query_data['id'].values

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    query_embeddings = torch.tensor(query_embeddings).to(device)
    cosine_sim = CosineSimilarity(dim=1)
    # create a dataloader for the library embeddings
    # create a TopK object for each query
    topks = [TopK(args.topk) for _ in range(query_embeddings.shape[0])]
    dataloader = DataLoader(library_embeddings, batch_size=args.batch_size, shuffle=False, collate_fn=embedded_collate_fn)
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
    lib_df = pd.read_csv(args.library_data, sep=args.delimiter)
    for i, topk in enumerate(topks):
        rec_id = query_ids[i]
        simi_and_idx = topk.get()
        idx = [x[1] for x in simi_and_idx]
        top_mols = lib_df[lib_df.index.isin(idx)]
        # sort the top_mols by the order of the idx
        top_mols = top_mols.reindex(idx)
        top_mols.to_csv(f"topk_mol_data_{rec_id}.csv", index=False)
        # use library_embeddings to get the embeddings of the topk molecules
        top_mol_emb = np.zeros((len(top_mols), emb_mols.shape[1]))
        for j, idx in enumerate(idx):
            top_mol_emb[j,:] = library_embeddings[idx][0].numpy()
        np.save(f"topk_mol_embeddings_{rec_id}.npy", top_mol_emb)
        for similarity, idx in simi_and_idx:
            print(f"Similarity: {similarity} ID: {lib_df.iloc[idx]['id']}")
        # for similarity in topk.get():
        #     print(f"Similarity: {similarity}")
    # for rec_seq, hits in sorted(top_hits.items()):


if __name__ == '__main__':
    args = argparse_topk()
    topk(args)
