import os
import argparse
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from datamodules import EmbedDataset
from model import DrugTargetCoembeddingLightning
from utils import get_featurizer

def get_args():
    parser = argparse.ArgumentParser(description='Generate embeddings from DrugTargetCoembeddingLightning model')
    parser.add_argument('--data-file', type=str, help='Path to file containing molecules to embed')
    parser.add_argument("--moltype", type=str, help="Molecule type", choices=["drug", "target"], default="target")])

    parser.add_argument("--drug-featurizer", help="Drug featurizer", dest="drug_featurizer", default="MorganFeaturizer")
    parser.add_argument("--target-featurizer", help="Target featurizer", dest="target_featurizer", default="ESM2Featurizer")

    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--output_path', type=str, help='path to save embeddings. Currently only supports numpy format.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for inference')
    parser.add_argument('--device', type=str, default=0, help='CUDA device. If CUDA is not available, this will be ignored.')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    drug_featurizer = get_featurizer(config.drug_featurizer)
    target_featurizer = get_featurizer(config.target_featurizer)

    model = DrugTargetCoembeddingLightning.load_from_checkpoint(args.checkpoint)
    model.eval()
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.device}") if use_cuda else torch.device("cpu")
    model.to(device)
    
    dataset = EmbedDataset(args.data_file, moltype, drug_featurizer, target_featurizer)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    embeddings = []
    with torch.no_grad():
        for mols in tqdm(dataloader):
            mols = mols.to(device)
            emb = model.embed(batch, sample_type=args.moltype)
            embeddings.append(emb.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    np.save(args.output_path, embeddings)


