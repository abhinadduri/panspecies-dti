import os
import argparse
from functools import partial
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from ultrafast.datamodules import EmbedDataset, embed_collate_fn
from ultrafast.model import DrugTargetCoembeddingLightning
from ultrafast.utils import get_featurizer


def embed_cli():
    parser = argparse.ArgumentParser(description='Generate embeddings from DrugTargetCoembeddingLightning model')
    parser.add_argument('--data-file', type=str, required=True, help='Path to file containing molecules to embed, in tsv format. With header and columns: "SMILES" for drugs, "Target Sequence" for targets')
    parser.add_argument("--moltype", type=str, help="Molecule type", choices=["drug", "target"], default="target")

    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_path', type=str, required=True, help='path to save embeddings. Currently only supports numpy format.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for inference')
    parser.add_argument('--device', type=str, default=0, help='CUDA device. If CUDA is not available, this will be ignored.')
    args = parser.parse_args()
    embed(**vars(args))


def embed(
    checkpoint: str,
    device: int,
    data_file: str,
    moltype: str,
    output_path: str,
    batch_size: int,
):
    args = argparse.Namespace(
        checkpoint=checkpoint,
        device=device,
        data_file=data_file,
        moltype=moltype,
        output_path=output_path,
        batch_size=batch_size
    )
    model = DrugTargetCoembeddingLightning.load_from_checkpoint(args.checkpoint)
    model.eval()
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{device}") if use_cuda else torch.device("cpu")
    model.to(device)
    
    if args.moltype == "drug":
        featurizer = get_featurizer(model.args.drug_featurizer)
    elif args.moltype == "target":
        featurizer = get_featurizer(model.args.target_featurizer)
    featurizer = featurizer.to(device)

    dataset = EmbedDataset(args.data_file, args.moltype, featurizer)

    collate_fn = partial(embed_collate_fn, moltype=args.moltype)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    embeddings = []
    with torch.no_grad():
        for mols in tqdm(dataloader, desc="Embedding", total=len(dataloader)):
            mols = mols.to(device)
            emb = model.embed(mols, sample_type=args.moltype)
            embeddings.append(emb.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)

    # if output_path contains directories that do not exist, create them
    if os.path.dirname(args.output_path) != '' and not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    np.save(args.output_path, embeddings)


if __name__ == '__main__':
    embed_cli()
