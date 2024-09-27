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
    parser.add_argument('--output-path', type=str, required=True, help='path to save embeddings. Currently only supports numpy format.')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for inference')
    parser.add_argument('--map-size', type=int, default=10000, help="Map size limit for the LMDB")
    parser.add_argument('--num-workers', type=int, default=-1, help='Number of processes for featurization and DataLoading')
    parser.add_argument('--device', type=str, default=0, help='CUDA device. If CUDA is not available, this will be ignored.')
    parser.add_argument('--ext', type=str, default="h5", choices=['h5', 'lmdb', 'pt'], help='File format to store the drug and target features before co-embedding.')
    args = parser.parse_args()
    embed(**vars(args))


def embed(
    checkpoint: str,
    device: int,
    data_file: str,
    moltype: str,
    output_path: str,
    batch_size: int,
    ext: str,
    map_size: int,
    num_workers: int,
):
    args = argparse.Namespace(
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

    model = DrugTargetCoembeddingLightning.load_from_checkpoint(args.checkpoint)
    model.eval()
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{device}") if use_cuda else torch.device("cpu")
    model.to(device)
    
    # make directory for saving featurizations
    if os.path.dirname(args.output_path) != '' and not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
    if args.moltype == "drug":
        featurizer = get_featurizer(model.args.drug_featurizer, batch_size=args.batch_size, save_dir=os.path.dirname(args.output_path), ext=ext, n_jobs=args.num_workers, map_size=map_size)
    elif args.moltype == "target":
        featurizer = get_featurizer(model.args.target_featurizer, batch_size=args.batch_size, save_dir=os.path.dirname(args.output_path), ext=ext)
    featurizer = featurizer.to(device)

    dataset = EmbedDataset(args.data_file, args.moltype, featurizer)

    collate_fn = partial(embed_collate_fn, moltype=args.moltype)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    embeddings = []
    with torch.no_grad():
        for mols in tqdm(dataloader, desc="Embedding", total=len(dataloader)):
            mols = mols.to(device)
            emb = model.embed(mols, sample_type=args.moltype)
            embeddings.append(emb.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)

    np.save(args.output_path, embeddings)


if __name__ == '__main__':
    embed_cli()
