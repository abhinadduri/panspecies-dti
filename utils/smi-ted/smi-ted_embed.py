#/usr/bin/python3
import yaml
import hashlib

from argparse import Namespace, ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
import pyxis as px
import torch

from load import load_smi_ted

from tqdm import tqdm
from rdkit import Chem


def canonicalize(s):
    mol = Chem.MolFromSmiles(s, sanitize=False)
    if mol is not None:
        return Chem.MolToSmiles(mol,canonical=True, isomericSmiles=False)
    else:
        return s


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--csv-dir','-C',required=True,help='directory containing train/val/test csvs to embed')
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir).absolute()
    if not csv_dir.is_dir():
        raise ValueError(f"csv-dir is not a directory: {csv_dir}")
    elif 'MERGED' in str(csv_dir):
        raise NotImplementedError("Currently not set up for the MERGED database")

    model_smi_ted = load_smi_ted(
        folder='.',
        ckpt_filename='smi-ted-Light_40.pt'
    )

    seq_list = set()
    for csvs in ["train","val","test"]:
        seq_df = pd.read_csv(csv_dir / Path(f"{csvs}.csv"))
        seq_list |= set(seq_df['SMILES'].to_list())

    seq_dict = {hashlib.md5(seq.encode()).hexdigest(): seq for seq in seq_list}
    sorted_ids = sorted(seq_dict.keys())
    id_to_idx = {seq: idx for idx, seq in enumerate(sorted_ids)}
    np.save(csv_dir / Path('SMITED_id_to_idx.npy'), id_to_idx)
    db = px.Writer(dirpath=str(csv_dir / Path('SMITED_features.lmdb')), map_size_limit=10000, ram_gb_limit=10)

    batch_size = 64
    with torch.no_grad():
        for i in tqdm(range(0, len(seq_list), batch_size)):
            batch_ids = np.array(sorted_ids[i:i+batch_size])
            batch_seq = [canonicalize(seq_dict[idx]) for idx in batch_ids]

            with torch.no_grad():
                embedding = model_smi_ted.encode(batch_seq, return_torch=True)
            # average pooling over tokens
            breakpoint()

            db.put_samples('ids', batch_ids, 'feats', embedding.detach().cpu().numpy())
    db.close()
