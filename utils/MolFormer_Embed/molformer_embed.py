#/usr/bin/python3
import yaml
import hashlib

from argparse import Namespace, ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
import pyxis as px
import torch

from fast_transformers.masking import LengthMask as LM
from tqdm import tqdm
from rdkit import Chem

from tokenizer.tokenizer import MolTranBertTokenizer
from train_pubchem_light import LightningModule

def canonicalize(s):
    return Chem.MolToSmiles(Chem.MolFromSmiles(s, sanitize=False), canonical=True, isomericSmiles=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--csv-dir','-C',required=True,help='directory containing train/val/test csvs to embed')
    parser.add_argument('--save-all',action="store_true",help="Save all tokens, instead of each token")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir).absolute()
    if not csv_dir.is_dir():
        raise ValueError(f"csv-dir is not a directory: {csv_dir}")

    with open('hparams.yaml', 'r') as f:
        config = Namespace(**yaml.safe_load(f))

    tokenizer = MolTranBertTokenizer('bert_vocab.txt')

    ckpt = 'N-Step-Checkpoint_3_30000.ckpt'

    lm = LightningModule(config, tokenizer.vocab).load_from_checkpoint(ckpt, config=config, vocab=tokenizer.vocab)
    lm = lm.to('cuda') if torch.cuda.is_available() else lm.to('cpu')

    if 'MERGED' not in str(csv_dir):
        seq_list = set()
        for csvs in ["train","val","test"]:
            seq_df = pd.read_csv(csv_dir / Path(f"{csvs}.csv"))
            seq_list |= set(seq_df['SMILES'].to_list())

        seq_dict = {hashlib.md5(seq.encode()).hexdigest(): seq for seq in seq_list}
        sorted_ids = sorted(seq_dict.keys())
        id_to_idx = {seq: idx for idx, seq in enumerate(sorted_ids)}
        np.save(csv_dir / Path('MoLFormer_id_to_idx.npy'), id_to_idx)
    else:
        seq_dict = np.load(csv_dir / 'id_to_smiles.npy', allow_pickle=True).item()
        sorted_ids = sorted(seq_dict.keys())

    lmdb_path = 'MoLFormer_features.lmdb'
    if args.save_all:
        lmdb_path = 'MoLFormerSeq_features.lmdb'
        
    db = px.Writer(dirpath=str(csv_dir / Path(lmdb_path)), map_size_limit=10000, ram_gb_limit=10)

    lm.eval()
    batch_size = 512
    for i in tqdm(range(0, len(sorted_ids), batch_size)):
        batch_ids = np.array(sorted_ids[i:i+batch_size])
        batch_seq = [canonicalize(seq_dict[idx]) for idx in batch_ids]

        batch_enc = tokenizer.batch_encode_plus(batch_seq, padding=True, add_special_tokens=True)
        idx, mask = torch.tensor(batch_enc['input_ids']).to(lm.device), torch.tensor(batch_enc['attention_mask']).to(lm.device)

        with torch.no_grad():
            token_embeddings = lm.blocks(lm.tok_emb(idx), length_mask=LM(mask.sum(-1)))

        if not args.save_all:
            # average pooling over tokens
            input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask

            db.put_samples('ids', batch_ids, 'feats', embedding.detach().cpu().numpy())
        else:
            for i, ids in enumerate(batch_ids):
                seq_len = mask[i,:].squeeze().sum()
                feats = token_embeddings[i].squeeze()[:seq_len].detach().cpu().numpy()

                db.put_samples('ids', np.array(ids)[np.newaxis,...], 'feats', feats[np.newaxis,...])
            
    db.close()


