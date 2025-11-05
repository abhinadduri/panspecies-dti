from __future__ import annotations

import torch
import os
import hashlib

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import pyxis as px
import typing as T


from multiprocessing import Pool
from numpy.random import choice
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
from tdc.benchmark_group import dti_dg_group
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Optional
from ultrafast.featurizers import Featurizer
from ultrafast.tdc_utils import compute_ESM_features, get_saprot_seq

def get_task_dir(task_name: str):
    """
    Get the path to data for each benchmark data set

    :param task_name: Name of benchmark
    :type task_name: str
    """

    task_paths = {
        "unittest": "./data/unittest_dummy_data",
        "biosnap": "./data/BIOSNAP/full_data",
        "biosnap_prot": "./data/BIOSNAP/unseen_protein",
        "biosnap_mol": "./data/BIOSNAP/unseen_drug",
        "bindingdb": "./data/BindingDB",
        "davis": "./data/DAVIS",
        "dti_dg": "./data/TDC",
        "dude": "./data/DUDe",
        "halogenase": "./data/EnzPred/halogenase_NaCl_binary",
        "bkace": "./data/EnzPred/duf_binary",
        "gt": "./data/EnzPred/gt_acceptors_achiral_binary",
        "esterase": "./data/EnzPred/esterase_binary",
        "kinase": "./data/EnzPred/davis_filtered",
        "phosphatase": "./data/EnzPred/phosphatase_chiral_binary",
        "merged": "./data/MERGED/huge_data",
        "custom": "./data/custom/",
    }

    return Path(task_paths[task_name.lower()]).resolve()

def compute_similar_sequences_single_target(target_sequence, train_sequences, threshold=0.9, tmp_dir='tmp_mmseqs'):
    """
    Compute sequence similarity for target protein(s) against training sequences using MMSeqs2.

    Args:
        target_sequence: Single target protein sequence string or list of sequences
        train_sequences: Dictionary mapping UniProt IDs to sequences
        threshold: Similarity threshold (0.0-1.0) for filtering
        tmp_dir: Temporary directory for MMSeqs2 files

    Returns:
        Set of UniProt IDs that have similarity >= threshold to any target sequence
    """
    import subprocess
    import tempfile

    os.makedirs(tmp_dir, exist_ok=True)

    # Create temporary FASTA files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False, dir=tmp_dir) as target_fasta:
        # Handle single sequence or list of sequences
        if isinstance(target_sequence, list):
            for idx, seq in enumerate(target_sequence):
                target_fasta.write(f">target_{idx}\n{seq}\n")
        else:
            target_fasta.write(f">target\n{target_sequence}\n")
        target_fasta_path = target_fasta.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False, dir=tmp_dir) as train_fasta:
        for seq_id, seq in train_sequences.items():
            train_fasta.write(f">{seq_id}\n{seq}\n")
        train_fasta_path = train_fasta.name

    # Create output file path
    output_file = os.path.join(tmp_dir, 'similar_sequences.tsv')

    try:
        # Run MMSeqs2
        commands = [
            f"mmseqs createdb {train_fasta_path} {tmp_dir}/queryDB",
            f"mmseqs createdb {target_fasta_path} {tmp_dir}/targetDB",
            f"mmseqs search {tmp_dir}/queryDB {tmp_dir}/targetDB {tmp_dir}/resultDB {tmp_dir} -s 7.5 --min-seq-id {threshold}",
            f"mmseqs convertalis {tmp_dir}/queryDB {tmp_dir}/targetDB {tmp_dir}/resultDB {output_file}"
        ]

        for cmd in commands:
            subprocess.run(cmd, shell=True, check=True, capture_output=True)

        # Process results
        similar_ids = set()
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                for line in f:
                    query_id = line.split('\t')[0]
                    similar_ids.add(query_id)

        return similar_ids

    finally:
        # Clean up temporary files
        for pattern in ['queryDB*', 'targetDB*', 'resultDB*', target_fasta_path, train_fasta_path, output_file]:
            subprocess.run(f"rm -f {tmp_dir}/{pattern} 2>/dev/null || true", shell=True)

def embed_collate_fn(args: T.Tuple[torch.Tensor, torch.Tensor], moltype="target"):
    """
    Collate function for PyTorch data loader -- turn a batch of molecules into a batch of tensors

    :param args: Batch of molecules
    :type args: Iterable[Tuple[torch.Tensor, torch.Tensor]]
    :param moltype: Molecule type
    :type moltype: str
    :return: Create a batch of examples
    :rtype: torch.Tensor
    """
    # m_emb = [a for a in args]
    if isinstance(args[0],list):
        args = [a[0] for a in args]


    if moltype == "drug":
        mols = torch.stack(args, 0)
    elif moltype == "target":
        mols = pad_sequence(args, batch_first=True)
    else:
        raise ValueError("moltype must be one of ['drug', 'target']")

    return mols

def drug_target_collate_fn(args: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    """
    Collate function for PyTorch data loader -- turn a batch of triplets into a triplet of batches

    :param args: Batch of training samples with molecule, protein, and affinity
    :type args: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    :return: Create a batch of examples
    :rtype: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    d_emb = [a[0] for a in args]
    t_emb = [a[1] for a in args]
    labs = [a[2] for a in args]

    drugs = torch.stack(d_emb, 0)
    targets = pad_sequence(t_emb, batch_first=True)
    labels = torch.stack(labs, 0)

    return drugs, targets, labels

def contrastive_collate_fn(args: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    """
    Collate function for PyTorch data loader -- turn a batch of triplets into a triplet of batches

    Specific collate function for contrastive dataloader

    :param args: Batch of training samples with anchor, positive, negative
    :type args: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    :return: Create a batch of examples
    :rtype: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    anchor_emb = [a[0] for a in args]
    pos_emb = [a[1] for a in args]
    neg_emb = [a[2] for a in args]

    anchors = pad_sequence(anchor_emb, batch_first=True)
    positives = torch.stack(pos_emb, 0)
    negatives = torch.stack(neg_emb, 0)

    return anchors, positives, negatives

def make_contrastive(
        df: pd.DataFrame,
        posneg_column: str,
        anchor_column: str,
        label_column: str,
        n_neg_per: int = 50,
    ):

    pos_df = df[df[label_column] == 1]
    neg_df = df[df[label_column] == 0]

    contrastive = []

    for _, r in pos_df.iterrows():
        for _ in range(n_neg_per):
            contrastive.append((r[anchor_column], r[posneg_column], choice(neg_df[posneg_column])))

    contrastive = pd.DataFrame(
        contrastive, columns=["Anchor", "Positive", "Negative"]
    )
    return contrastive

def make_diffprot_contrastive(
        df: pd.DataFrame,
        posneg_column: str,
        anchor_column: str,
        label_column: str,
        n_neg_per: int = 50,
    ):

    pos_df = df[df[label_column] == 1]

    contrastive = []

    for _, r in pos_df.iterrows():
        neg_df = pos_df[r[anchor_column] != pos_df[anchor_column]] # get all rows where the anchor is not the same
        for _ in range(n_neg_per):
            contrastive.append((r[anchor_column], r[posneg_column], choice(neg_df[posneg_column])))
    contrastive = pd.DataFrame(
        contrastive, columns=["Anchor", "Positive", "Negative"]
    )

    return contrastive

class BinaryDataset(Dataset):
    def __init__(
        self,
        drugs,
        targets,
        labels,
        drug_featurizer: Featurizer,
        target_featurizer: Featurizer,
    ):
        self.drugs = drugs
        self.targets = targets
        self.labels = labels

        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer

    def __len__(self):
        return len(self.drugs)

    def __getitem__(self, i: int):
        drug = self.drug_featurizer(self.drugs.iloc[i])
        target = self.target_featurizer(self.targets.iloc[i])
        label = torch.tensor(self.labels.iloc[i], dtype=torch.float32)

        return drug, target, label

class ContrastiveDataset(Dataset):
    def __init__(
        self,
        anchors,
        positives,
        negatives,
        posneg_featurizer: Featurizer,
        anchor_featurizer: Featurizer,
    ):
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives

        self.posneg_featurizer = posneg_featurizer
        self.anchor_featurizer = anchor_featurizer

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, i):

        anchorEmb = self.anchor_featurizer(self.anchors[i])
        positiveEmb = self.posneg_featurizer(self.positives[i])
        negativeEmb = self.posneg_featurizer(self.negatives[i])

        return anchorEmb, positiveEmb, negativeEmb

class EmbedInMemoryDataset(Dataset):
    def __init__(
        self,
        data: List[str],
        featurizer: Featurizer,
    ):
        self.data = data
        self.featurizer = featurizer

        print("Featurizing the data")
        self.featurizer.preload(self.data, write_first=True, seq_func=featurizer.prepare_string)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        s = self.featurizer.prepare_string(self.data[i])
        item = self.featurizer.features[s]

        return item

class EmbeddedDataset(Dataset):
    def __init__(self,
                 emb_file: str,
                 ):
        self.db, self.data = None, None
        if emb_file.endswith(".npy"):
            self.data = np.load(emb_file, mmap_mode="r")
        elif emb_file.endswith(".lmdb"):
            self.db = px.Reader(dirpath=str(emb_file), lock=False) # we only read

    def __len__(self):
        return len(self.data) if self.data is not None else len(self.db)

    def __getitem__(self, i):
        if self.data is not None:
            return torch.tensor(self.data[i]), i
        else:
            return torch.tensor(self.db[str(i)]['feats']), i

    def teardown(self):
        if self.db is not None:
            self.db.close()

class EmbedDataset(Dataset):
    def __init__(
        self,
        data_file: str,
        moltype: str,
        featurizer: Featurizer,
    ):
        self.data = pd.read_table(data_file, header=0, sep=None)
        self.moltype = moltype

        self.featurizer = featurizer

        self._column = "SMILES" if self.moltype == "drug" else "Target Sequence"
        print("Featurizing the data")
        self.featurizer.preload(self.data[self._column].unique().tolist(), write_first=True, seq_func=featurizer.prepare_string)
        self.db = None
        if str(self.featurizer._save_path).endswith("lmdb"):
            self.db = px.Reader(dirpath=str(self.featurizer._save_path), lock=False) # we only read

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if self.db is None:
            seq = self.featurizer.prepare_string(self.data[self._column].iloc[i])
            mol = self.featurizer.features[seq]
        else:
            mol = torch.from_numpy(self.db[i]['feats'])


        return mol

    def teardown(self, stage):
        if self.db is not None:
            self.db.close()

class DTIDataModule(pl.LightningDataModule):
    """ DataModule used for training on drug-target interaction data.
    Uses the following data sets:
    - biosnap
    - biosnap_prot
    - biosnap_mol
    - bindingdb
    - davis
    """
    def __init__(
            self,
            data_dir: str,
            drug_featurizer: Featurizer,
            target_featurizer: Featurizer,
            device: torch.device = torch.device("cpu"),
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: int = 0,
            header=0,
            index_col=0,
            sep=",",
        ):
        super().__init__()

        self._loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": drug_target_collate_fn,
        }

        self._csv_kwargs = {
            "header": header,
            "index_col": index_col,
            "sep": sep,
        }

        self._device = device

        self._data_dir = Path(data_dir)
        self._train_path = Path("train.csv")
        self._val_path = Path("val.csv")
        self._test_path = Path("test.csv")

        self._drug_column = "SMILES"
        self._target_column = "Target Sequence"
        self._label_column = "Label"

        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer

        self.drug_featurizer.ext = ".lmdb"
        self.target_featurizer.ext = ".lmdb"
        self.drug_featurizer._save_path = self.drug_featurizer.path.with_suffix(self.drug_featurizer.ext)
        self.target_featurizer._save_path = self.target_featurizer.path.with_suffix(self.target_featurizer.ext)
        if self.target_featurizer.name == "SaProt":
            self._train_path = Path("train_foldseek.csv")
            self._val_path = Path("val_foldseek.csv")
            self._test_path = Path("test_foldseek.csv")

        self.drug_db, self.target_db = None, None

    def prepare_data(self):
        """
        Featurize drugs and targets and save them to disk if they don't already exist
        """

        print(f"drug feat path: {self.drug_featurizer.path}\ntarget path:{self.target_featurizer.path}")
        if self.drug_featurizer.path.exists() and self.target_featurizer.path.exists():
            print("Drug and target featurizers already exist")
            return

        print(self._train_path)
        df_train = pd.read_csv(self._data_dir / self._train_path, **self._csv_kwargs, dtype={self._target_column: str})

        df_val = pd.read_csv(self._data_dir / self._val_path, **self._csv_kwargs, dtype={self._target_column: str})

        df_test = pd.read_csv(self._data_dir / self._test_path, **self._csv_kwargs, dtype={self._target_column: str})

        dataframes = [df_train, df_val, df_test]
        all_drugs = pd.concat([i[self._drug_column] for i in dataframes]).unique()
        all_targets = pd.concat([i[self._target_column] for i in dataframes]).unique()

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        if not self.drug_featurizer.path.exists():
            self.drug_featurizer.write_to_disk(all_drugs, file_path=self.drug_featurizer.path)

        if not self.target_featurizer.path.exists():
            self.target_featurizer.write_to_disk(all_targets, file_path=self.target_featurizer.path)

        self.drug_featurizer.cpu()
        self.target_featurizer.cpu()

    def setup(self, stage = None):
        self.df_train = pd.read_csv(self._data_dir / self._train_path, **self._csv_kwargs, dtype={self._target_column: str})
        self.df_val = pd.read_csv(self._data_dir / self._val_path, **self._csv_kwargs, dtype={self._target_column: str})
        self.df_test = pd.read_csv(self._data_dir / self._test_path, **self._csv_kwargs, dtype={self._target_column: str})

        self._dataframes = [self.df_train, self.df_val, self.df_test]

        all_drugs = pd.concat([i[self._drug_column] for i in self._dataframes]).unique()
        all_targets = pd.concat([i[self._target_column] for i in self._dataframes]).unique()

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        self.drug_featurizer.preload(all_drugs)
        self.drug_featurizer.cpu()

        self.target_featurizer.preload(all_targets)
        self.target_featurizer.cpu()

        if stage == "fit" or stage is None:
            self.data_train = BinaryDataset(
                self.df_train[self._drug_column],
                self.df_train[self._target_column],
                self.df_train[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

            self.data_val = BinaryDataset(
                self.df_val[self._drug_column],
                self.df_val[self._target_column],
                self.df_val[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

        if stage == "test" or stage is None:
            self.data_test = BinaryDataset(
                self.df_test[self._drug_column],
                self.df_test[self._target_column],
                self.df_test[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

    def train_dataloader(self):
        return DataLoader(self.data_train, **self._loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.data_val, **self._loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.data_test, **self._loader_kwargs)

    def teardown(self, stage:str):
        self.drug_featurizer.teardown(stage)
        self.target_featurizer.teardown(stage)


class TDCDataModule(pl.LightningDataModule):
    """ DataModule used for training on drug-target interaction data.
    Uses the dti_dg dataset
    """
    def __init__(
            self,
            data_dir: str,
            drug_featurizer: Featurizer,
            target_featurizer: Featurizer,
            device: torch.device = torch.device("cpu"),
            seed: int = 0,
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: int = 0,
            header=0,
            index_col=0,
            sep=",",
        ):
        super().__init__()

        self._loader_kwargs = {
                "batch_size": batch_size,
                "shuffle": shuffle,
                "num_workers": num_workers,
                "collate_fn": drug_target_collate_fn,
                }

        self._csv_kwargs = {
                "header": header,
                "index_col": index_col,
                "sep": sep,
                }

        self._device = device

        self._data_dir = Path(data_dir)
        self._seed = seed

        self._drug_column = "Drug"
        self._target_column = "Target"
        self._label_column = "Y"

        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer
        if self.target_featurizer.name == "SaProt":
            self._target_column = "Target Structure"
            self.target_struc_dict = None

        self.drug_featurizer.ext = ".lmdb"
        self.target_featurizer.ext = ".lmdb"
        self.drug_featurizer._save_path = self.drug_featurizer.path.with_suffix(self.drug_featurizer.ext)
        self.target_featurizer._save_path = self.target_featurizer.path.with_suffix(self.target_featurizer.ext)

        self.dg_group = dti_dg_group(path=self._data_dir)
        self.dg_benchmark = self.dg_group.get("bindingdb_patent")

    def prepare_data(self):

        train_val, test = (
                self.dg_benchmark["train_val"],
                self.dg_benchmark["test"],
                )

        all_drugs = pd.concat([train_val, test])[self._drug_column].unique()
        if self._target_column == "Target Structure":
            target_ids = pd.concat([train_val, test])["Target_ID"].unique()
            if (self._data_dir / Path("target_struc_dict.npy")).exists():
                self.target_struc_dict = np.load(self._data_dir / Path("target_struc_dict.npy"), allow_pickle=True).item()
            else:
                # get the sequence for each target_id in target_ids, using the Target column
                print("Computing SaProt sequences for TDC data with AFDB structures")
                self.target_struc_dict = self.compute_structure_features(target_ids)
            
            not_found = []
            for target_id in target_ids:
                if self.target_struc_dict[target_id] is None:
                    not_found.append(target_id)
            if len(not_found) > 0:
                print(f"Could not find sequences for {len(not_found)} targets")
                # get the sequence for each target_id in not_found, using the Target column
                not_found = pd.concat([train_val, test]).loc[pd.concat([train_val, test])["Target_ID"].isin(not_found)]
                # create a dictionary with the target_id as key and the sequence as value
                not_found_dict = dict(zip(not_found["Target_ID"], not_found["Target"]))

                for tid in not_found_dict.keys():
                    not_found_dict[tid] = '#'.join(list(not_found_dict[tid])) # just mask the structure tokens
                print(f"Masking the structure tokens for the {len(not_found)} examples")
                # esm_struct_dict = compute_ESM_features(not_found_dict)
                # # update the target_struc_dict with the sequences from esm_struct_dict
                self.target_struc_dict.update(not_found_dict)
            # add the sequences to the train_val and test dataframes
            train_val[self._target_column] = train_val["Target_ID"].map(self.target_struc_dict)
            test[self._target_column] = test["Target_ID"].map(self.target_struc_dict)
            np.save(self._data_dir / Path("target_struc_dict.npy"), self.target_struc_dict)
        all_targets = pd.concat([train_val, test])[self._target_column].unique()

        if self.drug_featurizer.path.exists() and self.target_featurizer.path.exists():
            print("Drug and target featurizers already exist")
            return

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        if not self.drug_featurizer.path.exists():
            self.drug_featurizer.write_to_disk(all_drugs)

        if not self.target_featurizer.path.exists():
            self.target_featurizer.write_to_disk(all_targets)

        self.drug_featurizer.cpu()
        self.target_featurizer.cpu()

    def setup(self, stage: T.Optional[str] = None):

        dg_name = self.dg_benchmark["name"]

        self.df_train, self.df_val = self.dg_group.get_train_valid_split(
                benchmark=dg_name, 
                split_type="default", 
                seed=self._seed
                )
        self.df_test = self.dg_benchmark["test"]
        if self._target_column == "Target Structure":
            assert self.target_struc_dict is not None
            self.df_train[self._target_column] = self.df_train["Target_ID"].map(self.target_struc_dict)
            self.df_val[self._target_column] = self.df_val["Target_ID"].map(self.target_struc_dict)
            self.df_test[self._target_column] = self.df_test["Target_ID"].map(self.target_struc_dict)

        self._dataframes = [self.df_train, self.df_val, self.df_test]

        all_drugs = pd.concat([i[self._drug_column] for i in self._dataframes]).unique()
        all_targets = pd.concat([i[self._target_column] for i in self._dataframes]).unique()

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        self.drug_featurizer.preload(all_drugs)
        self.drug_featurizer.cpu()

        self.target_featurizer.preload(all_targets)
        self.target_featurizer.cpu()

        if stage == "fit" or stage is None:
            self.data_train = BinaryDataset(
                self.df_train[self._drug_column],
                self.df_train[self._target_column],
                self.df_train[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

            self.data_val = BinaryDataset(
                self.df_val[self._drug_column],
                self.df_val[self._target_column],
                self.df_val[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

        if stage == "test" or stage is None:
            self.data_test = BinaryDataset(
                self.df_test[self._drug_column],
                self.df_test[self._target_column],
                self.df_test[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

    def compute_structure_features(self, target_ids):
        """
        Compute structure features for the target proteins
        """
        with Pool(self._loader_kwargs['num_workers']) as p:
            results = p.map(get_saprot_seq, target_ids)
        ids, seqs = zip(*results)
        structure_seqs = dict(zip(ids, seqs))
            
        return structure_seqs


    def train_dataloader(self):
        return DataLoader(self.data_train, **self._loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.data_val, **self._loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.data_test, **self._loader_kwargs)

class EnzPredDataModule(pl.LightningDataModule):
    """ DataModule used for training on drug-target interaction for enzymes.
    Uses the following data sets:
    - halogenase
    - bkace
    - gt
    - esterase
    - kinase
    - phosphatase
    """
    def __init__(
            self,
            data_dir: str,
            drug_featurizer: Featurizer,
            target_featurizer: Featurizer,
            device: torch.device = torch.device("cpu"),
            seed: int = 0,
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: int = 0,
            header=0,
            index_col=0,
            sep=",",
        ):
        super().__init__()

        self._loader_kwargs = {
                "batch_size": batch_size,
                "shuffle": shuffle,
                "num_workers": num_workers,
                "collate_fn": drug_target_collate_fn,
                }

        self._csv_kwargs = {
                "header": header,
                "index_col": index_col,
                "sep": sep,
                }

        self._device = device

        self._data_file = Path(data_dir).with_suffix(".csv")
        self._data_stem = Path(self._data_file.stem)
        self._data_dir = self._data_file.parent / self._data_file.stem
        self._seed = 0
        self._replicate = seed

        df = pd.read_csv(self._data_file, index_col=0)
        self._drug_column = df.columns[1]
        self._target_column = df.columns[0]
        self._label_column = df.columns[2]

        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer

    @classmethod
    def dataset_list(cls):
        return [
                "halogenase",
                "bkace",
                "gt",
                "esterase",
                "kinase",
                "phosphatase",
                ]

    def prepare_data(self):

        os.makedirs(self._data_dir, exist_ok=True)

        kfsplitter = KFold(n_splits=10, shuffle=True, random_state=self._seed)
        full_data = pd.read_csv(self._data_file, index_col=0)

        all_drugs = full_data[self._drug_column].unique()
        all_targets = full_data[self._target_column].unique()

        if self.drug_featurizer.path.exists() and self.target_featurizer.path.exists():
            print("Drug and target featurizers already exist")

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        if not self.drug_featurizer.path.exists():
            self.drug_featurizer.write_to_disk(all_drugs)

        if not self.target_featurizer.path.exists():
            self.target_featurizer.write_to_disk(all_targets)

        self.drug_featurizer.cpu()
        self.target_featurizer.cpu()

        for i, split in enumerate(kfsplitter.split(full_data)):
            fold_train = full_data.iloc[split[0]].reset_index(drop=True)
            fold_test = full_data.iloc[split[1]].reset_index(drop=True)
            #logg.debug(self._data_dir / self._data_stem.with_suffix(f".{i}.train.csv"))
            fold_train.to_csv(
                    self._data_dir / self._data_stem.with_suffix(f".{i}.train.csv"),
                    index=True,
                    header=True,
                )
            fold_test.to_csv(
                    self._data_dir / self._data_stem.with_suffix(f".{i}.test.csv"),
                    index=True,
                    header=True,
                )

    def setup(self, stage: T.Optional[str] = None):

        df_train = pd.read_csv(
                self._data_dir / self._data_stem.with_suffix(f".{self._replicate}.train.csv"),
                index_col=0,
            )
        self.df_train, self.df_val = train_test_split(df_train, test_size=0.1)
        self.df_test = pd.read_csv(
                self._data_dir / self._data_stem.with_suffix(f".{self._replicate}.test.csv"),
                index_col=0,
            )

        self._dataframes = [self.df_train, self.df_val, self.df_test]

        all_drugs = pd.concat([i[self._drug_column] for i in self._dataframes]).unique()
        all_targets = pd.concat([i[self._target_column] for i in self._dataframes]).unique()

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        self.drug_featurizer.preload(all_drugs)
        self.drug_featurizer.cpu()

        self.target_featurizer.preload(all_targets)
        self.target_featurizer.cpu()

        if stage == "fit" or stage is None:
            self.data_train = BinaryDataset(
                self.df_train[self._drug_column],
                self.df_train[self._target_column],
                self.df_train[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

            self.data_val = BinaryDataset(
                self.df_val[self._drug_column],
                self.df_val[self._target_column],
                self.df_val[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

        if stage == "test" or stage is None:
            self.data_test = BinaryDataset(
                self.df_test[self._drug_column],
                self.df_test[self._target_column],
                self.df_test[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

    def train_dataloader(self):
        return DataLoader(self.data_train, **self._loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.data_val, **self._loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.data_test, **self._loader_kwargs)

class DUDEDataModule(pl.LightningDataModule):
    def __init__(
            self,
            contrastive_split: str,
            drug_featurizer: Featurizer,
            target_featurizer: Featurizer,
            contrastive_type: str = "default",
            device: torch.device = torch.device("cpu"),
            n_neg_per: int = 50,
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: int = 0,
            header=0,
            index_col=None,
            sep="\t",
        ):
        super().__init__()

        self._loader_kwargs = {
                "batch_size": batch_size,
                "shuffle": shuffle,
                "num_workers": num_workers,
                "collate_fn": contrastive_collate_fn,
            }

        self._csv_kwargs = {
                "header": header,
                "index_col": index_col,
                "sep": sep,
            }

        self._device = device
        self._n_neg_per = n_neg_per

        self._data_dir = Path("./data/DUDe/")
        self._split = contrastive_split
        self._split_path = self._data_dir / Path(f"dude_{self._split}_type_train_test_split.csv")

        self._drug_id_column = "Molecule_ID"
        self._drug_column = "Molecule_SMILES"
        self._target_id_column = "Target_ID"
        self._target_column = "Target_Seq"
        self._label_column = "Label"

        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer

        assert contrastive_type in ["default", "diffprot"], "Contrastive type must be one of ['default', 'diffprot']"
        self.contrastive_type = contrastive_type

    def prepare_data(self):
        pass

    def setup(self, stage: T.Optional[str] = None):

        self.df_full = pd.read_csv(self._data_dir / Path("full.tsv"), **self._csv_kwargs)

        self.df_splits = pd.read_csv(self._split_path, header=None)
        self._train_list = self.df_splits[self.df_splits[1] == "train"][0].values
        self._test_list = self.df_splits[self.df_splits[1] == "test"][0].values

        self.df_train = self.df_full[self.df_full[self._target_id_column].isin(self._train_list)]
        self.df_test = self.df_full[self.df_full[self._target_id_column].isin(self._test_list)]

        if self.contrastive_type == "diffprot":
            self.train_contrastive = make_diffprot_contrastive(
                    self.df_train,
                    self._drug_column,
                    self._target_column,
                    self._label_column,
                    self._n_neg_per,
                )

        elif self.contrastive_type == "default":
            self.train_contrastive = make_contrastive(
                    self.df_train,
                    self._drug_column,
                    self._target_column,
                    self._label_column,
                    self._n_neg_per,
                )

        self._dataframes = [self.df_train]  # , self.df_test]

        all_drugs = pd.concat([i[self._drug_column] for i in self._dataframes]).unique()
        all_targets = pd.concat([i[self._target_column] for i in self._dataframes]).unique()

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        self.drug_featurizer.preload(all_drugs, write_first=True)
        self.drug_featurizer.cpu()

        self.target_featurizer.preload(all_targets, write_first=True)
        self.target_featurizer.cpu()

        if stage == "fit" or stage is None:
            self.data_train = ContrastiveDataset(
                self.train_contrastive["Anchor"],
                self.train_contrastive["Positive"],
                self.train_contrastive["Negative"],
                self.drug_featurizer,
                self.target_featurizer,
            )

    def train_dataloader(self):
        return DataLoader(self.data_train, **self._loader_kwargs)

class CombinedDataModule(pl.LightningDataModule):
    """DataModule that combines one of [DTIDataModule, TDCDataModule, EnzPredDataModule] and the DUDeDataModule
    """
    def __init__(
            self,
            task: str,
            task_kwargs: dict,
            contrastive_kwargs: dict,
            ):
        super().__init__()

        self.task = task
        self.task_kwargs = task_kwargs
        self.contrastive_kwargs = contrastive_kwargs

        if self.task == 'dti_dg':
            self.task_module = TDCDataModule(**self.task_kwargs)
        elif self.task in EnzPredDataModule.dataset_list():
            self.task_module = EnzPredDataModule(**self.task_kwargs)
        else:
            self.task_module = DTIDataModule(**self.task_kwargs)

        self.contrastive_module = DUDEDataModule(**self.contrastive_kwargs)

    def prepare_data(self):
        self.task_module.prepare_data()
        self.contrastive_module.prepare_data()

    def setup(self, stage: T.Optional[str] = None):
        self.task_module.setup(stage)
        self.contrastive_module.setup(stage)
    
    def train_dataloader(self):
        if self.trainer.current_epoch % 2 == 0:
            return self.task_module.train_dataloader()
        else:
            return self.contrastive_module.train_dataloader()

    def val_dataloader(self):
        return self.task_module.val_dataloader()

    def test_dataloader(self):
        return self.task_module.test_dataloader()

class LeashDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            drug_featurizer: Featurizer,
            target_featurizer: Featurizer,
            device: torch.device = torch.device("cpu"),
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: int = 0,
            header=0,
            index_col=0,
            sep=",",
        ):
        super().__init__()

        self._loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": drug_target_collate_fn,
        }

        self._csv_kwargs = {
            "header": header,
            "index_col": index_col,
            "sep": sep,
        }

        self._device = device

        self._data_dir = Path(data_dir)
        self._train_path = Path("train_conplex.csv")
        # self._val_path = Path("val.csv")
        self._test_path = Path("test_conplex.csv")

        self._drug_column = "molecule_smiles"
        self._target_column = "protein_name"
        self._label_column = "Label"

        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer

    def prepare_data(self):
        """
        Featurize drugs and targets and save them to disk if they don't already exist
        """

        print(f"drug feat path: {self.drug_featurizer.path}\ntarget path:{self.target_featurizer.path}")
        if self.drug_featurizer.path.exists() and self.target_featurizer.path.exists():
            print("Drug and target featurizers already exist")
            return

        df_train = pd.read_csv(self._data_dir / self._train_path, **self._csv_kwargs)

        df_test = pd.read_csv(self._data_dir / self._test_path, **self._csv_kwargs)

        dataframes = [df_train, df_test]
        # remove "[Dy]" from the drug smiles
        for df in dataframes:
            df[self._drug_column] = df[self._drug_column].str.replace("\[Dy\]", "")
        all_drugs = pd.concat([i[self._drug_column] for i in dataframes]).unique()
        all_target_names = pd.concat([i[self._target_column] for i in dataframes]).unique()
        all_targets = []
        for targ in all_target_names:
            with open(self._data_dir / f"{targ}.fasta") as f:
                all_targets.append(f.read().strip())

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        if not self.drug_featurizer.path.exists():
            self.drug_featurizer.write_to_disk(all_drugs, file_path=self.drug_featurizer.path)

        if not self.target_featurizer.path.exists():
            self.target_featurizer.write_to_disk(all_targets, file_path=self.target_featurizer.path)

        self.drug_featurizer.cpu()
        self.target_featurizer.cpu()

    def setup(self, stage = None):
        self.df_train = pd.read_csv(self._data_dir / self._train_path, **self._csv_kwargs)
        self.df_val = pd.read_csv(self._data_dir / self._val_path, **self._csv_kwargs)
        self.df_test = pd.read_csv(self._data_dir / self._test_path, **self._csv_kwargs)

        self._dataframes = [self.df_train, self.df_val, self.df_test]
        # remove "[Dy]" from the drug smiles
        for df in self._dataframes:
            df[self._drug_column] = df[self._drug_column].str.replace("\[Dy\]", "")

        all_drugs = pd.concat([i[self._drug_column] for i in self._dataframes]).unique()
        all_target_names = pd.concat([i[self._target_column] for i in self._dataframes]).unique()
        all_targets = {}
        for targ in all_target_names:
            with open(self._data_dir / f"{targ}.fasta") as f:
                all_targets[targ] = f.read().strip()


        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        self.drug_featurizer.preload(all_drugs)
        self.drug_featurizer.cpu()

        self.target_featurizer.preload(list(all_targets.values()))
        self.target_featurizer.cpu()

        # using the all_targets dictionary, map the target names to the sequences in the dataframes
        for df in self._dataframes:
            df[self._target_column] = df[self._target_column].map(all_targets)

        if stage == "fit" or stage is None:
            self.data_train = BinaryDataset(
                self.df_train[self._drug_column],
                self.df_train[self._target_column],
                self.df_train[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

            self.data_val = BinaryDataset(
                self.df_val[self._drug_column],
                self.df_val[self._target_column],
                self.df_val[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

        if stage == "test" or stage is None:
            self.data_test = BinaryDataset(
                self.df_test[self._drug_column],
                self.df_test[self._target_column],
                self.df_test[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

class MergedDataset(Dataset):
    def __init__(self, split, drug_db, target_db, id_to_smiles, id_to_target, tdim=1280, exclusion_file=None, neg_sample_ratio=3):
        """
        Constructor for the merged dataset, pooling DTI data from PubChem, BindingDB, and ChEMBL.

        `split`: one of 'train', 'test', or 'val'
        `drug_db`: a reference to the LMDB database of ligands that supports this dataset.
        `target_db`: a reference to the LMDB database of targets that supports this dataset.
        """
        self.split = split
        self.neg_sample_ratio = neg_sample_ratio

        # Ligand ID to smiles mapping
        self.id_to_smiles = id_to_smiles
        # Target ID to smiles mapping
        self.id_to_target = id_to_target

        # Load the sequence model's embedding dimension
        self.tdim = tdim

        id_list = []
        for k in list(self.id_to_target.keys()):
            if any(char.isdigit() for char in self.id_to_target[k]):
                continue
            id_list.append(k)

        self.id_to_target = {k: self.id_to_target[k] for k in id_list}
        self.id_list = id_list

        # connect the db id to the lmdb index
        self.id_to_drug_lmdb = {db_id: lmdb_id for lmdb_id, db_id in enumerate(self.id_to_smiles.keys())}
        self.id_to_prot_lmdb = {db_id: lmdb_id for lmdb_id, db_id in enumerate(id_list)}

        # Exclude some ID's for homology based analysis
        self.exclusion = set()
        if exclusion_file is not None:
            for line in open(exclusion_file.strip()):
                self.exclusion.add(line.strip())
            print(f"[{split}] Loaded {len(self.exclusion)} proteins to exclude from exclusion file")

        # Load positive and negative interactions
        if split == 'all':
            print('Training on all of train / val / test data to ship model.')
            # Combine all data for final model
            pos_data_train = pd.read_csv('data/MERGED/huge_data/merged_pos_uniq_train_rand.tsv', sep='\t')
            pos_data_val = pd.read_csv('data/MERGED/huge_data/merged_pos_uniq_val_rand.tsv', sep='\t')
            pos_data_test = pd.read_csv('data/MERGED/huge_data/merged_pos_uniq_test_rand.tsv', sep='\t')
            self.pos_data = pd.concat([pos_data_train, pos_data_val, pos_data_test])

            neg_data_train = pd.read_csv('data/MERGED/huge_data/merged_neg_uniq_train_rand.tsv', sep='\t')
            neg_data_val = pd.read_csv('data/MERGED/huge_data/merged_neg_uniq_val_rand.tsv', sep='\t')
            neg_data_test = pd.read_csv('data/MERGED/huge_data/merged_neg_uniq_test_rand.tsv', sep='\t')
            self.neg_data = pd.concat([neg_data_train, neg_data_val, neg_data_test])
        else:
            self.pos_data = pd.read_csv(f'data/MERGED/huge_data/merged_pos_uniq_{split}_rand.tsv', sep='\t')
            self.neg_data = pd.read_csv(f'data/MERGED/huge_data/merged_neg_uniq_{split}_rand.tsv', sep='\t')

        # Count how many data points contain excluded proteins
        if self.exclusion:
            pos_excluded = self.pos_data['aa_seq'].isin(self.exclusion).sum()
            neg_excluded = self.neg_data['aa_seq'].isin(self.exclusion).sum()
            total_pos = len(self.pos_data)
            total_neg = len(self.neg_data)

            print(f"\n[{split}] Data exclusion statistics:")
            print(f"  Positive interactions: {pos_excluded}/{total_pos} ({pos_excluded/total_pos*100:.2f}%) contain excluded proteins")
            print(f"  Negative interactions: {neg_excluded}/{total_neg} ({neg_excluded/total_neg*100:.2f}%) contain excluded proteins")
            print(f"  Total interactions affected: {pos_excluded + neg_excluded}/{total_pos + total_neg} ({(pos_excluded + neg_excluded)/(total_pos + total_neg)*100:.2f}%)\n")
        
        # Receive drug and target db's from the datamodule. we assume that concurrent reads are ok
        self.drug_db = drug_db
        self.target_db = target_db

        # Sample negative data for the epoch
        self.update_epoch_data()

    def update_epoch_data(self):
        """
        Samples a random number of negative data, equal to the amount of positive data we have.
        These will be used for the current epoch.
        """
        neg_sample_size = min(len(self.pos_data) * self.neg_sample_ratio, len(self.neg_data))
        self.epoch_neg_data = self.neg_data.sample(n=neg_sample_size, replace=False)

    def __len__(self):
        """
        Returns the total amount of data that we make visible during this epoch.
        This would be all the positive data, and some random subsample of the negative data.
        Therefore this always totals to 2 * len(self.pos_data).
        """
        return len(self.pos_data) + len(self.epoch_neg_data)

    def __getitem__(self, idx):
        """
        Get an item from this dataset by idx.
        If the idx is less than the size of the positive data, we return the positive example. 
        Otherwise return a negative example.
        """
        if idx < len(self.pos_data):
            interaction = self.pos_data.iloc[idx]
            label = 1.0
        else:
            interaction = self.epoch_neg_data.iloc[idx - len(self.pos_data)]
            label = 0.0

        drug_id, aa_id = interaction['ligand'], interaction['aa_seq']

        # aa_id is the uniprot id, simply check and see if this is blacklisted under mmseq threshold.
        drug_id = self.id_to_drug_lmdb[drug_id]
        drug_features = self.drug_db[drug_id]['feats']

        # if the uniprot id is to be excluded for homology analysis
        if self.split == 'all' and aa_id in self.exclusion:
            drug_features = np.zeros(drug_features.shape, dtype=np.float32)
            # if this is not ProtBert...
            target_features = np.zeros((1, self.tdim), dtype=np.float32)
        else:
            if aa_id not in self.id_to_prot_lmdb: # if the uniprot id is not in the map
                target_features = np.zeros((1, self.tdim), dtype=np.float32)
            else:
                target_entry = self.target_db[self.id_to_prot_lmdb[aa_id]]

                if aa_id in target_entry:
                    target_features = target_entry[aa_id]
                else:
                    target_features = np.zeros((1, self.tdim), dtype=np.float32)

        # Fetch the drug and target feature for this idx from LMDB
        return (
            torch.from_numpy(drug_features), # drug
            torch.from_numpy(target_features), # target
            torch.tensor(label, dtype=torch.float32) # label
        )

    def on_epoch_end(self):
        self.update_epoch_data()

class MergedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        drug_featurizer: Featurizer,
        target_featurizer: Featurizer,
        device: torch.device("cuda"),
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 17,
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_state: int = 42,
        header=0,
        index_col=0,
        sep=",",
        ship_model: bool = False,
        similarity_threshold: float = 0.9,
        target_protein_id: str = None,
    ):
        super().__init__()
        self._loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": drug_target_collate_fn,
        }

        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer
        self.ship_model = ship_model
        self.similarity_threshold = similarity_threshold
        self.target_protein_id = target_protein_id

        # Load in the ID to SMILES and ID to target sequence files
        self.id_to_smiles = np.load('data/MERGED/huge_data/id_to_smiles.npy', allow_pickle=True).item()
        if self.target_featurizer.name == "SaProt":
            self.id_to_target = np.load('data/MERGED/huge_data/id_to_saprot_sequence.npy', allow_pickle=True).item()
        else:
            self.id_to_target = np.load('data/MERGED/huge_data/id_to_sequence.npy', allow_pickle=True).item()

        id_list = []
        for k in list(self.id_to_target.keys()):
            if any(char.isdigit() for char in self.id_to_target[k]):
                continue
            id_list.append(k)

        self.id_to_target = {k: self.id_to_target[k] for k in id_list}
        self.id_list = id_list

        # connect the db id to the lmdb index
        self.id_to_drug_lmdb = {db_id: lmdb_id for lmdb_id, db_id in enumerate(self.id_to_smiles.keys())}
        self.id_to_prot_lmdb = {db_id: lmdb_id for lmdb_id, db_id in enumerate(id_list)}

        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        # these get initialized in setup
        self.drug_db = None
        self.target_db = None

    def setup(self, stage: Optional[str] = None):
        # Process drug and target databases if not already processed
        # this stores featurizations for the given ligand ids and target ids into LMDB files
        smiles_lmdb = 'data/MERGED/huge_data/smiles.lmdb'
        target_lmdb = f'data/MERGED/huge_data/{self.target_featurizer.name}_targets.lmdb'

        self.drug_featurizer.process_merged_drugs(self.id_to_smiles)
        self.target_featurizer.process_merged_targets(self.id_to_target)

        self.drug_db = px.Reader(dirpath=smiles_lmdb, lock=False) # we only read
        self.target_db = px.Reader(dirpath=target_lmdb, lock=False)

        tdim = self.target_featurizer.shape

        if self.ship_model: # Combine all data for final model, while excluding similar proteins
            import json
            import tempfile

            # Load LIT-PCBA sequences
            if self.target_featurizer.name == "SaProt":
                pcba_seq_file = 'data/lit_pcba/saprot_sequence_dict.json'
            else:
                pcba_seq_file = 'data/lit_pcba/lit_pcba_sequence_dict.json'

            pcba_sequences = json.load(open(pcba_seq_file))

            # Determine which target sequences to use
            if self.target_protein_id is None or self.target_protein_id.lower() == 'all':
                # Combine all LIT-PCBA sequences
                print(f"Computing sequence similarity for ALL LIT-PCBA proteins at threshold {self.similarity_threshold}")
                all_target_seqs = []
                for protein_id, seqs in pcba_sequences.items():
                    if isinstance(seqs, list):
                        all_target_seqs.extend(seqs)
                    else:
                        all_target_seqs.append(seqs)
                target_seqs = all_target_seqs
            else:
                # Single protein
                if self.target_protein_id not in pcba_sequences:
                    raise ValueError(f"Target protein {self.target_protein_id} not found in LIT-PCBA dataset")

                print(f"Computing sequence similarity for {self.target_protein_id} at threshold {self.similarity_threshold}")
                target_seqs = pcba_sequences[self.target_protein_id]
                if not isinstance(target_seqs, list):
                    target_seqs = [target_seqs]

            # Compute similar sequences on-the-fly
            total_train_sequences = len(self.id_to_target)
            print(f"\n{'='*60}")
            print(f"MMseqs2 Dataset Filtering Statistics")
            print(f"{'='*60}")
            print(f"Total training sequences: {total_train_sequences}")

            similar_ids = compute_similar_sequences_single_target(
                target_seqs,
                self.id_to_target,
                threshold=self.similarity_threshold
            )

            num_excluded = len(similar_ids)
            exclusion_percentage = (num_excluded / total_train_sequences) * 100 if total_train_sequences > 0 else 0

            print(f"Similar sequences found (>={self.similarity_threshold} threshold): {num_excluded}")
            print(f"Sequences to exclude: {num_excluded} ({exclusion_percentage:.2f}%)")
            print(f"Remaining sequences: {total_train_sequences - num_excluded} ({100 - exclusion_percentage:.2f}%)")
            print(f"{'='*60}\n")

            # Write to temporary file
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
            for uniprot_id in similar_ids:
                temp_file.write(f"{uniprot_id}\n")
            temp_file.close()
            exclusion_file = temp_file.name

            self.data_all = MergedDataset('all', self.drug_db, self.target_db, self.id_to_smiles, self.id_to_target, tdim=tdim, exclusion_file=exclusion_file)
            self.data_test = MergedDataset('test', self.drug_db, self.target_db, self.id_to_smiles, self.id_to_target, tdim=tdim, exclusion_file=exclusion_file)
        else:
            # Regular setup for train/val/test (no exclusions)
            if stage == "fit" or stage is None:
                self.data_train = MergedDataset('train', self.drug_db, self.target_db, self.id_to_smiles, self.id_to_target, tdim=tdim)
                self.data_val = MergedDataset('val', self.drug_db, self.target_db, self.id_to_smiles, self.id_to_target, tdim=tdim)
            if stage == "test" or stage is None:
                self.data_test = MergedDataset('test', self.drug_db, self.target_db, self.id_to_smiles, self.id_to_target, tdim=tdim)

    def train_dataloader(self):
        if self.ship_model:
            return DataLoader(self.data_all, **self._loader_kwargs, pin_memory=True)
        return DataLoader(self.data_train, **self._loader_kwargs, pin_memory=True)
    
    def val_dataloader(self):
        if self.ship_model:
            return DataLoader(self.data_test, **self._loader_kwargs, pin_memory=True)
        return DataLoader(self.data_val, **self._loader_kwargs, pin_memory=True)

    def test_dataloader(self):
        if self.ship_model:
            return []
        return DataLoader(self.data_test, **self._loader_kwargs, pin_memory=True)

    def predict_dataloader(self):
        if self.ship_model:
            return DataLoader(self.data_all, **self._loader_kwargs, pin_memory=True)
        return None

    def teardown(self, stage: str):
        self.drug_db.close()
        self.target_db.close()