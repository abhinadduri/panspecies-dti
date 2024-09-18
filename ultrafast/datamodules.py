import torch
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import pyxis as px
import typing as T


from numpy.random import choice
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
from tdc.benchmark_group import dti_dg_group
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from ultrafast.featurizers import Featurizer

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
        "test_data": "./data/test_data",
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
        "leash": "./data/leash/",
    }

    return Path(task_paths[task_name.lower()]).resolve()

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        seq = self.featurizer.prepare_string(self.data[self._column].iloc[i])
        mol = self.featurizer.features[seq]

        return mol

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

class DTIStructDataModule(DTIDataModule):
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
        super().__init__(
            data_dir,
            drug_featurizer,
            target_featurizer,
            device,
            batch_size,
            shuffle,
            num_workers,
            header,
            index_col,
            sep,
        )
        self._data_dir = Path(data_dir)
        self._train_path = Path("train_foldseek.csv")
        self._val_path = Path("val_foldseek.csv")
        self._test_path = Path("test_foldseek.csv")

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

    def prepare_data(self):

        dg_group = dti_dg_group(path=self._data_dir)
        dg_benchmark = dg_group.get("bindingdb_patent")

        train_val, test = (
                dg_benchmark["train_val"],
                dg_benchmark["test"],
                )

        all_drugs = pd.concat([train_val, test])[self._drug_column].unique()
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

        dg_group = dti_dg_group(path=self._data_dir)
        dg_benchmark = dg_group.get("bindingdb_patent")
        dg_name = dg_benchmark["name"]

        self.df_train, self.df_val = dg_group.get_train_valid_split(
                benchmark=dg_name, 
                split_type="default", 
                seed=self._seed
                )
        self.df_test = dg_benchmark["test"]

        self._dataframes = [self.df_train, self.df_val]

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
    def__init__(self, split, drug_db, target_db, neg_sample_ratio=1.0):
        """
        Constructor for the merged dataset, pooling DTI data from PubChem, BindingDB, and ChEMBL.

        `split`: one of 'train', 'test', or 'val'
        `drug_db`: a reference to the LMDB database of ligands that supports this dataset.
        `target_db`: a reference to the LMDB database of targets that supports this dataset.
        """
        self.split = split
        self.neg_sample_ratio = neg_sample_ratio

        # Load ligand ID to smiles mapping
        self.id_to_smiles = np.load('data/MERGED/huge_data/id_to_smiles.npy', allow_pickle=True).item()
        # Load ligand ID to smiles mapping
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

        # Load positive and negative interactions
        self.pos_data = pd.read_csv(f'data/MERGED/huge_data/merged_pos_uniq_{split}_rand.tsv', sep='\t')
        self.neg_data = pd.read_csv(f'data/MERGED/huge_data/merged_neg_uniq_{split}_rand.tsv', sep='\t')

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
        drug_id = self.id_to_drug_lmdb[drug_id]

        if aa_id not in self.id_to_prot_lmdb:
            target_features = np.zeros((1280))
        else:
            target_entry = self.target_db[self.id_to_prot_lmdb[aa_id]]

            if aa_id in target_entry:
                target_features = target_entry[aa_id]
            else:
                target_features = np.zeros((1280))

        # Fetch the drug and target feature for this idx from LMDB
        return (
            torch.from_numpy(self.drug_db[drug_id]['feats']), # drug
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
        device: torch.device = torch.device("cpu"),
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 16,
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_state: int = 42,
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

    # Load in the ID to SMILES and ID to target sequence files
    self.id_to_smiles = np.load('data/MERGED/huge_data/id_to_smiles.npy', allow_pickle=True).item()
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

    self.drug_featurizer = drug_featurizer
    self.target_featurizer = target_featurizer

    self.test_size = test_size
    self.val_size = val_size
    self.random_state = random_state

    self.drug_db = px.Reader(dirpath='data/MERGED/huge_data/smiles.lmdb', lock=False) # we only read
    self.target_db = px.Reader(dirpath='data/MERGED/huge_data/targets.lmdb', lock=False)

    def setup(self, stage: Optional[str] = None):
        # Process drug and target databases if not already processed
        # this stores featurizations for the given ligand ids and target ids into LMDB files
        self.drug_featurizer.process_merged_drugs(self.id_to_smiles)
        self.target_featurizer.process_merged_targets(self.id_to_target)

        if stage == "fit" or stage is None:
            self.data_train = MergedDataset('train', self.drug_db, self.target_db)
            self.data_val = MergedDataset('val', self.drug_db, self.target_db)
        if stage == "test" or stage is None:
            self.data_test = MergedDataset('test', self.drug_db, self.target_db)

    def train_dataloader(self):
        return DataLoader(self.data_train, **self._loader_kwargs, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.data_val, **self._loader_kwargs, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, **self._loader_kwargs, pin_memory=True)

    def teardown(self, stage: str):
        self.drug_db.close()
        self.target_db.close()











