import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

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
        label = torch.tensor(self.labels.iloc[i])

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

class DTIDataModule(pl.LightningDataModule):
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

        if self.drug_featurizer.path.exists() and self.target_featurizer.path.exists():
            logg.warning("Drug and target featurizers already exist")
            return

        df_train = pd.read_csv(self._data_dir / self._train_path, **self._csv_kwargs)

        df_val = pd.read_csv(self._data_dir / self._val_path, **self._csv_kwargs)

        df_test = pd.read_csv(self._data_dir / self._test_path, **self._csv_kwargs)

        dataframes = [df_train, df_val, df_test]
        all_drugs = pd.concat([i[self._drug_column] for i in dataframes]).unique()
        all_targets = pd.concat([i[self._target_column] for i in dataframes]).unique()

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        if not self.drug_featurizer.path.exists():
            self.drug_featurizer.write_to_disk(all_drugs)

        if not self.target_featurizer.path.exists():
            self.target_featurizer.write_to_disk(all_targets)

        self.drug_featurizer.cpu()
        self.target_featurizer.cpu()

    def setup(self, stage = None):
        self.df_train = pd.read_csv(self._data_dir / self._train_path, **self._csv_kwargs)
        self.df_val = pd.read_csv(self._data_dir / self._val_path, **self._csv_kwargs)
        self.df_test = pd.read_csv(self._data_dir / self._test_path, **self._csv_kwargs)

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

class TDCDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def prepare_data(self):

    def setup(self):

    def train_dataloader(self):

    def val_dataloader(self):

    def test_dataloader(self):

class DTEnzPredDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

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

    def setup(self):

    def train_dataloader(self):

    def val_dataloader(self):

    def test_dataloader(self):

class DUDEDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):

    def setup(self):

    def train_dataloader(self):

    def val_dataloader(self):

    def test_dataloader(self):


