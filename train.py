import numpy as np
import pandas as pd

import torch
from torch import nn
import pytorch_lightning as pl

import wandb
from omegaconf import OmegaConf
from pathlib import Path

import argparse

from datamodules import (
        get_task_dir,
        DTIDataModule,
        TDCDataModule,
        DUDEDataModule,
        EnzPredDataModule,
        )
from model import DrugTargetCoembeddingLightning
from trainloop import ConPlexEpochLoop
from utils import get_featurizer

parser = argparse.ArgumentParser(description="PLM_DTI Training.")
parser.add_argument("--exp-id", required=True, help="Experiment ID", dest="experiment_id")
parser.add_argument("--config", help="YAML config file", default="configs/default_config.yaml")
parser.add_argument("--wandb-proj", help="Weights and Biases Project",dest="wandb_proj")
parser.add_argument("--task", choices=[
    "biosnap",
    "bindingdb",
    "davis",
    "biosnap_prot",
    "biosnap_mol",
    "dti_dg",
    ], type=str, help="Task name. Could be biosnap, bindingdb, davis, biosnap_prot, biosnap_mol, dti_dg.",
)
parser.add_argument("--drug-featurizer", help="Drug featurizer", dest="drug_featurizer")
parser.add_argument("--target-featurizer", help="Target featurizer", dest="target_featurizer")
parser.add_argument("--distance-metric", help="Distance in embedding space to supervise with", dest="distance_metric")
parser.add_argument("--epochs", type=int, help="number of total epochs to run")
parser.add_argument("--lr", "--learning-rate", type=float, help="initial learning rate", dest="lr",)
parser.add_argument("--clr", type=float, help="initial learning rate", dest="clr")
parser.add_argument("--r", "--replicate", type=int, help="Replicate", dest="replicate")
parser.add_argument("--d", "--device", type=int, help="CUDA device", dest="device")
parser.add_argument("--verbosity", type=int, help="Level at which to log", dest="verbosity")
parser.add_argument("--checkpoint", default=None, help="Model weights to start from")

parser.add_argument("--num-layers-target", type=int, help="Number of layers in target transformer", dest="num_layers_target")
parser.add_argument("--dropout", type=float, help="Dropout rate for transformer", dest="dropout")

args = parser.parse_args()
config = OmegaConf.load(args.config)
args_overrides = {k: v for k, v in vars(args).items() if v is not None}
config.update(args_overrides)

save_dif = f'{config.get("model_save_dir", ".")}/{config.experiment_id}'
os.makedirs(save_dir, exist_ok=True)

# Set CUDA device
device_no = config.device
use_cuda = torch.cuda.is_available()
device = torch.device(f"cuda:{device_no}" if use_cuda else "cpu")
print(f"Using CUDA device {device}")

# Set random state
print(f"Setting random state {config.replicate}")
torch.manual_seed(config.replicate)
np.random.seed(config.replicate)

# Load data
print("Preparing DataModule")
task_dir = get_task_dir(config.task)

drug_featurizer = get_featurizer(config.drug_featurizer, save_dir=task_dir)
target_featurizer = get_featurizer(config.target_featurizer, save_dir=task_dir)

if config.task == 'dti_dg':
    config.classify = False
    config.watch_metric = "val/pcc"
    datamodule = TDCDataModule(
            task_dir,
            drug_featurizer,
            target_featurizer,
            device=device,
            seed=config.replicate,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
        )
elif config.task in EnzPredDataModule.dataset_list():
    # Not implemented yet
    RuntimeError("EnzPredDataModule not implemented yet")
else:
    config.classify = True
    config.watch_metric = "val/aupr"
    datamodule = DTIDataModule(
            task_dir,
            drug_featurizer,
            target_featurizer,
            device=device,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
        )
datamodule.prepare_data()
datamodule.setup()

if config.contrastive:
    print("Loading contrastive data (DUDE)")
    dude_drug_featurizer = get_featurizer(config.drug_featurizer, save_dir=get_task_dir("DUDe"))
    dude_target_featurizer = get_featurizer(config.target_featurizer, save_dir=get_task_dir("DUDe"))

    contrastive_datamodule = DUDEDataModule(
            config.contrastive_split,
            dude_drug_featurizer,
            dude_target_featurizer,
            device=device,
            batch_size=config.contrastive_batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
            )
    contrastive_datamodule.prepare_data()
    contrastive_datamodule.setup(stage="fit")

# Load model
print("Initializing model")
model = DrugTargetCoembeddingLightning(
        drug_dim=config.drug_shape,
        target_dim=config.target_shape,
        latent_dim=config.latent_dim,
        classify=config.classify,
        contrastive=config.contrastive,
        num_layers_target=config.num_layers_target,
        dropout=config.dropout,
        )

# Train model
trainer = pl.Trainer()
trainer.training_epoch_loop = ConPlexEpochLoop(contrastive=config.contrastive)
trainer.fit(
        model,
        train_dataloaders=[datamodule.train_dataloader(), contrastive_datamodule.train_dataloader()], #this won't work, need to change
        val_dataloaders=datamodule.val_dataloader(),
        )


