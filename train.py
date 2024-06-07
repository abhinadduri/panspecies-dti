import numpy as np
import pandas as pd

import torch
from torch import nn
import pytorch_lightning as pl

import argparse

from datamodules import (
        DTIDataModule,
        TDCDataModule,
        DUDEDataModule,
        DTEnzPredDataModule,
        # other data modules here
)
from model import DrugTargetCoembeddingLightning

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True)
# Add other arguments here
args = parser.parse_args()

# Load data
task_datamodule = DTIDataModule(
        # Add data module arguments here
        ) # change depending on task
task_datamodule.setup()

contrastive_datamodule = DUDEDataModule(
        # Add data module arguments here
        )
contrastive_datamodule.setup()

# Load model
model = DrugTargetCoembeddingLightning(
        # Add model arguments here
        )

# Train model
trainer = pl.Trainer()
trainer.fit(model, contrastive_datamodule) # also need to pass in task_datamodule


