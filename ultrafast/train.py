import os
import numpy as np
import pandas as pd

import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger

import wandb
from omegaconf import OmegaConf
from pathlib import Path

import argparse

from ultrafast.callbacks import eval_pcba
from ultrafast.datamodules import (
    get_task_dir,
    DTIDataModule,
    TDCDataModule,
    DUDEDataModule,
    EnzPredDataModule,
    CombinedDataModule,
    MergedDataModule
)
from ultrafast.model import DrugTargetCoembeddingLightning
from ultrafast.drug_only_model import DrugOnlyLightning
from ultrafast.utils import get_featurizer, xavier_normal

class PCBAEvaluationCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        eval_pcba(trainer, pl_module)

def train_cli():
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
        "merged",
        "custom",
        ], type=str, help="Task name. Could be biosnap, bindingdb, davis, biosnap_prot, biosnap_mol, dti_dg.",
    )
    parser.add_argument("--drug-featurizer", help="Drug featurizer", dest="drug_featurizer")
    parser.add_argument("--target-featurizer", help="Target featurizer", dest="target_featurizer")
    parser.add_argument('--ligand-only', action="store_true", help="Only use ligand features")
    parser.add_argument("--epochs", type=int, help="number of total epochs to run")
    parser.add_argument("--lr", "--learning-rate", type=float, help="initial learning rate", dest="lr",)
    parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for optimizer")
    parser.add_argument("--clr", type=float, help="contrastive initial learning rate", dest="clr")
    parser.add_argument("--CEWeight", "-C", default=1.0, type=float, help="Cross Entropy loss weight", dest="CEWeight")
    parser.add_argument("--InfoNCEWeight","-I", default=0.0, type=float, help="InfoNCE loss weight", dest="InfoNCEWeight")
    parser.add_argument("--InfoNCETemp", "-T", default=1.0, type=float, help="InfoNCE temperature", dest="InfoNCETemp")
    parser.add_argument("--r", "--replicate", type=int, help="Replicate", dest="replicate")
    parser.add_argument("--d", "--device", default=0, type=int, help="CUDA device", dest="device")
    parser.add_argument("--checkpoint", default=None, help="Model weights to start from")
    parser.add_argument('--prot-proj', choices=["avg","agg","transformer"], help="Change the protein projector method")
    parser.add_argument('--out-type', choices=['cls','mean'], help="use cls token or mean of everything else")

    parser.add_argument("--num-layers-target", type=int, help="Number of layers in target transformer", dest="num_layers_target")
    parser.add_argument("--num-heads-agg", type=int, default=4, help="Number of attention heads for learned aggregation", dest="num_heads_agg")
    parser.add_argument("--dropout", type=float, help="Dropout rate for transformer", dest="dropout")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size for training/val/test")
    parser.add_argument("--num-workers", type=int, default=0, help="number of workers for intial data processing and dataloading during training")
    parser.add_argument("--no-wandb", action="store_true", help="Do not use wandb")
    parser.add_argument("--model-size", default="small", choices=["small", "large"], help="Choose the size of the model")
    parser.add_argument("--ship-model", help="Train a final to ship model, while excluding the uniprot id's specified by this argument.", dest="ship_model")
    parser.add_argument("--eval-pcba", action="store_true", help="Evaluate PCBA during validation")
    parser.add_argument("--sigmoid-scalar", type=int, default=5, dest="sigmoid_scalar")

    args = parser.parse_args()
    train(**vars(args))

def train(
    experiment_id: str,
    config: str,
    wandb_proj: str,
    task: str,
    drug_featurizer: str,
    target_featurizer: str,
    ligand_only: bool,
    epochs: int,
    lr: float,
    weight_decay: float,
    clr: float,
    CEWeight: float,
    InfoNCEWeight: float,
    InfoNCETemp: float,
    replicate: int,
    device: int,
    checkpoint: str,
    prot_proj: str,
    out_type: str,
    num_layers_target: int,
    dropout: float,
    batch_size: int,
    num_workers: int,
    no_wandb: bool,
    num_heads_agg: int,
    model_size: str,
    ship_model: str,
    eval_pcba: bool,
    sigmoid_scalar: int,
):
    args = argparse.Namespace(
        experiment_id=experiment_id,
        config=config,
        wandb_proj=wandb_proj,
        task=task,
        drug_featurizer=drug_featurizer,
        target_featurizer=target_featurizer,
        ligand_only=ligand_only,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        clr=clr,
        CEWeight=CEWeight,
        InfoNCEWeight=InfoNCEWeight,
        InfoNCETemp=InfoNCETemp,
        replicate=replicate,
        device=device,
        checkpoint=checkpoint,
        prot_proj=prot_proj,
        out_type=out_type,
        num_layers_target=num_layers_target,
        dropout=dropout,
        batch_size=batch_size,
        num_workers=num_workers,
        no_wandb=no_wandb,
        num_heads_agg=num_heads_agg,
        model_size=model_size,
        ship_model=ship_model,
        eval_pcba=eval_pcba,
        sigmoid_scalar=sigmoid_scalar,
    )
    config = OmegaConf.load(args.config)
    args_overrides = {k: v for k, v in vars(args).items() if v is not None}
    config.update(args_overrides)

    save_dir = f'{config.get("model_save_dir", ".")}/{config.experiment_id}'

    # Set CUDA device
    device_no = config.device
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{device_no}" if use_cuda else "cpu")
    print(f"Using CUDA device {device}")
    torch.set_float32_matmul_precision('medium')

    # Set random state
    print(f"Setting random state {config.replicate}")
    torch.manual_seed(config.replicate)
    np.random.seed(config.replicate)

    # Load data
    print("Preparing DataModule")
    task_dir = get_task_dir(config.task)

    drug_featurizer = get_featurizer(config.drug_featurizer, save_dir=task_dir, n_jobs=config.num_workers)

    target_featurizer = get_featurizer(config.target_featurizer, save_dir=task_dir)

    # Set up task dm arguments
    if config.task == 'dti_dg':
        config.classify = False
        config.watch_metric = "val/mse"
        task_dm_kwargs = {
                "data_dir": task_dir,
                "drug_featurizer": drug_featurizer,
                "target_featurizer": target_featurizer,
                "device": device,
                "seed": config.replicate,
                "batch_size": config.batch_size,
                "shuffle": config.shuffle,
                "num_workers": config.num_workers,
                }
    elif config.task in EnzPredDataModule.dataset_list():
        # Not implemented yet
        RuntimeError("EnzPredDataModule not implemented yet")
    else:
        config.classify = True
        config.watch_metric = "val/aupr"
        task_dm_kwargs = {
                "data_dir": task_dir,
                "drug_featurizer": drug_featurizer,
                "target_featurizer": target_featurizer,
                "device": device,
                "batch_size": config.batch_size,
                "shuffle": config.shuffle,
                "num_workers": config.num_workers,
                }

    if config.contrastive:
        print("Loading contrastive data (DUDE)")
        dude_drug_featurizer = get_featurizer(config.drug_featurizer, save_dir=get_task_dir("DUDe"), ext='pt')
        dude_target_featurizer = get_featurizer(config.target_featurizer, save_dir=get_task_dir("DUDe"))

        contrastive_dm_kwargs = {
                "contrastive_split": config.contrastive_split,
                "drug_featurizer": dude_drug_featurizer,
                "target_featurizer": dude_target_featurizer,
                "device": device,
                "batch_size": config.contrastive_batch_size,
                "shuffle": config.shuffle,
                "num_workers": config.num_workers,
                "contrastive_type": config.contrastive_type,
                }

        datamodule = CombinedDataModule(
                task=config.task,
                task_kwargs=task_dm_kwargs,
                contrastive_kwargs=contrastive_dm_kwargs,
                )
    else:
        if config.task == 'dti_dg':
            datamodule = TDCDataModule(**task_dm_kwargs)
        elif config.task in EnzPredDataModule.dataset_list():
            RuntimeError("EnzPredDataModule not implemented yet")
        else:
            datamodule = DTIDataModule(**task_dm_kwargs)

    if config.task != 'merged':
        datamodule.prepare_data() # this task is already setup
    else:
        datamodule = MergedDataModule(**task_dm_kwargs, ship_model=ship_model)
    datamodule.setup()

    # Load model
    if not args.ligand_only:
        if args.checkpoint:
            print(f"Loading model from checkpoint: {args.checkpoint}")
            model = DrugTargetCoembeddingLightning.load_from_checkpoint(
                args.checkpoint,
                drug_dim=drug_featurizer.shape,
                target_dim=target_featurizer.shape,
                latent_dim=config.latent_dimension,
                classify=config.classify,
                contrastive=config.contrastive,
                InfoNCEWeight=config.InfoNCEWeight,
                prot_proj=config.prot_proj,
                dropout=config.dropout,
                device=device,
                args=config
            )
        else:
            print("Initializing new model")
            model = DrugTargetCoembeddingLightning(
                drug_dim=drug_featurizer.shape,
                target_dim=target_featurizer.shape,
                latent_dim=config.latent_dimension,
                classify=config.classify,
                contrastive=config.contrastive,
                InfoNCEWeight=config.InfoNCEWeight,
                prot_proj=config.prot_proj,
                dropout=config.dropout,
                args=config
            )
    else:
        print(f"Loading model from checkpoint: {args.checkpoint}")
        if args.checkpoint:
            model = DrugOnlyLightning.load_from_checkpoint(
                args.checkpoint,
                drug_dim=drug_featurizer.shape,
                latent_dim=config.latent_dimension,
                classify=config.classify,
                contrastive=config.contrastive,
                InfoNCEWeight=config.InfoNCEWeight,
                dropout=config.dropout,
                device=device,
                args=config
            )
        else:
            print("Initializing new model")
            model = DrugOnlyLightning(
                drug_dim=drug_featurizer.shape,
                latent_dim=config.latent_dimension,
                classify=config.classify,
                contrastive=config.contrastive,
                InfoNCEWeight=config.InfoNCEWeight,
                dropout=config.dropout,
                args=config
            )

    if args.checkpoint:
        print(f"Loading model from checkpoint: {args.checkpoint}")
        model = DrugTargetCoembeddingLightning.load_from_checkpoint(
            args.checkpoint,
            drug_dim=drug_featurizer.shape,
            target_dim=target_featurizer.shape,
            latent_dim=config.latent_dimension,
            classify=config.classify,
            contrastive=config.contrastive,
            InfoNCEWeight=config.InfoNCEWeight,
            prot_proj=config.prot_proj,
            dropout=config.dropout,
            device=device,
            args=config
        )
    else:
        print("Initializing new model")
        model = DrugTargetCoembeddingLightning(
            drug_dim=drug_featurizer.shape,
            target_dim=target_featurizer.shape,
            latent_dim=config.latent_dimension,
            classify=config.classify,
            contrastive=config.contrastive,
            InfoNCEWeight=config.InfoNCEWeight,
            prot_proj = config.prot_proj,
            dropout=config.dropout,
            args=config
        )

    if not config.no_wandb:
        wandb_logger = WandbLogger(project=config.wandb_proj, log_model='all')
        wandb_logger.watch(model)
        if hasattr(wandb_logger.experiment.config, 'update'):
            wandb_logger.experiment.config.update(OmegaConf.to_container(config, resolve=True, throw_on_missing=True))
        wandb_logger.experiment.tags = [config.task, config.experiment_id, config.target_featurizer, config.model_size]

    if config.task == 'merged' and args.ship_model:
        # save every epoch
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=-1,
            dirpath=save_dir,
            verbose=True
        )
    else:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=config.watch_metric,
            mode="max" if "mse" not in config.watch_metric else "min",
            filename=config.task,
            dirpath=save_dir,
            verbose=True
        )

    callbacks = [checkpoint_callback]
    if args.eval_pcba:
        callbacks.append(PCBAEvaluationCallback())

    callbacks = [checkpoint_callback]
    if args.eval_pcba:
        callbacks.append(PCBAEvaluationCallback())

    # Train model
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        strategy="auto",
        logger=wandb_logger if not config.no_wandb else None,
        max_epochs=config.epochs,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1 if config.contrastive else 0,
        # Disable testing for final model mode
        limit_test_batches=0 if ship_model else 1.0,
    )

    if ship_model:
        # Train on all data and test with best weights
        trainer.fit(model, datamodule=datamodule)
        trainer.test(datamodule=datamodule, ckpt_path=checkpoint_callback.best_model_path)
        # Save the final model
        trainer.save_checkpoint(f"{save_dir}/ship_model.ckpt")
    else:
        # Regular training with validation
        trainer.fit(model, datamodule=datamodule)
        # Test model using best weights
        if config.epochs == 0:
            ckpt = config.checkpoint
        else:
            ckpt = checkpoint_callback.best_model_path
        trainer.test(datamodule=datamodule, ckpt_path=ckpt)


if __name__ == '__main__':
    train()
