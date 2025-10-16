from __future__ import annotations

import os
import numpy as np
import pandas as pd  # noqa: F401

import torch
from torch import nn  # noqa: F401
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger

import wandb
from omegaconf import OmegaConf
from pathlib import Path  # noqa: F401
import argparse

from ultrafast.callbacks import eval_pcba
from ultrafast.datamodules import (
    get_task_dir,
    DTIDataModule,
    TDCDataModule,
    DUDEDataModule,
    EnzPredDataModule,
    CombinedDataModule,
    MergedDataModule,
)
from ultrafast.model import DrugTargetCoembeddingLightning
from ultrafast.drug_only_model import DrugOnlyLightning
from ultrafast.utils import get_featurizer, xavier_normal  # noqa: F401


# ------------------ Callback ------------------ #
class PCBAEvaluationCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        # Respect CLI/config options if present
        eval_pcba(
            trainer,
            pl_module,
            pcba_dir=getattr(pl_module.args, "pcba_dir", "data/lit_pcba"),
            target_name=getattr(pl_module.args, "pcba_target", None),
        )


# ------------------ CLI entry ------------------ #
def train_cli():
    parser = argparse.ArgumentParser(description="PLM_DTI Training.")

    # Core exp
    parser.add_argument("--exp-id", required=True, dest="experiment_id", help="Experiment ID")
    parser.add_argument("--config", default="configs/default_config.yaml", help="YAML config file")
    parser.add_argument("--wandb-proj", dest="wandb_proj", help="Weights and Biases Project")

    # Task
    parser.add_argument(
        "--task",
        choices=[
            "biosnap",
            "bindingdb",
            "davis",
            "biosnap_prot",
            "biosnap_mol",
            "dti_dg",
            "merged",
            "custom",
        ],
        type=str,
        help="Task name.",
    )
    parser.add_argument("--drug-featurizer", dest="drug_featurizer", help="Drug featurizer")
    parser.add_argument("--target-featurizer", dest="target_featurizer", help="Target featurizer")

    # Training
    parser.add_argument("--ligand-only", action="store_true", help="Only use ligand features")
    parser.add_argument("--epochs", type=int, help="number of total epochs to run")
    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float, help="initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for optimizer")
    parser.add_argument("--clr", type=float, dest="clr", help="contrastive initial learning rate")
    parser.add_argument("--CEWeight", "-C", default=1.0, type=float, help="Cross Entropy loss weight", dest="CEWeight")
    parser.add_argument("--InfoNCEWeight", "-I", default=0.0, type=float, help="InfoNCE loss weight", dest="InfoNCEWeight")
    parser.add_argument("--InfoNCETemp", "-T", default=1.0, type=float, help="InfoNCE temperature", dest="InfoNCETemp")
    parser.add_argument("--r", "--replicate", dest="replicate", type=int, help="Replicate")
    parser.add_argument("--d", "--device", dest="device", default=0, type=int, help="CUDA device")
    parser.add_argument("--checkpoint", default=None, help="Model weights to start from")
    parser.add_argument("--prot-proj", choices=["avg", "agg", "transformer"], help="Protein projector method")
    parser.add_argument("--out-type", choices=["cls", "mean"], help="use cls token or mean of everything else")
    parser.add_argument("--num-layers-target", dest="num_layers_target", type=int, help="Target transformer layers")
    parser.add_argument("--num-heads-agg", dest="num_heads_agg", type=int, default=4, help="Attention heads")
    parser.add_argument("--dropout", type=float, help="Dropout rate")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size for training/val/test")
    parser.add_argument("--num-workers", type=int, default=0, help="dataloader workers")
    parser.add_argument("--no-wandb", action="store_true", help="Do not use wandb")
    parser.add_argument("--model-size", default="small", choices=["small", "large"], help="Choose the size of the model")
    parser.add_argument("--ship-model", dest="ship_model", help="Enable ship mode (truthy path/string)")
    parser.add_argument("--eval-pcba", action="store_true", help="Evaluate PCBA during validation")
    parser.add_argument("--sigmoid-scalar", type=int, default=5, dest="sigmoid_scalar")

    # NEW single-target eval + dynamic ship filtering parameters
    parser.add_argument(
        "--pcba-target",
        type=str,
        default=None,
        help="Evaluate only this Lit-PCBA target (also used for ship exclusion if provided)",
    )
    parser.add_argument(
        "--ship-sim-threshold",
        type=float,
        default=0.90,
        help="Sequence identity threshold for removal when ship training",
    )
    parser.add_argument(
        "--pcba-dir",
        type=str,
        default="data/lit_pcba",
        help="Path to Lit-PCBA root",
    )

    args = parser.parse_args()
    train(**vars(args))


# ------------------ Train ------------------ #
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
    eval_pcba_flag: bool,  # Renamed to avoid conflict with imported function
    sigmoid_scalar: int,
    # NEW
    pcba_target: str,
    ship_sim_threshold: float,
    pcba_dir: str,
):
    # Merge CLI into config
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
        eval_pcba=eval_pcba_flag,  # Use renamed variable
        sigmoid_scalar=sigmoid_scalar,
        # NEW
        pcba_target=pcba_target,
        ship_sim_threshold=ship_sim_threshold,
        pcba_dir=pcba_dir,
    )
    cfg = OmegaConf.load(args.config)
    cfg.update({k: v for k, v in vars(args).items() if v is not None})

    # Validate ship mode parameters
    if cfg.ship_model and not cfg.pcba_target:
        print("WARNING: ship_model is enabled but no pcba_target specified.")
        print("This will use file-based exclusion (if ship_model is a file path) or fail.")
        print("For single-target filtering, use: --ship-model true --pcba-target TARGET_NAME")

    save_dir = f'{cfg.get("model_save_dir", ".")}/{cfg.experiment_id}'

    # Device
    device_no = cfg.device
    use_cuda = torch.cuda.is_available()
    torch_device = torch.device(f"cuda:{device_no}" if use_cuda else "cpu")
    print(f"Using CUDA device {torch_device}")
    torch.set_float32_matmul_precision("medium")

    # Repro
    print(f"Setting random state {cfg.replicate}")
    torch.manual_seed(cfg.replicate)
    np.random.seed(cfg.replicate)

    # Data / featurizers
    print("Preparing DataModule")
    task_dir = get_task_dir(cfg.task)
    drug_feat = get_featurizer(cfg.drug_featurizer, save_dir=task_dir, n_jobs=cfg.num_workers)
    target_feat = get_featurizer(cfg.target_featurizer, save_dir=task_dir)

    # Task-specific dm kwargs
    if cfg.task == "dti_dg":
        cfg.classify = False
        cfg.watch_metric = "val/mse"
        task_dm_kwargs = dict(
            data_dir=task_dir,
            drug_featurizer=drug_feat,
            target_featurizer=target_feat,
            device=torch_device,
            seed=cfg.replicate,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers,
        )
    elif cfg.task in EnzPredDataModule.dataset_list():
        raise RuntimeError("EnzPredDataModule not implemented yet")
    else:
        cfg.classify = True
        cfg.watch_metric = "val/aupr"
        task_dm_kwargs = dict(
            data_dir=task_dir,
            drug_featurizer=drug_feat,
            target_featurizer=target_feat,
            device=torch_device,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers,
        )

    # Build datamodule
    if cfg.contrastive:
        print("Loading contrastive data (DUDE)")
        dude_drug = get_featurizer(cfg.drug_featurizer, save_dir=get_task_dir("DUDe"), ext="pt")
        dude_target = get_featurizer(cfg.target_featurizer, save_dir=get_task_dir("DUDe"))
        contrastive_dm_kwargs = dict(
            contrastive_split=cfg.contrastive_split,
            drug_featurizer=dude_drug,
            target_featurizer=dude_target,
            device=torch_device,
            batch_size=cfg.contrastive_batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers,
            contrastive_type=cfg.contrastive_type,
        )
        datamodule = CombinedDataModule(task=cfg.task, task_kwargs=task_dm_kwargs, contrastive_kwargs=contrastive_dm_kwargs)
    else:
        if cfg.task == "dti_dg":
            datamodule = TDCDataModule(**task_dm_kwargs)
        elif cfg.task in EnzPredDataModule.dataset_list():
            raise RuntimeError("EnzPredDataModule not implemented yet")
        elif cfg.task == "merged":
            # For merged task with new single-target filtering
            datamodule = MergedDataModule(
                **task_dm_kwargs,
                ship_model=cfg.ship_model,
                ship_model_target=cfg.pcba_target,  # Direct access after config merge
                ship_sim_threshold=cfg.ship_sim_threshold,
                pcba_dir=cfg.pcba_dir,
            )
        else:
            datamodule = DTIDataModule(**task_dm_kwargs)

    # Only call prepare_data/setup for non-merged tasks
    # MergedDataModule handles its own setup
    if cfg.task != "merged":
        datamodule.prepare_data()
        datamodule.setup()

    # -------- Model (single, de-duped) -------- #
    ckpt_path = cfg.get("checkpoint", None)

    if not cfg.ligand_only:
        if ckpt_path:
            print(f"Loading model from checkpoint: {ckpt_path}")
            model = DrugTargetCoembeddingLightning.load_from_checkpoint(
                ckpt_path,
                drug_dim=drug_feat.shape,
                target_dim=target_feat.shape,
                latent_dim=cfg.latent_dimension,
                classify=cfg.classify,
                contrastive=cfg.contrastive,
                InfoNCEWeight=cfg.InfoNCEWeight,
                prot_proj=cfg.prot_proj,
                dropout=cfg.dropout,
                device=torch_device,
                args=cfg,
            )
        else:
            print("Initializing new DrugTargetCoembeddingLightning")
            model = DrugTargetCoembeddingLightning(
                drug_dim=drug_feat.shape,
                target_dim=target_feat.shape,
                latent_dim=cfg.latent_dimension,
                classify=cfg.classify,
                contrastive=cfg.contrastive,
                InfoNCEWeight=cfg.InfoNCEWeight,
                prot_proj=cfg.prot_proj,
                dropout=cfg.dropout,
                args=cfg,
            )
    else:
        if ckpt_path:
            print(f"Loading ligand-only model from checkpoint: {ckpt_path}")
            model = DrugOnlyLightning.load_from_checkpoint(
                ckpt_path,
                drug_dim=drug_feat.shape,
                latent_dim=cfg.latent_dimension,
                classify=cfg.classify,
                contrastive=cfg.contrastive,
                InfoNCEWeight=cfg.InfoNCEWeight,
                dropout=cfg.dropout,
                device=torch_device,
                args=cfg,
            )
        else:
            print("Initializing new DrugOnlyLightning")
            model = DrugOnlyLightning(
                drug_dim=drug_feat.shape,
                latent_dim=cfg.latent_dimension,
                classify=cfg.classify,
                contrastive=cfg.contrastive,
                InfoNCEWeight=cfg.InfoNCEWeight,
                dropout=cfg.dropout,
                args=cfg,
            )

    # -------- Logging & Checkpoints -------- #
    if not cfg.no_wandb:
        wandb_logger = WandbLogger(project=cfg.wandb_proj, log_model="all")
        wandb_logger.watch(model)
        if hasattr(wandb_logger.experiment.config, "update"):
            wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
        wandb_logger.experiment.tags = [cfg.task, cfg.experiment_id, cfg.target_featurizer, cfg.model_size]
    else:
        wandb_logger = None

    if cfg.task == "merged" and cfg.ship_model:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=-1, dirpath=save_dir, verbose=True)
    else:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=cfg.watch_metric,
            mode="max" if "mse" not in cfg.watch_metric else "min",
            filename=cfg.task,
            dirpath=save_dir,
            verbose=True,
        )

    callbacks = [checkpoint_callback]
    if cfg.eval_pcba:
        callbacks.append(PCBAEvaluationCallback())

    # -------- Trainer -------- #
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        strategy="auto",
        logger=wandb_logger,
        max_epochs=cfg.epochs,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1 if cfg.contrastive else 0,
        limit_test_batches=0 if cfg.ship_model else 1.0,
    )

    # -------- Fit/Test -------- #
    if cfg.ship_model:
        trainer.fit(model, datamodule=datamodule)

        # Use best checkpoint if available; otherwise current weights
        test_ckpt = checkpoint_callback.best_model_path or None
        trainer.test(datamodule=datamodule, ckpt_path=test_ckpt)
        trainer.save_checkpoint(f"{save_dir}/ship_model.ckpt")

        # Automatically evaluate best checkpoint on Lit-PCBA
        if cfg.eval_pcba:
            eval_ckpt = checkpoint_callback.best_model_path or None
            if eval_ckpt:
                print(f"\nEvaluating best checkpoint on Lit-PCBA: {eval_ckpt}\n")
                model_eval = model.__class__.load_from_checkpoint(eval_ckpt)
            else:
                print("\nEvaluating current weights on Lit-PCBA (no best checkpoint found)\n")
                model_eval = model
            eval_pcba(
                trainer,
                model_eval,
                pcba_dir=cfg.pcba_dir,
                target_name=cfg.pcba_target,  # Direct access after config merge
            )

    else:
        trainer.fit(model, datamodule=datamodule)
        final_ckpt = cfg.get("checkpoint", None) if cfg.epochs == 0 else (checkpoint_callback.best_model_path or None)
        trainer.test(datamodule=datamodule, ckpt_path=final_ckpt)

        # Automatically evaluate best checkpoint on Lit-PCBA
        if cfg.eval_pcba:
            eval_ckpt = checkpoint_callback.best_model_path or None
            if eval_ckpt:
                print(f"\nEvaluating best checkpoint on Lit-PCBA: {eval_ckpt}\n")
                model_eval = model.__class__.load_from_checkpoint(eval_ckpt)
            else:
                print("\nEvaluating current weights on Lit-PCBA (no best checkpoint found)\n")
                model_eval = model
            eval_pcba(
                trainer,
                model_eval,
                pcba_dir=cfg.pcba_dir,
                target_name=cfg.pcba_target,  # Direct access after config merge
            )


if __name__ == '__main__':
    train_cli()