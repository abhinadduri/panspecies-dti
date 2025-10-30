from __future__ import annotations

import os
import argparse
from pathlib import Path
import shutil  # <-- added

import numpy as np
import torch
from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger

import wandb

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
from ultrafast.utils import get_featurizer


class PCBAEvaluationCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        pcba_dir = getattr(pl_module.args, "pcba_dir", "data/lit_pcba")
        target_name = getattr(pl_module.args, "pcba_target", None)

        if target_name is not None and str(target_name).strip().lower() in {"all", "*"}:
            target_name = None

        eval_pcba(
            trainer,
            pl_module,
            pcba_dir=pcba_dir,
            target_name=target_name,
        )


def _read_targets_from_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    out = []
    with path.open() as fh:
        for ln in fh:
            s = ln.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def _resolve_pcba_targets(pcba_dir: str) -> list[str]:
    env_file = os.environ.get("TARGET_FILE")
    if env_file:
        lst = _read_targets_from_file(Path(env_file))
        if lst:
            return lst
    lst = _read_targets_from_file(Path(pcba_dir) / "targets.txt")
    if lst:
        return lst
    raise RuntimeError(
        "No targets list found. Set TARGET_FILE to a file with one target per line "
        "or create data/lit_pcba/targets.txt"
    )


def _build_task_dm_kwargs(cfg, torch_device, drug_feat, target_feat):
    task = cfg.get("task")
    if task == "dti_dg":
        cfg["classify"] = False
        if "watch_metric" not in cfg:
            cfg["watch_metric"] = "val/mse"
        return dict(
            data_dir=get_task_dir(task),
            drug_featurizer=drug_feat,
            target_featurizer=target_feat,
            device=torch_device,
            seed=cfg.get("replicate", 0),
            batch_size=cfg.get("batch_size", 64),
            shuffle=cfg.get("shuffle", True),
            num_workers=cfg.get("num_workers", 0),
        )
    elif task in EnzPredDataModule.dataset_list():
        raise RuntimeError("EnzPredDataModule not implemented yet")
    else:
        cfg["classify"] = True
        if "watch_metric" not in cfg:
            cfg["watch_metric"] = "val/aupr"
        return dict(
            data_dir=get_task_dir(task),
            drug_featurizer=drug_feat,
            target_featurizer=target_feat,
            device=torch_device,
            batch_size=cfg.get("batch_size", 64),
            shuffle=cfg.get("shuffle", True),
            num_workers=cfg.get("num_workers", 0),
        )


def _build_datamodule(cfg, torch_device, drug_feat, target_feat):
    task_dm_kwargs = _build_task_dm_kwargs(cfg, torch_device, drug_feat, target_feat)

    if cfg.get("contrastive", False):
        print("Loading contrastive data (DUDE)")
        dude_drug = get_featurizer(cfg.get("drug_featurizer"), save_dir=get_task_dir("DUDe"), ext="pt")
        dude_target = get_featurizer(cfg.get("target_featurizer"), save_dir=get_task_dir("DUDe"))
        contrastive_dm_kwargs = dict(
            contrastive_split=cfg.get("contrastive_split", "default"),
            drug_featurizer=dude_drug,
            target_featurizer=dude_target,
            device=torch_device,
            batch_size=cfg.get("contrastive_batch_size", 64),
            shuffle=cfg.get("shuffle", True),
            num_workers=cfg.get("num_workers", 0),
            contrastive_type=cfg.get("contrastive_type", "default"),
        )
        return CombinedDataModule(task=cfg.get("task"), task_kwargs=task_dm_kwargs, contrastive_kwargs=contrastive_dm_kwargs)

    task = cfg.get("task")
    if task == "dti_dg":
        return TDCDataModule(**task_dm_kwargs)
    elif task in EnzPredDataModule.dataset_list():
        raise RuntimeError("EnzPredDataModule not implemented yet")
    elif task == "merged":
        return MergedDataModule(
            **task_dm_kwargs,
            ship_model=bool(cfg.get("ship_model", False)),
            ship_model_target=cfg.get("pcba_target"),
            ship_sim_threshold=cfg.get("ship_sim_threshold", 0.90),
            pcba_dir=cfg.get("pcba_dir", "data/lit_pcba"),
        )
    else:
        return DTIDataModule(**task_dm_kwargs)


def _build_model(cfg, torch_device, drug_feat, target_feat):
    ckpt_path = cfg.get("checkpoint", None)
    ligand_only = cfg.get("ligand_only", False)
    latent_dim = cfg.get("latent_dimension", 512)

    if not ligand_only:
        if ckpt_path:
            print(f"Loading model from checkpoint: {ckpt_path}")
            return DrugTargetCoembeddingLightning.load_from_checkpoint(
                ckpt_path,
                drug_dim=drug_feat.shape,
                target_dim=target_feat.shape,
                latent_dim=latent_dim,
                classify=cfg.get("classify", True),
                contrastive=cfg.get("contrastive", False),
                InfoNCEWeight=cfg.get("InfoNCEWeight", 0.0),
                prot_proj=cfg.get("prot_proj", "avg"),
                dropout=cfg.get("dropout", 0.1),
                device=torch_device,
                args=cfg,
            )
        print("Initializing new DrugTargetCoembeddingLightning")
        return DrugTargetCoembeddingLightning(
            drug_dim=drug_feat.shape,
            target_dim=target_feat.shape,
            latent_dim=latent_dim,
            classify=cfg.get("classify", True),
            contrastive=cfg.get("contrastive", False),
            InfoNCEWeight=cfg.get("InfoNCEWeight", 0.0),
            prot_proj=cfg.get("prot_proj", "avg"),
            dropout=cfg.get("dropout", 0.1),
            args=cfg,
        )
    else:
        if ckpt_path:
            print(f"Loading ligand-only model from checkpoint: {ckpt_path}")
            return DrugOnlyLightning.load_from_checkpoint(
                ckpt_path,
                drug_dim=drug_feat.shape,
                latent_dim=latent_dim,
                classify=cfg.get("classify", True),
                contrastive=cfg.get("contrastive", False),
                InfoNCEWeight=cfg.get("InfoNCEWeight", 0.0),
                dropout=cfg.get("dropout", 0.1),
                device=torch_device,
                args=cfg,
            )
        print("Initializing new DrugOnlyLightning")
        return DrugOnlyLightning(
            drug_dim=drug_feat.shape,
            latent_dim=latent_dim,
            classify=cfg.get("classify", True),
            contrastive=cfg.get("contrastive", False),
            InfoNCEWeight=cfg.get("InfoNCEWeight", 0.0),
            dropout=cfg.get("dropout", 0.1),
            args=cfg,
        )


def _build_checkpoint_callbacks(cfg, save_dir: str):
    is_regression = (cfg.get("task") == "dti_dg")
    watch_metric = "val/mse" if is_regression else "val/aupr"
    mode = "min" if is_regression else "max"
    user_watch = cfg.get("watch_metric")
    if user_watch in ("val/aupr", "val/mse"):
        watch_metric = user_watch
        mode = "min" if user_watch == "val/mse" else "max"
    elif user_watch:
        print(f"[INFO] Ignoring unsupported watch_metric='{user_watch}'. Using '{watch_metric}'.")
    metric_key_for_filename = "val_mse" if is_regression else "val_aupr"

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor=watch_metric,
            mode=mode,
            dirpath=save_dir,
            filename="{epoch:03d}-{" + metric_key_for_filename + ":.5f}",
            save_top_k=-1,
            verbose=True,
            every_n_epochs=1,
        )
    ]
    return callbacks


def _write_canonical_ckpt(best_path: str | None, save_dir: str, exp_id: str) -> str | None:
    """
    Copy the best checkpoint to a stable filename <exp_id>.ckpt
    so tests can find e.g. best_models/unittest/unittest.ckpt.
    """
    if not best_path or not os.path.exists(best_path):
        print("[WARN] No best checkpoint found to canonicalize.")
        return None
    os.makedirs(save_dir, exist_ok=True)
    canon = os.path.join(save_dir, f"{exp_id}.ckpt")
    try:
        shutil.copy2(best_path, canon)
        print(f"[INFO] Wrote canonical checkpoint: {canon}")
        return canon
    except Exception as e:
        print(f"[WARN] Could not write canonical checkpoint: {e}")
        return None


def _run_one_training(cfg):
    save_dir = f'{cfg.get("model_save_dir", ".")}/{cfg.get("experiment_id")}'
    device_no = cfg.get("device", 0)
    use_cuda = torch.cuda.is_available()
    torch_device = torch.device(f"cuda:{device_no}" if use_cuda else "cpu")
    print(f"Using CUDA device {torch_device}")
    torch.set_float32_matmul_precision("medium")
    print(f"Setting random state {cfg.get('replicate', 0)}")
    torch.manual_seed(cfg.get("replicate", 0))
    np.random.seed(cfg.get("replicate", 0))
    print("Preparing DataModule")
    task_dir = get_task_dir(cfg.get("task"))
    drug_feat = get_featurizer(cfg.get("drug_featurizer"), save_dir=task_dir, n_jobs=cfg.get("num_workers", 0))
    target_feat = get_featurizer(cfg.get("target_featurizer"), save_dir=task_dir)
    datamodule = _build_datamodule(cfg, torch_device, drug_feat, target_feat)
    if cfg.get("task") != "merged":
        datamodule.prepare_data()
        datamodule.setup()

    model = _build_model(cfg, torch_device, drug_feat, target_feat)

    if not cfg.get("no_wandb", False):
        wandb_logger = WandbLogger(project=cfg.get("wandb_proj"), log_model="all")
        wandb_logger.watch(model)
        if hasattr(wandb_logger.experiment.config, "update"):
            wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
        wandb_logger.experiment.tags = [
            cfg.get("task"),
            cfg.get("experiment_id"),
            cfg.get("target_featurizer"),
            cfg.get("model_size"),
        ]
    else:
        wandb_logger = None

    ckpt_callbacks = _build_checkpoint_callbacks(cfg, save_dir)

    callbacks = [*ckpt_callbacks]
    if cfg.get("eval_pcba", False):
        callbacks.append(PCBAEvaluationCallback())

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        strategy="auto",
        logger=wandb_logger,
        max_epochs=cfg.get("epochs", 100),
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1 if cfg.get("contrastive", False) else 0,
        limit_test_batches=1.0,
    )

    if cfg.get("ship_model"):
        trainer.fit(model, datamodule=datamodule)
        best_ckpt = ckpt_callbacks[0].best_model_path or None

        # Write canonical checkpoint <exp_id>.ckpt for tests
        _write_canonical_ckpt(best_ckpt, save_dir, cfg.get("experiment_id"))

        trainer.test(datamodule=datamodule, ckpt_path=best_ckpt)
        trainer.save_checkpoint(f"{save_dir}/ship_model.ckpt")

        if cfg.get("eval_pcba", False):
            pcba_cb = next((c for c in ckpt_callbacks if getattr(c, "monitor", None) and c.monitor.startswith("pcba/")), None)
            eval_ckpt = (pcba_cb.best_model_path if pcba_cb and pcba_cb.best_model_path else ckpt_callbacks[0].best_model_path) or None
            if eval_ckpt:
                print(f"\nEvaluating best checkpoint on Lit-PCBA: {eval_ckpt}\n")
                model_eval = model.__class__.load_from_checkpoint(eval_ckpt)
            else:
                print("\nEvaluating current weights on Lit-PCBA (no best checkpoint found)\n")
                model_eval = model
            eval_pcba(
                trainer,
                model_eval,
                pcba_dir=cfg.get("pcba_dir", "data/lit_pcba"),
                target_name=cfg.get("pcba_target"),
            )
    else:
        trainer.fit(model, datamodule=datamodule)
        pcba_cb = next((c for c in ckpt_callbacks if getattr(c, "monitor", None) and c.monitor.startswith("pcba/")), None)
        chosen_cb = pcba_cb or ckpt_callbacks[0]

        best_ckpt = chosen_cb.best_model_path or None

        # Write canonical checkpoint <exp_id>.ckpt for tests
        _write_canonical_ckpt(best_ckpt, save_dir, cfg.get("experiment_id"))

        final_ckpt = cfg.get("checkpoint", None) if cfg.get("epochs", 0) == 0 else best_ckpt
        trainer.test(datamodule=datamodule, ckpt_path=final_ckpt)

        if cfg.get("eval_pcba", False):
            eval_ckpt = chosen_cb.best_model_path or None
            if eval_ckpt:
                print(f"\nEvaluating best checkpoint on Lit-PCBA: {eval_ckpt}\n")
                model_eval = model.__class__.load_from_checkpoint(eval_ckpt)
            else:
                print("\nEvaluating current weights on Lit-PCBA (no best checkpoint found)\n")
                model_eval = model
            eval_pcba(
                trainer,
                model_eval,
                pcba_dir=cfg.get("pcba_dir", "data/lit_pcba"),
                target_name=cfg.get("pcba_target"),
            )


def train_cli():
    parser = argparse.ArgumentParser(description="PLM_DTI Training.")
    parser.add_argument("--exp-id", required=True, dest="experiment_id", help="Experiment ID")
    parser.add_argument("--config", default="configs/default_config.yaml", help="YAML config file")
    parser.add_argument("--wandb-proj", dest="wandb_proj", help="Weights and Biases Project")
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
    parser.add_argument("--batch-size", type=int, default=64, help="batch size for training and eval")
    parser.add_argument("--num-workers", type=int, default=0, help="dataloader workers")
    parser.add_argument("--no-wandb", action="store_true", help="Do not use wandb")
    parser.add_argument("--model-size", default="small", choices=["small", "large"], help="Choose the size of the model")
    parser.add_argument("--ship-model", action="store_true", help="Enable ship mode with on-the-fly similarity exclusions (MMseqs2)")
    parser.add_argument("--ship-sim-threshold", type=float, default=0.90, help="Sequence identity threshold for removal")
    parser.add_argument("--pcba-dir", type=str, default="data/lit_pcba", help="Path to Lit-PCBA root")
    parser.add_argument("--eval-pcba", action="store_true", help="Evaluate PCBA during validation")
    parser.add_argument("--sigmoid-scalar", type=int, default=5, dest="sigmoid_scalar")
    parser.add_argument("--pcba-target", type=str, default=None, help="Single Lit-PCBA target. Use ALL or * (or omit) to evaluate all targets each epoch.")
    parser.add_argument("--watch-metric", type=str, help="Primary metric to monitor for checkpoints (e.g., val/aupr, val/auroc, pcba/auroc)")
    parser.add_argument("--also-monitor", type=str, help="Comma-separated list of additional metrics to checkpoint (e.g., val/auroc,pcba/auroc)")

    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.update({k: v for k, v in vars(args).items() if v is not None})

    if "shuffle" not in cfg:
        cfg["shuffle"] = True
    if "latent_dimension" not in cfg:
        cfg["latent_dimension"] = 512
    if "num_workers" not in cfg:
        cfg["num_workers"] = 0

    ship_model_val = bool(cfg.get("ship_model", False))
    pcba_target_val = cfg.get("pcba_target", None)

    run_all = pcba_target_val is None or str(pcba_target_val).strip().lower() in {"all", "*"}
    if ship_model_val and run_all:
        print("Running MERGED training with ALL targets excluded (ship mode) and ALL targets evaluated each epoch")
        cfg["pcba_target"] = None
        _run_one_training(cfg)
        return

    if ship_model_val and not run_all and (pcba_target_val is None or str(pcba_target_val).strip() == ""):
        raise ValueError("--ship-model requires --pcba-target <TARGET> or set --pcba-target ALL to evaluate all targets.")

    _run_one_training(cfg)


def train(*, exp_id: str, config: str = "configs/default_config.yaml", **kwargs):
    if not exp_id:
        raise ValueError("exp_id is required")

    base = OmegaConf.load(config)
    updates = dict(kwargs)
    updates["experiment_id"] = exp_id
    cfg = OmegaConf.merge(base, updates)

    if "shuffle" not in cfg:
        cfg["shuffle"] = True
    if "latent_dimension" not in cfg:
        cfg["latent_dimension"] = 512
    if "num_workers" not in cfg:
        cfg["num_workers"] = 0

    _run_one_training(cfg)


if __name__ == "__main__":
    train_cli()
