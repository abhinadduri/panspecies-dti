from __future__ import annotations

import glob
import json
import os
from functools import partial
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from ultrafast.datamodules import EmbedInMemoryDataset, embed_collate_fn
from ultrafast.utils import get_featurizer, CalcAUC, CalcBEDROC, CalcEnrichment

__all__ = ["PCBAEvaluationCallback", "eval_pcba"]

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback

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
except Exception:
    PCBAEvaluationCallback = None  # type: ignore


def _pick_pcba_seqdict_path(pcba_dir: str, model) -> str:
    featur = getattr(model.args, "target_featurizer", None)
    featur_name = None
    if isinstance(featur, str):
        featur_name = featur
    elif featur is not None:
        featur_name = getattr(featur, "name", None) or featur.__class__.__name__

    is_saprot = featur_name is not None and ("saprot" in featur_name.lower())
    fname = "saprot_sequence_dict.json" if is_saprot else "lit_pcba_sequence_dict.json"
    return os.path.join(pcba_dir, fname)


def eval_pcba(
    trainer,
    model,
    pcba_dir: str = "data/lit_pcba",
    target_name: Optional[str] = None,
) -> Dict:
    if target_name is not None and str(target_name).strip().lower() in {"all", "*"}:
        target_name = None

    if not os.path.isdir(pcba_dir):
        raise FileNotFoundError(f"Expected Lit PCBA data at {pcba_dir}.")

    seq_dict_path = _pick_pcba_seqdict_path(pcba_dir, model)
    if not os.path.isfile(seq_dict_path):
        other = (
            "lit_pcba_sequence_dict.json"
            if os.path.basename(seq_dict_path) == "saprot_sequence_dict.json"
            else "saprot_sequence_dict.json"
        )
        other_path = os.path.join(pcba_dir, other)
        raise FileNotFoundError(
            f"Missing sequence dictionary: {seq_dict_path}\n"
            f"If you intended to use the other format, check: {other_path}"
        )
    with open(seq_dict_path, "r") as f:
        seq_dict = json.load(f)

    target_folders: List[str] = []
    for target_folder in glob.glob(os.path.join(pcba_dir, "*")):
        if not os.path.isdir(target_folder):
            continue
        tname = os.path.basename(target_folder)
        if tname in ("__pycache__", "cache"):
            continue
        if target_name is not None and tname != target_name:
            continue
        target_folders.append(target_folder)

    if not target_folders:
        raise ValueError(f"No Lit PCBA targets found that match {target_name!r}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_targets: List[str] = []
    all_aurocs: List[float] = []
    all_bedrocs: List[float] = []
    all_efs: Dict[float, List[float]] = {}

    single_target_mode = target_name is not None
    ef_levels = getattr(getattr(model, "args", {}), "ef_levels", [0.005, 0.01, 0.05])
    ef_levels = [float(x) for x in ef_levels]

    for target_folder in target_folders:
        target = os.path.basename(target_folder)
        out_dir = os.path.join(pcba_dir, target)

        if target not in seq_dict:
            raise KeyError(
                f"Target {target!r} not found in {seq_dict_path}. "
                "Confirm it was included when building the sequence dict."
            )
        target_sequences = seq_dict[target]

        # Ligands
        active_smiles = set()
        all_smiles: List[str] = []
        with open(os.path.join(target_folder, "actives.smi"), "r") as f:
            for line in f:
                s = line.strip().split(" ")[0]
                active_smiles.add(s)
                all_smiles.append(s)
        with open(os.path.join(target_folder, "inactives.smi"), "r") as f:
            for line in f:
                s = line.strip().split(" ")[0]
                all_smiles.append(s)
        all_smiles = np.array(all_smiles)

        # Drug embeddings
        drug_featurizer = get_featurizer(
            model.args.drug_featurizer,
            save_dir=out_dir,
            batch_size=2048 * 8,
        ).to(device)

        drug_ds = EmbedInMemoryDataset(all_smiles, drug_featurizer)
        drug_loader = DataLoader(
            drug_ds,
            batch_size=2048,
            shuffle=False,
            collate_fn=partial(embed_collate_fn, moltype="drug"),
        )

        drug_embeddings = []
        with torch.no_grad():
            for mols in tqdm(drug_loader, desc=f"[{target}] Embedding ligands", total=len(drug_loader)):
                mols = mols.to(device)
                emb = model.embed(mols, sample_type="drug")
                drug_embeddings.append(emb.cpu().numpy())
        drug_embeddings = np.concatenate(drug_embeddings, axis=0)

        # Target embeddings
        target_featurizer = get_featurizer(
            model.args.target_featurizer,
            save_dir=out_dir,
            batch_size=16,
        ).to(device)

        target_ds = EmbedInMemoryDataset(target_sequences, target_featurizer)
        target_loader = DataLoader(
            target_ds,
            batch_size=16,
            shuffle=False,
            collate_fn=partial(embed_collate_fn, moltype="target"),
        )

        target_embeddings = []
        with torch.no_grad():
            for seqs in tqdm(target_loader, desc=f"[{target}] Embedding targets", total=len(target_loader)):
                seqs = seqs.to(device)
                emb = model.embed(seqs, sample_type="target")
                target_embeddings.append(emb.cpu().numpy())
        target_embeddings = np.concatenate(target_embeddings, axis=0)
        if target_embeddings.ndim == 1:
            target_embeddings = np.expand_dims(target_embeddings, axis=0)

        similarity_matrix = cosine_similarity(target_embeddings, drug_embeddings)
        max_similarities = np.max(similarity_matrix, axis=0)

        scores = []
        for i, smile in enumerate(all_smiles):
            is_active = 1 if smile in active_smiles else 0
            scores.append((max_similarities[i], is_active))
        scores.sort(key=lambda x: x[0], reverse=True)

        bedroc = float(CalcBEDROC(scores, 1, 85.0))
        auroc = float(CalcAUC(scores, 1))

        efs_out = CalcEnrichment(scores, 1, ef_levels)
        if isinstance(efs_out, dict):
            efs_dict = {float(k): float(v) for k, v in efs_out.items()}
        else:
            efs_dict = {float(k): float(v) for k, v in zip(ef_levels, efs_out)}

        all_targets.append(target)
        all_bedrocs.append(bedroc)
        all_aurocs.append(auroc)
        for k, v in efs_dict.items():
            all_efs.setdefault(float(k), []).append(float(v))

        if hasattr(model, "log"):
            model.log(f"pcba/{target}/auroc", auroc, on_epoch=True, prog_bar=False, logger=True)
            model.log(f"pcba/{target}/bedroc_85", bedroc, on_epoch=True, prog_bar=False, logger=True)
            for k, v in sorted(efs_dict.items()):
                model.log(f"pcba/{target}/ef_{k}", v, on_epoch=True, prog_bar=False, logger=True)

    if not single_target_mode:
        avg_auroc = float(np.mean(all_aurocs)) if all_aurocs else float("nan")
        avg_bedroc = float(np.mean(all_bedrocs)) if all_bedrocs else float("nan")
        avg_efs: Dict[float, float] = {k: float(np.mean(v)) if v else float("nan") for k, v in all_efs.items()}

        if hasattr(model, "log"):
            model.log("pcba/avg_auroc", avg_auroc, on_epoch=True, prog_bar=True, logger=True)
            model.log("pcba/avg_bedroc_85", avg_bedroc, on_epoch=True, prog_bar=False, logger=True)
            for k, v in sorted(avg_efs.items()):
                model.log(f"pcba/avg_ef_{k}", v, on_epoch=True, prog_bar=False, logger=True)

        ordered_avg_efs = {k: avg_efs[k] for k in sorted(avg_efs)}
        print(f"Average EF: {ordered_avg_efs}")
        print(f"Average BEDROC_85: {avg_bedroc:.3f}")
        print(f"Average AUROC: {avg_auroc:.3f}")
    else:
        avg_auroc = all_aurocs[0] if all_aurocs else float("nan")
        avg_bedroc = all_bedrocs[0] if all_bedrocs else float("nan")
        avg_efs = {k: (v[0] if v else float("nan")) for k, v in all_efs.items()}

        ordered_avg_efs = {k: avg_efs[k] for k in sorted(avg_efs)}
        only_target = all_targets[0] if all_targets else str(target_name)
        print(f"Single target {only_target} EF: {ordered_avg_efs}")
        print(f"Single target {only_target} BEDROC_85: {avg_bedroc:.3f}")
        print(f"Single target {only_target} AUROC: {avg_auroc:.3f}")

    return {
        "targets": all_targets,
        "aurocs": all_aurocs,
        "bedrocs": all_bedrocs,
        "efs": all_efs,
        "avg": {
            "auroc": avg_auroc,
            "bedroc_85": avg_bedroc,
            "efs": (avg_efs if single_target_mode else {k: float(np.mean(v)) for k, v in all_efs.items()}),
        },
    }
