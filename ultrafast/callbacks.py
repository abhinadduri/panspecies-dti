from __future__ import annotations

import glob
import json
import numpy as np
import os
import torch
from functools import partial
from typing import Optional, List
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

from ultrafast.datamodules import EmbedInMemoryDataset, embed_collate_fn
from ultrafast.utils import get_featurizer, CalcAUC, CalcBEDROC, CalcEnrichment


def _pick_pcba_seqdict_path(pcba_dir: str, model) -> str:
    """
    Choose the Lit-PCBA target sequence dictionary based on the model's target featurizer.
    - If SaProt-style featurizer: use saprot_sequence_dict.json
    - Otherwise: use lit_pcba_sequence_dict.json
    """
    featur = getattr(model.args, "target_featurizer", None)
    featur_name = None
    if isinstance(featur, str):
        featur_name = featur
    elif featur is not None:
        # Try common places a name might live
        featur_name = getattr(featur, "name", None) or featur.__class__.__name__

    is_saprot = featur_name is not None and ("saprot" in featur_name.lower())

    fname = "saprot_sequence_dict.json" if is_saprot else "lit_pcba_sequence_dict.json"
    return os.path.join(pcba_dir, fname)


def eval_pcba(trainer, model, pcba_dir: str = "data/lit_pcba", target_name: Optional[str] = None):
    """
    Evaluate the model checkpoint against the Lit-PCBA dataset.

    Requirements:
      - Either of these must exist in pcba_dir (selected automatically):
          * saprot_sequence_dict.json   (structure-aware SaProt tokens)
          * lit_pcba_sequence_dict.json (plain amino-acid sequences)
      - Each target folder must contain actives.smi and inactives.smi.
    """
    if not os.path.isdir(pcba_dir):
        raise FileNotFoundError(f"Expected Lit-PCBA data at {pcba_dir}.")

    # Pick and load the sequence dict
    seq_dict_path = _pick_pcba_seqdict_path(pcba_dir, model)
    if not os.path.isfile(seq_dict_path):
        # Fall back: if the picked one is missing but the *other* exists, hint about it
        other = ("lit_pcba_sequence_dict.json"
                 if os.path.basename(seq_dict_path) == "saprot_sequence_dict.json"
                 else "saprot_sequence_dict.json")
        other_path = os.path.join(pcba_dir, other)
        raise FileNotFoundError(
            f"Missing sequence dictionary: {seq_dict_path}\n"
            f"If you intended to use the other format, check: {other_path}"
        )
    seq_dict = json.load(open(seq_dict_path))

    # Build the list of target folders, optionally filtered by target_name
    target_folders: List[str] = []
    for target_folder in glob.glob(f"{pcba_dir}/*"):
        if not os.path.isdir(target_folder):
            continue
        tname = os.path.basename(target_folder)
        if tname in ("__pycache__", "cache"):
            continue
        if target_name is not None and tname != target_name:
            continue
        target_folders.append(target_folder)

    if not target_folders:
        raise ValueError(f"No Lit-PCBA targets found that match {target_name!r}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_targets, all_aurocs, all_bedrocs = [], [], []
    all_efs = {0.005: [], 0.01: [], 0.05: []}

    single_target_mode = target_name is not None

    for target_folder in target_folders:
        target = os.path.basename(target_folder)
        out_dir = os.path.join(pcba_dir, target)

        # Ensure target exists in selected dict
        if target not in seq_dict:
            raise KeyError(
                f"Target {target!r} not found in {seq_dict_path}. "
                "Confirm you included it when building the sequence dict."
            )
        target_sequences = seq_dict[target]

        # Load SMILES
        active_smiles = set()
        all_smiles = []
        with open(os.path.join(target_folder, "actives.smi")) as f:
            for line in f:
                s = line.strip().split(" ")[0]
                active_smiles.add(s)
                all_smiles.append(s)
        with open(os.path.join(target_folder, "inactives.smi")) as f:
            for line in f:
                s = line.strip().split(" ")[0]
                all_smiles.append(s)
        all_smiles = np.array(all_smiles)

        # Drug featurizer
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

        # Target featurizer
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

        # Similarity and metrics
        similarity_matrix = cosine_similarity(target_embeddings, drug_embeddings)
        max_similarities = np.max(similarity_matrix, axis=0)

        scores = []
        for i, smile in enumerate(all_smiles):
            is_active = 1 if smile in active_smiles else 0
            scores.append((max_similarities[i], is_active))
        scores.sort(key=lambda x: x[0], reverse=True)

        bedroc = CalcBEDROC(scores, 1, 85.0)
        auroc = CalcAUC(scores, 1)
        efs = CalcEnrichment(scores, 1, [0.005, 0.01, 0.05])

        all_targets.append(target)
        all_bedrocs.append(bedroc)
        all_aurocs.append(auroc)
        for i, k in enumerate([0.005, 0.01, 0.05]):
            all_efs[k].append(efs[i])

        if trainer.logger:
            trainer.logger.experiment.log(
                {
                    f"pcba/{target}/AUROC": auroc,
                    f"pcba/{target}/BEDROC_85": bedroc,
                    f"pcba/{target}/EF_0.005": efs[0],
                    f"pcba/{target}/EF_0.01": efs[1],
                    f"pcba/{target}/EF_0.05": efs[2],
                },
                step=trainer.global_step,
            )

    # Only calculate and log averages if we evaluated multiple targets
    if not single_target_mode:
        avg_auroc = float(np.mean(all_aurocs)) if all_aurocs else float("nan")
        avg_bedroc = float(np.mean(all_bedrocs)) if all_bedrocs else float("nan")
        avg_efs = {k: float(np.mean(v)) if v else float("nan") for k, v in all_efs.items()}

        if trainer.logger:
            trainer.logger.experiment.log(
                {
                    "pcba/avg_AUROC": avg_auroc,
                    "pcba/avg_BEDROC_85": avg_bedroc,
                    "pcba/avg_EF_0.005": avg_efs[0.005],
                    "pcba/avg_EF_0.01": avg_efs[0.01],
                    "pcba/avg_EF_0.05": avg_efs[0.05],
                },
                step=trainer.global_step,
            )

        print(f"Average EF: {avg_efs}")
        print(f"Average BEDROC_85: {avg_bedroc:.3f}")
        print(f"Average AUROC: {avg_auroc:.3f}")
    else:
        avg_auroc = all_aurocs[0] if all_aurocs else float("nan")
        avg_bedroc = all_bedrocs[0] if all_bedrocs else float("nan")
        avg_efs = {k: v[0] if v else float("nan") for k, v in all_efs.items()}

        print(f"Single target {target_name} EF: {avg_efs}")
        print(f"Single target {target_name} BEDROC_85: {avg_bedroc:.3f}")
        print(f"Single target {target_name} AUROC: {avg_auroc:.3f}")

    return {
        "targets": all_targets,
        "aurocs": all_aurocs,
        "bedrocs": all_bedrocs,
        "efs": all_efs,
        "avg": {"auroc": avg_auroc, "bedroc_85": avg_bedroc, "efs": avg_efs},
    }