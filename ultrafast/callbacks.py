from __future__ import annotations

import glob
import json
import numpy as np
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import typing as T

from functools import partial
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader
from ultrafast.datamodules import EmbedInMemoryDataset, embed_collate_fn
from ultrafast.utils import get_featurizer, CalcAUC, CalcBEDROC, CalcEnrichment

def eval_dude(trainer, model, dude_dir='data/dude_vs'):
    """
    Evaluate the model checkpoint against the dude vs dataset.
    """

    all_targets = []
    all_aurocs = []
    all_bedrocs = []
    all_efs = {0.005: [], 0.01: [], 0.05: []}

    for target_folder in glob.glob(f'{dude_dir}/*'):
        if not os.path.isdir(target_folder):
            continue

        target = target_folder.split('/')[-1]

        # Get the SMILES data file for this target
        active_smiles = set()
        all_smiles = []
        for line in open(f'{target_folder}/actives_final.ism'):
            smiles_str = line.strip().split(' ')[0]
            active_smiles.add(smiles_str)
            all_smiles.append(smiles_str)

        for line in open(f'{target_folder}/decoys_final.ism'):
            smiles_str = line.strip().split(' ')[0]
            all_smiles.append(smiles_str)
        all_smiles = np.array(all_smiles)

        model.eval()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # Featurize the smiles
        drug_featurizer = get_featurizer(
            model.args.drug_featurizer,
            save_dir=target_folder,
            batch_size=2048 * 8,
        )
        drug_featurizer = drug_featurizer.to(device)

        dataset = EmbedInMemoryDataset(all_smiles, drug_featurizer)
        collate_fn = partial(embed_collate_fn, moltype='drug')
        dataloader = DataLoader(dataset, batch_size=2048, shuffle=False, collate_fn=collate_fn)
        drug_embeddings = []
        with torch.no_grad():
            for mols in tqdm(dataloader, desc="Embedding", total=len(dataloader)):
                mols = mols.to(device)
                emb = model.embed(mols, sample_type='drug')
                drug_embeddings.append(emb.cpu().numpy())
        drug_embeddings = np.concatenate(drug_embeddings, axis=0)
        print(drug_embeddings.shape)

        # Featurize the target sequences
        if model.args.target_featurizer == "SaProtFeaturizer":
            target_toks_file = f'{dude_dir}/dude_sequence_dict.json'
        else:
            target_toks_file = f'{dude_dir}/dude_sequence_dict.json'
        target_sequences = [json.load(open(target_toks_file))[target]]

        target_featurizer = get_featurizer(
            model.args.target_featurizer,
            save_dir=target_folder,
            batch_size=16,
        )
        target_featurizer = target_featurizer.to(device)

        dataset = EmbedInMemoryDataset(target_sequences, target_featurizer)
        collate_fn = partial(embed_collate_fn, moltype='target')
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
        target_embeddings = []
        with torch.no_grad():
            for seqs in tqdm(dataloader, desc="Embedding", total=len(dataloader)):
                seqs =seqs .to(device)
                emb = model.embed(seqs , sample_type='target')
                target_embeddings.append(emb.cpu().numpy())
        target_embeddings = np.concatenate(target_embeddings, axis=0)
        print(target_embeddings.shape)

        # expand target embeddings if needed
        if target_embeddings.ndim == 1:
            target_embeddings = np.expand_dims(target_embeddings, axis=0)
        print(target_embeddings.shape)

        similarity_matrix = cosine_similarity(target_embeddings, drug_embeddings)
        max_similarities = np.max(similarity_matrix, axis=0)

        # Prepare input for CalcBEDROC
        scores = []
        for i, smile in enumerate(all_smiles):
            is_active = 1 if smile in active_smiles else 0
            scores.append((max_similarities[i], is_active))
        
        # Sort scores in descending order of similarity
        scores.sort(key=lambda x: x[0], reverse=True)

        # Calculate BEDROC_85, AUROC, and EF
        bedroc = CalcBEDROC(scores, 1, 85.0)
        auroc = CalcAUC(scores, 1)
        efs = CalcEnrichment(scores, 1, [0.005, 0.01, 0.05])
        
        all_targets.append(target)
        for i, ef in enumerate(efs):
            all_efs[list(all_efs.keys())[i]].append(ef)
        all_bedrocs.append(bedroc)
        all_aurocs.append(auroc)

        # # Log individual target metrics
        # if trainer.logger:
        #     trainer.logger.experiment.log({
        #         f"dude/{target}/AUROC": auroc,
        #         f"dude/{target}/BEDROC_85": bedroc,
        #         f"dude/{target}/EF_0.005": efs[0],
        #         f"dude/{target}/EF_0.01": efs[1],
        #         f"dude/{target}/EF_0.05": efs[2],
        #     }, step=trainer.global_step)
        

    # Calculate and log average metrics
    avg_auroc = np.mean(all_aurocs)
    avg_bedroc = np.mean(all_bedrocs)
    avg_efs = {k: np.mean(v) for k, v in all_efs.items()}

    if trainer.logger:
        trainer.logger.experiment.log({
            "dude/avg_AUROC": avg_auroc,
            "dude/avg_BEDROC_85": avg_bedroc,
            "dude/avg_EF_0.005": avg_efs[0.005],
            "dude/avg_EF_0.01": avg_efs[0.01],
            "dude/avg_EF_0.05": avg_efs[0.05],
        }, step=trainer.global_step)


    print(f"Average EF: {avg_efs}")
    print(f"Average BEDROC_85: {avg_bedroc:.3f}")
    print(f"Average AUROC: {avg_auroc:.3f}")



def eval_pcba(trainer, model, pcba_dir='data/lit_pcba'):
    """
    Evaluate the model checkpoint against the lit-pcba dataset.

    # If structure aware ckpt, use the correct token file. detect this automatically
    # Assert an error if drug/lit_pcba does not exist, prompting the user to run data/download_pcba.py
    """

    all_targets = []
    all_aurocs = []
    all_bedrocs = []
    all_efs = {0.005: [], 0.01: [], 0.05: []}

    for target_folder in glob.glob(f'{pcba_dir}/*'):
        if not os.path.isdir(target_folder):
            continue

        target = target_folder.split('/')[-1]
        out_dir = f'{pcba_dir}/{target}'

        # Get the SMILES data file for this target
        active_smiles = set()
        all_smiles = []
        for line in open(f'{target_folder}/actives.smi'):
            smiles_str = line.strip().split(' ')[0]
            active_smiles.add(smiles_str)
            all_smiles.append(smiles_str)

        for line in open(f'{target_folder}/inactives.smi'):
            smiles_str = line.strip().split(' ')[0]
            all_smiles.append(smiles_str)
        all_smiles = np.array(all_smiles)

        model.eval()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # Featurize the smiles
        drug_featurizer = get_featurizer(
            model.args.drug_featurizer,
            save_dir=out_dir,
            batch_size=2048 * 8,
        )
        drug_featurizer = drug_featurizer.to(device)

        dataset = EmbedInMemoryDataset(all_smiles, drug_featurizer)
        collate_fn = partial(embed_collate_fn, moltype='drug')
        dataloader = DataLoader(dataset, batch_size=2048, shuffle=False, collate_fn=collate_fn)
        drug_embeddings = []
        with torch.no_grad():
            for mols in tqdm(dataloader, desc="Embedding", total=len(dataloader)):
                mols = mols.to(device)
                emb = model.embed(mols, sample_type='drug')
                drug_embeddings.append(emb.cpu().numpy())
        drug_embeddings = np.concatenate(drug_embeddings, axis=0)

        # Featurize the target sequences
        if model.args.target_featurizer == "SaProtFeaturizer":
            target_toks_file = f'{pcba_dir}/saprot_sequence_dict.json'
        else:
            target_toks_file = f'{pcba_dir}/lit_pcba_sequence_dict.json'
        target_sequences = json.load(open(target_toks_file))[target]

        target_featurizer = get_featurizer(
            model.args.target_featurizer,
            save_dir=out_dir,
            batch_size=16,
        )
        target_featurizer = target_featurizer.to(device)

        dataset = EmbedInMemoryDataset(target_sequences, target_featurizer)
        collate_fn = partial(embed_collate_fn, moltype='target')
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
        target_embeddings = []
        with torch.no_grad():
            for seqs in tqdm(dataloader, desc="Embedding", total=len(dataloader)):
                seqs =seqs .to(device)
                emb = model.embed(seqs , sample_type='target')
                target_embeddings.append(emb.cpu().numpy())
        target_embeddings = np.concatenate(target_embeddings, axis=0)

        # expand target embeddings if needed
        if target_embeddings.ndim == 1:
            target_embeddings = np.expand_dims(target_embeddings, axis=0)

        similarity_matrix = cosine_similarity(target_embeddings, drug_embeddings)
        max_similarities = np.max(similarity_matrix, axis=0)

        # Prepare input for CalcBEDROC
        scores = []
        for i, smile in enumerate(all_smiles):
            is_active = 1 if smile in active_smiles else 0
            scores.append((max_similarities[i], is_active))
        
        # Sort scores in descending order of similarity
        scores.sort(key=lambda x: x[0], reverse=True)

        # Calculate BEDROC_85, AUROC, and EF
        bedroc = CalcBEDROC(scores, 1, 85.0)
        auroc = CalcAUC(scores, 1)
        efs = CalcEnrichment(scores, 1, [0.005, 0.01, 0.05])
        
        all_targets.append(target)
        for i, ef in enumerate(efs):
            all_efs[list(all_efs.keys())[i]].append(ef)
        all_bedrocs.append(bedroc)
        all_aurocs.append(auroc)

        # Log individual target metrics
        if trainer.logger:
            trainer.logger.experiment.log({
                f"pcba/{target}/AUROC": auroc,
                f"pcba/{target}/BEDROC_85": bedroc,
                f"pcba/{target}/EF_0.005": efs[0],
                f"pcba/{target}/EF_0.01": efs[1],
                f"pcba/{target}/EF_0.05": efs[2],
            }, step=trainer.global_step)
        

    # Calculate and log average metrics
    avg_auroc = np.mean(all_aurocs)
    avg_bedroc = np.mean(all_bedrocs)
    avg_efs = {k: np.mean(v) for k, v in all_efs.items()}

    if trainer.logger:
        trainer.logger.experiment.log({
            "pcba/avg_AUROC": avg_auroc,
            "pcba/avg_BEDROC_85": avg_bedroc,
            "pcba/avg_EF_0.005": avg_efs[0.005],
            "pcba/avg_EF_0.01": avg_efs[0.01],
            "pcba/avg_EF_0.05": avg_efs[0.05],
        }, step=trainer.global_step)


    print(f"Average EF: {avg_efs}")
    print(f"Average BEDROC_85: {avg_bedroc:.3f}")
    print(f"Average AUROC: {avg_auroc:.3f}")


