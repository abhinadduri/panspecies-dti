from __future__ import annotations

import h5py
import torch
import multiprocessing
import numpy as np
import typing as T
import datamol as dm
import esm
import requests
import os
import pyxis as px

from functools import partial
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, pipeline
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Chem import rdFingerprintGenerator
from ultrafast.utils import canonicalize
from ultrafast.saprot_utils import load_esm_saprot

def sanitize_string(s):
    if isinstance(s, str):
        return s.replace("/", "|")
    elif pd.isna(s):
        return "NA"
    else:
        return str(s).replace("/", "|")

def batched(iterable, batch_size, func=None):
    """A simple batching function using only standard Python."""
    batch = []
    for item in iterable:
        batch.append(item) if func is None else batch.append(func(item))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:  # Don't forget the last batch if it's not full
        yield batch

class Featurizer:
    def __init__(self, name: str, shape: int, save_dir: Path=Path().absolute(), ext: str="h5", batch_size: int = 32):
        self._name = name
        self._shape = shape
        self._save_path = save_dir / Path(f"{self._name}_features.{ext}")

        self._preloaded = False
        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self._cuda_registry = {}
        self._on_cuda = False
        self._features = {}
        self._file_dir = None

        self._batch_size = batch_size

    def __call__(self, seq: str) -> torch.Tensor:
        if seq not in self.features:
            self._features[seq] = self.transform(seq)

        return self._features[seq]

    def _register_cuda(self, k: str, v, f=None):
        """
        Register an object as capable of being moved to a CUDA device
        """
        self._cuda_registry[k] = (v, f)

    def _transform(self, seq: List[str]) -> torch.Tensor:
        raise NotImplementedError

    def _update_device(self, device: torch.device):
        self._device = device
        for k, (v, f) in self._cuda_registry.items():
            if f is None:
                try:
                    self._cuda_registry[k] = (v.to(self._device), None)
                except RuntimeError as e:
                    print(e)
                    print(device)
                    print(type(self._device))
                    print(self._device)
            else:
                self._cuda_registry[k] = (f(v, self._device), f)

    def transform(self, seq_batch: List[str]) -> torch.Tensor:
        with torch.set_grad_enabled(False):
            feats = self._transform(seq_batch)
            return feats

    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> int:
        return self._shape

    @property
    def path(self) -> Path:
        return self._save_path

    @property
    def features(self) -> dict:
        return self._features

    @property
    def on_cuda(self) -> bool:
        return self._on_cuda

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device) -> Featurizer:
        self._update_device(device)
        self._on_cuda = device.type == "cuda"
        return self

    def cuda(self, device: torch.device) -> Featurizer:
        """
        Perform model computations on CUDA, move saved embeddings to CUDA device
        """
        self._update_device(device)
        self._on_cuda = True
        return self

    def cpu(self) -> Featurizer:
        """
        Perform model computations on CPU, move saved embeddings to CPU
        """
        self._update_device(torch.device("cpu"))
        self._on_cuda = False
        return self

    @staticmethod
    def prepare_string(seq):
        return seq

    def write_to_disk(
            self, seq_list: T.List[str], verbose: bool = True, file_path: Path = None, seq_func=None
    ) -> None:
        if file_path is not None:
            # this is easier for now than changing the code above
            out_path = file_path
        else:
            out_path = self._save_path

        print(f"Writing {self.name} features to {out_path}")
        batch_size = self._batch_size
        total_seqs = len(seq_list)

        if str(out_path).endswith('.h5'):
            with h5py.File(out_path, "a") as h5fi:
                with tqdm(total=total_seqs, desc=self.name) as pbar:
                    for batch in batched(seq_list, batch_size, func=seq_func):
                        batch_results = self.transform(batch)

                        for seq, feats in zip(batch, batch_results):
                            seq_h5 = sanitize_string(seq)
                            if seq_h5 in h5fi:
                                continue
                            dset = h5fi.require_dataset(seq_h5, feats.shape, np.float32)
                            dset[:] = feats.cpu().numpy()
                        pbar.update(batch_size)

        elif str(out_path).endswith('.pt'):
            features = {}
            seq_set = set(seq_list)
            with tqdm(total=total_seqs, desc=self.name) as pbar:
                for batch in batched(seq_list, batch_size):
                    batch_results = self.transform(batch)

                    for seq, feats in zip(batch, batch_results):
                        features[seq] = self.transform(seq)
                    pbar.update(batch_size)

            torch.save(features,out_path)

        elif str(out_path).endswith('.lmdb'):
            db = px.Writer(dirpath=str(out_path), map_size_limit=10000, ram_gb_limit=10)
            with tqdm(total=total_seqs, desc=self.name) as pbar:
                for batch in batched(seq_list, batch_size):
                    batch_results = self.transform(batch)
                    for seq, result in zip(batch, batch_results):
                        db.put_samples(seq, result)
                    pbar.update(batch_size)

            db.close()

    def _read_chunk(file_path, chunk):
        result = {}
        with h5py.File(file_path, "r") as h5fi:
            for seq in chunk:
                seq_h5 = sanitize_string(seq)
                if seq_h5 in h5fi:
                    feats = torch.from_numpy(h5fi[seq_h5][:])
                    result[seq] = feats
        return result

    def process_merged_drugs(self, id_to_smiles):
        """
        This function is intended to featurize the huge database file, and should not be used for any other database.
        Individual featurizers are able to process `batch_size` number of elements at once, so provide this size.
        """

        lmdb_path = 'data/MERGED/huge_data/smiles.lmdb'
        if os.path.exists(lmdb_path):
            print(f"File {lmdb_path} exists, skipping processing smiles.")
            return

        # Sort the IDs to ensure ascending order
        sorted_ids = sorted(id_to_smiles.keys())

        # Open LMDB environment, give it 5GB just in case.
        db = px.Writer(dirpath=lmdb_path, map_size_limit=50000, ram_gb_limit=10)

        # For each batch of 2048 id's, featurize them, e.g., call self.transform on a list of sequences
        batch_size = 2048 * 8
        for i in tqdm(range(0, len(sorted_ids), batch_size)):
            batch_ids = np.array(sorted_ids[i:i+batch_size])
            batch_smiles = [id_to_smiles[idx] for idx in batch_ids]

            # Compute Morgan fingerprints for the batch
            fingerprints = self.transform(batch_smiles)

            # for idx, result in zip(batch_ids, fingerprints):
            db.put_samples('ids', batch_ids, 'feats', fingerprints.numpy())

        print(f"Processed and stored {len(sorted_ids)} drug fingerprints in LMDB.")

        # Loading this data will require picking bin ids, then sampling from within those bin ids

    def process_merged_targets(self, id_to_target):
        """ 
        This function is intended to featurize the huge database file, and should not be used for any other database.
        """

        lmdb_path = f'data/MERGED/huge_data/{self.name}_targets.lmdb'
        if os.path.exists(lmdb_path):
            print(f"File {lmdb_path} exists, skipping processing protein targets.")
            return

        # retain only the keys for valid proteins
        id_list = []
        for k in list(id_to_target.keys()):
            if any(char.isdigit() for char in id_to_target[k]):
                continue
            id_list.append(k)

        # Open the LMDB env, give it 5GB just in case.
        db = px.Writer(dirpath=lmdb_path, map_size_limit=100000, ram_gb_limit=10)

        # For each batch, featurize them, e.g., call self.transform on a list of sequence
        batch_size = 16
        for i in tqdm(range(0, len(id_list), batch_size)):
            batch_ids = np.array(id_list[i:i+batch_size])
            # id to target is uniprot to aaseq
            batch_targets = [id_to_target[seq_id] for seq_id in batch_ids]

            feats = self.transform(batch_targets)

            for seq, results in zip(batch_ids, feats):
                seq_data = results.numpy()[np.newaxis, ...]
                db.put_samples(seq, seq_data)

    def preload(
        self,
        seq_list: T.List[str],
        verbose: bool = True,
        write_first: bool = True,
        single_file: bool = True,
        **kwargs
    ) -> None:
        print(f"Preloading {self.name} features from {self.path}")

        if write_first and not self._save_path.exists():
            self.write_to_disk(seq_list, verbose=verbose, **kwargs)

        if self._save_path.exists():
            if str(self._save_path).endswith('.h5'):
                with h5py.File(self._save_path, "r") as h5fi:
                    for seq in tqdm(seq_list, disable=not verbose, desc=self.name):
                        if "seq_func" in kwargs:
                            seq = kwargs["seq_func"](seq)
                        seq_h5 = sanitize_string(seq)
                        if seq_h5 in h5fi:
                            feats = torch.from_numpy(h5fi[seq_h5][:])
                        else:
                            feats = self.transform(seq)


                        self._features[seq] = feats
            elif str(self._save_path).endswith('.pt'):
                self._features.update(torch.load(self._save_path))

        else:
            for seq in tqdm(seq_list, disable=not verbose, desc=self.name):
                if "seq_func" in kwargs:
                    seq = kwargs["seq_func"](seq)
                feats = self.transform(seq)


                self._features[seq] = feats
            self._preloaded = True

class ChemGPTFeaturizer(Featurizer):
    def __init__(self, shape: int = 768, save_dir: Path = Path().absolute(), ext: str = "h5", batch_size: int = 32, n_jobs=-1):
        super().__init__("RoBertaZinc", shape, save_dir, ext, batch_size)
        self.transformer = PretrainedHFTransformer(kind='Roberta-Zinc480M-102M', notation='selfies', dtype=float, device=self._device)

    def _transform_single(self, smile: str) -> torch.Tensor:
        try:
            mol = dm.to_mol(smile)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smile}")
            features = self.transformer([smile])
            return torch.from_numpy(features[0]).float()
        except Exception as e:
            print(f"Error featurizing SMILES {smile}: {e}")
            return torch.zeros(self.shape)

    def _transform(self, batch_smiles: List[str]) -> torch.Tensor:
        try:
            mols = [dm.to_mol(smile) for smile in batch_smiles]
            invalid_indices = [i for i, mol in enumerate(mols) if mol is None]
            
            if invalid_indices:
                print(f"Invalid SMILES at indices: {invalid_indices}")
                for idx in invalid_indices:
                    mols[idx] = dm.to_mol("")  # Empty molecule as placeholder
            
            features = self.transformer(batch_smiles)
            features = torch.from_numpy(features).float()
            
            # Replace features for invalid SMILES with zero vectors
            for idx in invalid_indices:
                features[idx] = torch.zeros(self.shape)
            
            return features
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("CUDA out of memory during batch processing. Falling back to sequential processing.")
                return torch.stack([self._transform_single(smile) for smile in batch_smiles])
            else:
                raise e
        except Exception as e:
            print(f"Error during batch featurization: {e}")
            return torch.stack([self._transform_single(smile) for smile in batch_smiles])

class MorganFeaturizer(Featurizer):
    def __init__(
        self,
        shape: int = 2048,
        radius: int = 2,
        save_dir: Path = Path().absolute(),
        ext: str = "h5",
        batch_size: int = 2048,
        n_jobs: int = -1,
    ):
        super().__init__("Morgan", shape, save_dir, ext, batch_size)

        self._radius = radius
        # number of CPU workers to convert molecules to morgan fingerprints
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        print(f"Setup morgan featurizer with {self.n_jobs} workers")

    def smiles_to_morgan(self, smile: str):
        """
        Convert smiles into Morgan Fingerprint.
        :param smile: SMILES string
        :type smile: str
        :return: Morgan fingerprint
        :rtype: np.ndarray
        """
        if not isinstance(smile, str):
            if pd.isna(smile):
                print(f"Invalid SMILES: NaN")
                return np.zeros((self.shape,))
            else:
                smile = str(smile)

        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=self._radius,fpSize=self.shape)
        try:
            smile = canonicalize(smile)
            mol = Chem.MolFromSmiles(smile)
            features_vec = fpgen.GetFingerprint(mol)
            features = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(features_vec, features)
        except Exception as e:
            print(f"rdkit not found this smiles for morgan: {smile} convert to all 0 features")
            print(e)
            features = np.zeros((self.shape,))
        return features

    def _transform(self, batch_smiles: List[str]) -> torch.Tensor:
        with multiprocessing.Pool(processes=self.n_jobs) as pool:
            smiles_to_morgan_partial = partial(self.smiles_to_morgan)
            all_feats = pool.map(smiles_to_morgan_partial, batch_smiles)

            all_feats = [
                torch.from_numpy(feat).squeeze().float() if feat.shape[0] == self.shape \
                        else torch.zeros(self.shape) for feat in all_feats
            ]
            return torch.stack(all_feats, dim=0)

class ProtBertFeaturizer(Featurizer):
    def __init__(self, save_dir: Path = Path().absolute(), per_tok=False, **kwargs):
        super().__init__("ProtBert", 1024, save_dir)


        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self._max_len = 1024
        self.per_tok = per_tok

        self._protbert_tokenizer = AutoTokenizer.from_pretrained(
            "Rostlab/prot_bert",
            do_lower_case=False,
            cache_dir=f"models/huggingface/transformers",
        )
        self._protbert_model = AutoModel.from_pretrained(
            "Rostlab/prot_bert",
            cache_dir=f"models/huggingface/transformers",
        )
        self._protbert_feat = pipeline(
            "feature-extraction",
            model=self._protbert_model,
            tokenizer=self._protbert_tokenizer,
            device=self._device,
        )

        self._register_cuda("model", self._protbert_model)
        self._register_cuda(
            "featurizer", self._protbert_feat, self._feat_to_device
        )

    def _feat_to_device(self, pipe, device):
        self._device = device

        if device.type == "cpu":
            d = -1
        else:
            d = device.index

        pipe = pipeline(
            "feature-extraction",
            model=self._protbert_model,
            tokenizer=self._protbert_tokenizer,
            device=d,
        )
        self._protbert_feat = pipe
        return pipe

    def _space_sequence(self, x):
        return " ".join(list(x))

    def _transform(self, seqs: List[str]):
        max_seq_len = self._max_len - 2
        # Truncate sequences if necessary
        seqs = [seq[:max_seq_len] for seq in seqs]

        # Apply space_sequence to all sequences in the batch
        spaced_seqs = [self._space_sequence(seq) for seq in seqs]
        ids = self._protbert_tokenizer(spaced_seqs, padding=True, return_tensors="pt")
        input_ids = torch.tensor(ids['input_ids']).to(self._device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self._device)

        embeddings = self._protbert_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = embeddings.last_hidden_state.detach().cpu()

        results = []
        for i, seq in enumerate(seqs):
            seq_len = len(seq)
            start_idx = 1
            end_idx = seq_len + 1
            feats = embeddings[i].squeeze()[start_idx:end_idx]
            
            results.append(feats)

        return results

class ESM2Featurizer(Featurizer):
    def __init__(self, shape: int = 1280, save_dir: Path = Path().absolute(), ext: str = "h5", batch_size: int = 16):
        super().__init__("ESM2", shape, save_dir, ext, batch_size)

        print(f"Using ESM2 featurizer with {self._batch_size} batches")
        
        # Load ESM-2 model
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()

        # overload default of CPU
        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # Move model to GPU if available
        self.model = self.model.to(self._device)
        self.model.eval()  # Set the model to evaluation mode

    def _transform_single(self, seq: str) -> torch.Tensor:
        try:
            data = [("protein", seq)]
            _, _, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self._device)

            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_embeddings = results["representations"][33].detach().cpu()

            return token_embeddings[0, 1:].squeeze(0)  # Return the full sequence embedding
        except Exception as e:
            print(f"Error featurizing single sequence: {seq}")
            return torch.zeros((len(seq), self.shape)) # zero vector for each token

    def _transform(self, seqs: List[str]) -> List[torch.Tensor]:
        results = []
        try:
            data = [("protein", seq) for seq in seqs]
            _, _, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self._device)

            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_embeddings = results["representations"][33].detach().cpu()

            results = [token_embeddings[j, 1:].squeeze(0) for j in range(len(seqs))]
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                print("CUDA out of memory during batch processing. Falling back to sequential processing.")
                results = [self._transform_single(seq) for seq in seqs]
            else:
                raise e
        except Exception as e:
            print(f"Error during batch featurization: {e}")
            results = [self._transform_single(seq) for seq in seqs]
        
        torch.cuda.empty_cache()  # Clear GPU cache after processing
        return results
      
    @staticmethod
    def sanitize_string(s):
        return ''.join(c if c.isalnum() else '_' for c in s)
       
# SaProt Featurizer
class SaProtFeaturizer(Featurizer):
    def __init__(self, shape: int = 1280, save_dir: Path = Path().absolute(), ext: str = "h5", batch_size: int = 16):
        super().__init__("SaProt", shape, save_dir, ext, batch_size)
        
        # Load SaProt model
        model_path = "SaProt_650M_AF2.pt"
        if not Path(model_path).exists():
            # download the model file from the following link https://huggingface.co/westlake-repl/SaProt_650M_AF2/resolve/main/SaProt_650M_AF2.pt?download=true
            print("Downloading SaProt model...")
            response = requests.get("https://huggingface.co/westlake-repl/SaProt_650M_AF2/resolve/main/SaProt_650M_AF2.pt?download=true", model_path)  # download the model file
            with open(model_path, "wb") as f:
                f.write(response.content)

        self._max_len = 1024

            
        self.model, self.alphabet = load_esm_saprot(model_path)
        self.batch_converter = self.alphabet.get_batch_converter()

        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # Move model to GPU if available
        self.model = self.model.to(self._device)
        self.model.eval()  # Set the model to evaluation mode

    def _transform_single(self, seq: str) -> torch.Tensor:
        seq = SaProtFeaturizer.prepare_string(seq)
        try:
            data = [("protein", seq)]
            _, _, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self._device)

            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_embeddings = results["representations"][33].detach().cpu()

            return token_embeddings[0, 1:].squeeze(0)  # Return the full sequence embedding
        except Exception as e:
            print(f"Error featurizing single sequence: {seq}")
            return torch.zeros((len(seq), self.shape)) # zero vector for each token

    def _transform(self, seqs: List[str]) -> List[torch.Tensor]:
        results = []
        try:
            data = [("protein", seq) for seq in seqs]
            _, _, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self._device)

            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_embeddings = results["representations"][33].detach().cpu()

            results = [token_embeddings[j, 1:].squeeze(0) for j in range(len(seqs))]
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("CUDA out of memory during batch processing. Falling back to sequential processing.")
                results = [self._transform_single(seq) for seq in seqs]
            else:
                raise e
        except Exception as e:
            print(f"Error during batch featurization: {e}")
            results = [self._transform_single(seq) for seq in seqs]
        
        torch.cuda.empty_cache()  # Clear GPU cache after processing
        return results

    @staticmethod
    def prepare_string(seq, max_len=1024):
        if seq.isupper() and '#' not in seq: #if no 3Di tokens
            seq = '#'.join(seq) + '#'
        if len(seq) > max_len - 2:
            seq = seq[: max_len * 2 - 2]
        return seq
       
    @staticmethod
    def sanitize_string(s):
        return ''.join(c if c.isalnum() else '_' for c in s)
