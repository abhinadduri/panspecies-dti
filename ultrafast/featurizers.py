from __future__ import annotations

import h5py
import torch
import numpy as np
import typing as T
import datamol as dm
import esm
import requests

from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer
from pathlib import Path
from functools import lru_cache
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, pipeline
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.rdmolops import RDKFingerprint
from ultrafast.utils import canonicalize
from ultrafast.saprot_utils import load_esm_saprot

def sanitize_string(s):
    return str(s).replace("/", "|")

class Featurizer:
    def __init__(self, name: str, shape: int, save_dir: Path=Path().absolute(), ext: str="h5"):
        self._name = name
        self._shape = shape
        self._save_path = save_dir / Path(f"{self._name}_features.{ext}")

        self._preloaded = False
        self._device = torch.device("cpu")
        self._cuda_registry = {}
        self._on_cuda = False
        self._features = {}
        self._file_dir = None

    def __call__(self, seq: str) -> torch.Tensor:
        if seq not in self.features:
            self._features[seq] = self.transform(seq)

        return self._features[seq]

    def _register_cuda(self, k: str, v, f=None):
        """
        Register an object as capable of being moved to a CUDA device
        """
        self._cuda_registry[k] = (v, f)

    def _transform(self, seq: str) -> torch.Tensor:
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
        # for k, v in self._features.items():
        #     self._features[k] = v.to(device)

    @lru_cache(maxsize=5000)
    def transform(self, seq: str) -> torch.Tensor:
        with torch.set_grad_enabled(False):
            feats = self._transform(seq)
            if self._on_cuda:
                feats = feats.to(self.device)
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

    def write_to_disk(
            self, seq_list: T.List[str], verbose: bool = True, file_path: Path = None
    ) -> None:
        if file_path is not None:
            # this is easier for now than changing the code above
            out_path = file_path
        else:
            out_path = self._save_path

        if str(out_path).endswith('.h5'):
            print(f"Writing {self.name} features to {out_path}")
            with h5py.File(out_path, "a") as h5fi:
                for seq in tqdm(seq_list, disable=not verbose, desc=self.name):
                    seq_h5 = sanitize_string(seq)
                    if seq_h5 in h5fi:
                        print(f"{seq} already in h5file")
                    feats = self.transform(seq)
                    dset = h5fi.require_dataset(seq_h5, feats.shape, np.float32)
                    dset[:] = feats.cpu().numpy()
        elif str(out_path).endswith('.pt'):
            features = {}
            seq_set = set(seq_list)
            for seq in tqdm(seq_set, disable=not verbose, desc=self.name):
                features[seq] = self.transform(seq)
            torch.save(features,out_path)

    def preload(
        self,
        seq_list: T.List[str],
        verbose: bool = True,
        write_first: bool = True,
        single_file: bool = True,
    ) -> None:
        print(f"Preloading {self.name} features from {self.path}")

        if write_first and not self._save_path.exists():
            self.write_to_disk(seq_list, verbose=verbose)

        if self._save_path.exists():
            if str(self._save_path).endswith('.h5'):
                with h5py.File(self._save_path, "r") as h5fi:
                    for seq in tqdm(seq_list, disable=not verbose, desc=self.name):
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
                feats = self.transform(seq)


                self._features[seq] = feats
            self._preloaded = True

        # seqs_sanitized = [sanitize_string(s) for s in seq_list]
        # feat_dict = load_hdf5_parallel(self._save_path, seqs_sanitized,n_jobs=32)
        # self._features.update(feat_dict)

class ChemGPTFeaturizer(Featurizer):
    def __init__(self, shape: int = 768, save_dir: Path = Path().absolute(), ext: str = "h5"):
        super().__init__("RoBertaZinc", shape, save_dir, ext)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer = PretrainedHFTransformer(kind='Roberta-Zinc480M-102M', notation='selfies', dtype=float, device=device)

    def _transform(self, smile: str) -> torch.Tensor:
        try:
            mol = dm.to_mol(smile)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smile}")
            features = self.transformer([smile])
            return torch.from_numpy(features[0]).float()
        except Exception as e:
            print(f"Error featurizing SMILES {smile}: {e}")
            return torch.zeros(self.shape)

    def write_to_disk(self, seq_list: List[str], verbose: bool = True, file_path: Path = None) -> None:
        if file_path is not None:
            out_path = file_path
        else:
            out_path = self._save_path

        print(f"Writing {self.name} features to {out_path}")
        
        with h5py.File(out_path, "a") as h5fi:
            for seq in tqdm(seq_list, disable=not verbose, desc=self.name):
                seq_h5 = sanitize_string(seq)
                if seq_h5 in h5fi:
                    logger.info(f"{seq} already in h5file")
                    continue
                try:
                    feats = self.transform(seq)
                    dset = h5fi.require_dataset(seq_h5, feats.shape, np.float32)
                    dset[:] = feats.cpu().numpy()
                except Exception as e:
                    logger.error(f"Error processing {seq}: {e}")

class MorganFeaturizer(Featurizer):
    def __init__(
        self,
        shape: int = 2048,
        radius: int = 2,
        save_dir: Path = Path().absolute(),
        ext: str = "h5",
    ):
        super().__init__("Morgan", shape, save_dir, ext)

        self._radius = radius

    def smiles_to_morgan(self, smile: str):
        """
        Convert smiles into Morgan Fingerprint.
        :param smile: SMILES string
        :type smile: str
        :return: Morgan fingerprint
        :rtype: np.ndarray
        """
        try:
            smile = canonicalize(smile)
            mol = Chem.MolFromSmiles(smile)
            features_vec = AllChem.GetMorganFingerprintAsBitVect(
                mol, self._radius, nBits=self.shape
            )
            features = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(features_vec, features)
        except Exception as e:
            print(f"rdkit not found this smiles for morgan: {smile} convert to all 0 features")
            print(e)
            features = np.zeros((self.shape,))
        return features

    def _transform(self, smile: str) -> torch.Tensor:
        # feats = torch.from_numpy(self._featurizer(smile)).squeeze().float()
        feats = (
            torch.from_numpy(self.smiles_to_morgan(smile)).squeeze().float()
        )
        if feats.shape[0] != self.shape:
            print("Failed to featurize: appending zero vector")
            feats = torch.zeros(self.shape)
        return feats

class ChemGPTAndFptFeaturizer(Featurizer):
    def __init__(self, chemgpt_shape: int = 256, fp_shape: int = 2048, fp_radius: int = 2, save_dir: Path = Path().absolute(), ext: str = "h5"):
        total_shape = chemgpt_shape + fp_shape
        super().__init__("ChemGPTAndFpt", total_shape, save_dir, ext)
        self.chemgpt_shape = chemgpt_shape
        self.fp_shape = fp_shape
        self.fp_radius = fp_radius
        self.transformer = PretrainedHFTransformer(kind='ChemGPT-19M', notation='selfies', dtype=float)

    def _transform(self, smile: str) -> torch.Tensor:
        try:
            # ChemGPT features
            mol = dm.to_mol(smile)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smile}")
            chemgpt_features = self.transformer([smile])
            chemgpt_tensor = torch.from_numpy(chemgpt_features[0]).float()

            # Morgan fingerprint features
            mol = Chem.MolFromSmiles(smile)
            fp_vec = AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_radius, nBits=self.fp_shape)
            fp_array = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp_vec, fp_array)
            fp_tensor = torch.from_numpy(fp_array).float()

            # Concatenate features
            return torch.cat([chemgpt_tensor, fp_tensor])

        except Exception as e:
            print(f"Error featurizing SMILES {smile}: {e}")
            return torch.zeros(self.shape)

    def write_to_disk(self, seq_list: List[str], verbose: bool = True, file_path: Path = None) -> None:
        if file_path is not None:
            out_path = file_path
        else:
            out_path = self._save_path

        print(f"Writing {self.name} features to {out_path}")
        
        with h5py.File(out_path, "a") as h5fi:
            for seq in tqdm(seq_list, disable=not verbose, desc=self.name):
                seq_h5 = sanitize_string(seq)
                if seq_h5 in h5fi:
                    print(f"{seq} already in h5file")
                    continue
                try:
                    feats = self.transform(seq)
                    dset = h5fi.require_dataset(seq_h5, feats.shape, np.float32)
                    dset[:] = feats.cpu().numpy()
                except Exception as e:
                    print(f"Error processing {seq}: {e}")

class ProtBertFeaturizer(Featurizer):
    def __init__(self, save_dir: Path = Path().absolute(), per_tok=False):
        super().__init__("ProtBert", 1024, save_dir)


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
        )

        self._register_cuda("model", self._protbert_model)
        self._register_cuda(
            "featurizer", self._protbert_feat, self._feat_to_device
        )

    def _feat_to_device(self, pipe, device):

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

    def _transform(self, seq: str):
        if len(seq) > self._max_len - 2:
            seq = seq[: self._max_len - 2]

        embedding = torch.tensor(self._cuda_registry["featurizer"][0](self._space_sequence(seq)))
        seq_len = len(seq)
        start_Idx = 1
        end_Idx = seq_len + 1
        feats = embedding.squeeze()[start_Idx:end_Idx]

        if self.per_tok:
            return feats

        # return the entire embedding
        return feats

class ESM2Featurizer(Featurizer):
    def __init__(self, shape: int = 1280, save_dir: Path = Path().absolute(), ext: str = "h5"):
        super().__init__("ESM2", shape, save_dir, ext)
        
        # Load ESM-2 model
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Move model to GPU if available
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device_)
        self.model.eval()  # Set the model to evaluation mode

    def _transform(self, seq: str) -> torch.Tensor:
        try:
            data = [("protein", seq)]
            batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)

            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_embeddings = results["representations"][33]

            # Return the full sequence of embeddings instead of just the CLS token
            return token_embeddings[0, 1:].squeeze(0)  # This will be a 2D tensor (sequence length, feature dimension)

        except Exception as e:
            print(f"Error featurizing sequence: {e}")
            return torch.zeros((1, self.shape), device=self.device)
        
    def write_to_disk(self, seq_list: List[str], verbose: bool = True, file_path: Path = None) -> None:
        if file_path is not None:
            out_path = file_path
        else:
            out_path = self._save_path

        print(f"Writing {self.name} features to {out_path}")
        
        with h5py.File(out_path, "a") as h5fi:
            for seq in tqdm(seq_list, disable=not verbose, desc=self.name):
                seq_h5 = self.sanitize_string(seq)
                if seq_h5 in h5fi:
                    print(f"{seq} already in h5file")
                    continue
                try:
                    feats = self.transform(seq)
                    dset = h5fi.require_dataset(seq_h5, feats.shape, np.float32)
                    dset[:] = feats.cpu().numpy()
                except Exception as e:
                    print(f"Error processing {seq}: {e}")

    @staticmethod
    def sanitize_string(s):
        return ''.join(c if c.isalnum() else '_' for c in s)

# SaProt Featurizer
class SaProtFeaturizer(Featurizer):
    def __init__(self, shape: int = 1280, save_dir: Path = Path().absolute(), ext: str = "h5"):
        super().__init__("SaProt", shape, save_dir, ext)
        
        # Load SaProt model
        model_path = "SaProt_650M_AF2.pt"
        if not Path(model_path).exists():
            # download the model file from the following link https://huggingface.co/westlake-repl/SaProt_650M_AF2/resolve/main/SaProt_650M_AF2.pt?download=true
            print("Downloading SaProt model...")
            response = requests.get("https://huggingface.co/westlake-repl/SaProt_650M_AF2/resolve/main/SaProt_650M_AF2.pt?download=true", model_path)  # download the model file
            with open(model_path, "wb") as f:
                f.write(response.content)

            
        self.model, self.alphabet = load_esm_saprot(model_path)
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Move model to GPU if available
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device_)
        self.model.eval()  # Set the model to evaluation mode

    def _transform(self, seq: str) -> torch.Tensor:
        try:
            data = [("protein", seq)]
            batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)

            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_embeddings = results["representations"][33]

            # Return the full sequence of embeddings instead of just the CLS token
            return token_embeddings[0, 1:].squeeze(0)  # This will be a 2D tensor (sequence length, feature dimension)

        except Exception as e:
            print(f"Error featurizing sequence: {e}")
            return torch.zeros((1, self.shape), device=self.device)
        
    def write_to_disk(self, seq_list: List[str], verbose: bool = True, file_path: Path = None) -> None:
        if file_path is not None:
            out_path = file_path
        else:
            out_path = self._save_path

        print(f"Writing {self.name} features to {out_path}")
        
        with h5py.File(out_path, "a") as h5fi:
            for seq in tqdm(seq_list, disable=not verbose, desc=self.name):
                try:
                    seq_h5 = self.sanitize_string(seq)
                except Exception as e:
                    print(f"Error processing {seq}: e")
                    continue
                if seq_h5 in h5fi:
                    print(f"{seq} already in h5file")
                    continue
                try:
                    feats = self.transform(seq)
                    dset = h5fi.require_dataset(seq_h5, feats.shape, np.float32)
                    dset[:] = feats.cpu().numpy()
                except Exception as e:
                    print(f"Error processing {seq}: {e}")

    @staticmethod
    def sanitize_string(s):
        return ''.join(c if c.isalnum() else '_' for c in s)
