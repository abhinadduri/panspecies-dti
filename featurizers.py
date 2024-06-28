from __future__ import annotations

import h5py
import torch
import numpy as np
import typing as T
import datamol as dm

from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer
from pathlib import Path
from functools import lru_cache
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, pipeline
from rdkit.Chem import AllChem as Chem
from rdkit import DataStructs
from rdkit.Chem.rdmolops import RDKFingerprint
from utils import canonicalize

def sanitize_string(s):
    return s.replace("/", "|")

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
            seq_h5 = sanitize_string(seq)
            if not self._preloaded and self._h5 is not None and seq_h5 in self._h5:
                return torch.from_numpy(self._h5[seq_h5][:])
            else:
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
        for k, v in self._features.items():
            self._features[k] = v.to(device)

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

                        if self._on_cuda:
                            feats = feats.to(self.device)

                        self._features[seq] = feats
            elif str(self._save_path).endswith('.pt'):
                self._features.update(torch.load(self._save_path))

        else:
            for seq in tqdm(seq_list, disable=not verbose, desc=self.name):
                feats = self.transform(seq)

                if self._on_cuda:
                    feats = feats.to(self.device)

                self._features[seq] = feats
            self._preloaded = True

        # seqs_sanitized = [sanitize_string(s) for s in seq_list]
        # feat_dict = load_hdf5_parallel(self._save_path, seqs_sanitized,n_jobs=32)
        # self._features.update(feat_dict)

        self._update_device(self.device)

class ChemGPTFeaturizer(Featurizer):
    def __init__(self, shape: int = 1024, save_dir: Path = Path().absolute(), ext: str = "h5"):
        super().__init__("ChemGPT", shape, save_dir, ext)
        self.transformer = PretrainedHFTransformer(kind='ChemGPT-19M', notation='selfies', dtype=float)

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
            features_vec = Chem.GetMorganFingerprintAsBitVect(
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
