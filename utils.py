import featurizers

from rdkit import Chem
from rdkit.Chem import AllChem
from torch.nn.init import xavier_normal_

import numpy as np
import torch
from torch_geometric.data import Data

from typing import List

def onek_encoding(value, choices) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the `value` in a list of length `len(choices)`.
             If `value` is not in `choices`, then a `ValueError` is raised
    """
    encoding = [0] * len(choices)
    index = choices.index(value)
    encoding[index] = 1

    return encoding

def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding with an extra category for uncommon values.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

def get_featurizer(featurizer_string, *args, **kwargs):
    return getattr(featurizers, featurizer_string)(*args, **kwargs)

def xavier_normal(model):
    for name, param in model.named_parameters():
        if name.endswith('bias'):
            param.data.fill_(0)
        else:
            xavier_normal_(param.data)

def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return None

# Class to featurize a molecular graph to use with GNNs
class Molecule:
    def featurize_atom(self, atom: Chem.rdchem.Atom) -> torch.Tensor:
        return AtomFeatures.stokes_features(atom)

    @staticmethod
    def mass_features(atom: Chem.rdchem.Atom) -> torch.Tensor:
        return torch.tensor([atom.GetMass()])

    @staticmethod
    def stokes_features(atom: Chem.rdchem.Atom) -> torch.Tensor:
        atomic_num = onek_encoding_unk(atom.GetAtomicNum() - 1, list(range(100)))
        degree = onek_encoding_unk(atom.GetTotalDegree(), list(range(6)))
        charge = onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
        chiral_tag = onek_encoding_unk(int(atom.GetChiralTag()), list(range(4)))
        num_hs = onek_encoding_unk(int(atom.GetTotalNumHs()), list(range(5)))
        hybridization = onek_encoding_unk(int(atom.GetHybridization()), [ Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2, ])
        aromatic = [1 if atom.GetIsAromatic() else 0]
        mass = [atom.GetMass() * 0.01]
        features = sum([atomic_num, degree, charge, chiral_tag, num_hs, hybridization, aromatic, mass], start = [])
        features = np.array(features, dtype=np.float32)
        return torch.from_numpy(features)

    @staticmethod
    def bond_features(bond: Chem.rdchem.Bond):
        bt = bond.GetBondType()
        fbond = [
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
        ]
        fbond = np.array(fbond, dtype=np.float32)
        return fbond
