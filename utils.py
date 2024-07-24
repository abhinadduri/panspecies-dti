import featurizers
from rdkit import Chem
from torch.nn.init import xavier_normal_

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


