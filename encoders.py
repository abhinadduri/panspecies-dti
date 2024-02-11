from rdkit import Chem    
from rdkit.Chem.Fingerprints import FingerprintMols    
from rdkit.DataStructs import FingerprintSimilarity    
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Chem import AllChem

# Obtain the fingerprint of a molecule given its SMILES representation.
#
# The fingerprint returned is a Morgan fingerprint with radius 3 and 1024 bits. Each bit
# represents the presence or absence of a substructure in the molecule.
def fingerprint(smiles: str) -> ExplicitBitVect:
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)

# Returns the tanimoto similarity of two SMILES strings.
#
# The tanimoto similarity is the jaccard similarity of the fingerprints of the two molecules.
def tanimoto(smiles1: str, smiles2: str) -> float:
    fp1 = fingerprint(smiles1)
    fp2 = fingerprint(smiles2)
    return FingerprintSimilarity(fp1, fp2)
