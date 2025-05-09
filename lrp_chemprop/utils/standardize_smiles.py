from rdkit import Chem
from molvs import Standardizer

class SMILES_standardize:
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        Chem.SanitizeMol(self.mol)

    def standardizer(self):
        self.mol = Standardizer().fragment_parent(self.mol, skip_standardize=False)
        self.mol = Standardizer().isotope_parent(self.mol, skip_standardize=False)
        self.mol = Standardizer().charge_parent(self.mol, skip_standardize=False)
        #self.mol = Standardizer().stereo_parent(self.mol, skip_standardize=False)
        self.mol = Standardizer().tautomer_parent(self.mol, skip_standardize=False)
