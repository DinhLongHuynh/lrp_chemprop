from rdkit import Chem
from molvs import Standardizer

class SMILES_standardize:
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        Chem.SanitizeMol(self.mol)

    def standardizer(self):
        try:
            self.std_mol = Standardizer().fragment_parent(self.mol, skip_standardize=False)
            self.std_mol  = Standardizer().isotope_parent(self.std_mol, skip_standardize=False)
            self.std_mol  = Standardizer().charge_parent(self.std_mol, skip_standardize=False)
            #self.std_mol = Standardizer().stereo_parent(self.std_mol, skip_standardize=False)
            self.std_mol = Standardizer().tautomer_parent(self.std_mol, skip_standardize=False)
            return Chem.MolToSmiles(self.std_mol, canonical = True)
        except Exception:
            return self.smiles
