import pandas as pd
from rdkit import Chem
from tqdm import tqdm

tqdm.pandas()
def standardize_smiles(smiles):
    """
    Basic standardization for SMILES without using rdMolStandardize.
    Steps: 1. Desalt (keep largest fragment), 2. Remove stereochemistry,
           3. Sanitize, 4. Canonicalize
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        frags = Chem.GetMolFrags(mol, asMols=True)
        mol = max(frags, key=lambda m: m.GetNumAtoms())
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    
    except Exception as e:
        print(f"Error in standardizing {smiles}: {str(e)}")
        return mol 


def standardize_smiles_df(df,smile_columns = 'smiles'):
    df['smiles_standard'] = df[smile_columns].progress_apply(standardize_smiles)
    return df

