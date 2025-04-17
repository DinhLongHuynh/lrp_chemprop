import pandas as pd
from rdkit import Chem
from tqdm import tqdm

tqdm.pandas()
def standardize_smiles(smiles,desalt,remove_stereo):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        if desalt:
            frags = Chem.GetMolFrags(mol, asMols=True)
            mol = max(frags, key=lambda m: m.GetNumAtoms())
        if remove_stereo:
            Chem.RemoveStereochemistry(mol)
        
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    
    except Exception as e:
        print(f"Error in standardizing {smiles}: {str(e)}")
        return mol 


def standardize_smiles_df(df,smile_columns = 'smiles',desalt=True,remove_stereo=True):
    df['smiles_standard'] = df[smile_columns].progress_apply(lambda smiles: standardize_smiles(smiles,desalt=desalt,remove_stereo=remove_stereo))
    return df

