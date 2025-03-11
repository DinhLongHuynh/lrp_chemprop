import pandas as pd
import numpy as np
import torch
from chemprop import data
import rdkit
from rdkit import Chem
from chemprop import data, featurizers



class Data_Preprocessor:
    '''A class to prepare Chemprop dataset from Pandas dataframe.'''
    
    def __init__(self):
        pass

    
    def is_hbd(self,atom):
        '''Check if an atom is a Hydrogen Bond Donor (HBD). An atom is considered an HBD if it's N or O with at least one hydrogen.
        
        Parameters:
        ----------
        atom: RDKit atom object.
            
        Returns:
        ----------
        bool: True if atom is HBD, False otherwise.
        '''
        
        if atom.GetAtomicNum() not in [7, 8]:  # 7 for N, 8 for O
            return False
        
        n_hydrogens = atom.GetTotalNumHs()
        return n_hydrogens > 0

    def is_hba(self,atom):
        '''Check if an atom is a Hydrogen Bond Acceptor (HBA). An atom is considered an HBA if it's N or O with a lone pair electron
        
        Parameters:
        ----------
        atom: RDKit atom object.
            
        Returns:
        ----------
        bool: True if atom is HBD, False otherwise.
        '''
        
        atomic_num = atom.GetAtomicNum()
        if atomic_num not in [7, 8]: 
            return False
        
        valence = atom.GetTotalValence()
        if atomic_num == 7:  
            return valence <= 3 
        else: 
            return valence <= 2  
    

    def dataset_generator_no_HBD_HBA(self):
        '''Prepare chemprop dataset without additional HBD/HBA feature.
    
        Returns:
        ----------
        dataset (Chemprop dataset): Chemprop dataset.
        '''
        
        def datapoint_generator(df,smiles,y,addH):
            smis = df.loc[:,smiles].values
            ys = df.loc[:,[y]].values
            datapoints = [data.MoleculeDatapoint.from_smi(smi,y,add_h=addH) for smi,y in zip(smis,ys)]

            return datapoints

        datapoints = datapoint_generator(df=self.df,smiles=self.smiles_column,y=self.target_column,addH=self.addH)
        dataset = data.MoleculeDataset(datapoints, featurizer=self.featurizer)
        return dataset
    

    def dataset_generator_HBD_HBA(self):
        '''Prepare chemprop dataset with additional HBD/HBA feature.
    
        Returns:
        ----------
        dataset (Chemprop dataset): Chemprop dataset.
        '''
        
        def datapoint_generator(df,smiles,y,addH):
            smis = df.loc[:,smiles].values
            ys = df.loc[:,[y]].values
            mols = [Chem.MolFromSmiles(smi) for smi in smis]

            mol_HBs = []
            for mol in mols:
                mol_HB = [[],[]]
                for atom in mol.GetAtoms():
                    if self.is_hbd(atom):
                        mol_HB[0].append(1)
                    else:
                        mol_HB[0].append(0)
                    
                    if self.is_hba(atom):
                        mol_HB[1].append(1)
                    else:
                        mol_HB[1].append(0)
                mol_HB = np.array(mol_HB).T
                mol_HBs.append(mol_HB)

            datapoints = [data.MoleculeDatapoint.from_smi(smi,y,add_h=addH,V_f=mol_HB) for smi,y,mol_HB in zip(smis,ys,mol_HBs)]

            return datapoints

        datapoints = datapoint_generator(df=self.df,smiles=self.smiles_column,y=self.target_column,addH=self.addH)
        dataset = data.MoleculeDataset(datapoints, featurizer=self.featurizer)
        return dataset
    

    def generate(self, df, smiles_column = 'smiles', target_column='docking_score', addH=False, HB = False,
                 featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()):
        '''Generate chemprop dataset according to a given configuration

        Parameters:
        ----------
        df (Pandas DataFrame): a data frame that contains SMILES code of compounds.
        smiles_column (str): a string that indicates SMILES column in the data frame.
        target_column (str): a string that indicates the target column (i.e. docking_scores, solubility) in the data frame.
        addH (boolean): to incorporate explicit hydrogen atoms into a molecular graph.
        HB (boolean): to incorporate additional HBD/HBA features for each atom in BatchMolGraph.
        featurizer (Chemprop Featurizer): a Featurizer from Chemprop to encode features for atoms, bonds, and molecules.
    
        Returns:
        ----------
        dataset (Chemprop dataset): Chemprop dataset
        '''
                     
        self.df = df
        self.smiles_column = smiles_column
        self.target_column = target_column
        self.addH = addH
        self.HB = HB
        self.featurizer = featurizer
        
        if self.HB:
            return self.dataset_generator_HBD_HBA()
        
        else: 
            return self.dataset_generator_no_HBD_HBA()
