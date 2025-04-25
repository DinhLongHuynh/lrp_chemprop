import pandas as pd
import numpy as np
import torch
from chemprop import data
import rdkit
from rdkit import Chem
from chemprop import data, featurizers
from chemprop.featurizers.molecule import MorganBinaryFeaturizer



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
        

    def get_mol_HBD_HBA(self,mols):
        '''A function to generate HBD_HBA properties for molecules
        
        Parameters:
        ---------
        mols (list): list of RDKit mol objects.
        
        Returns:
        ----------
        mol_HBs (list): list of array that contain HBD-HBA descriptor for molecules, shape of each array is (n_atom, 2)  '''
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
        return mol_HBs

    

    def dataset_generator(self):
        '''Prepare chemprop dataset without additional HBD/HBA feature.
    
        Returns:
        ----------
        dataset (Chemprop dataset): Chemprop dataset.
        '''
        
        morgan_fp = MorganBinaryFeaturizer()
        def datapoint_generator(df,smiles,y,addH,HB,morgan,weight):
            smis = df.loc[:,smiles].values
            ys = df.loc[:,[y]].values
            mols = [Chem.MolFromSmiles(smi) for smi in smis]

            if weight!= None:
                weights = df.loc[:,weight].values

            if HB:
                mol_HBs = self.get_mol_HBD_HBA(mols)
            else:
                mol_HBs = [None]*len(smis)

            if morgan:
                x_ds = [morgan_fp(mol) for mol in mols]
            else:
                x_ds = [None]*len(smis)
            
            datapoints = [data.MoleculeDatapoint.from_smi(smi,y,add_h=addH, V_f = mol_HB, x_d = x_d, weight=weight) for smi, y, mol_HB, x_d, weight in zip(smis,ys,mol_HBs,x_ds, weights)]
            return datapoints

        datapoints = datapoint_generator(df=self.df,smiles=self.smiles_column,y=self.target_column,addH=self.addH,HB=self.HB,morgan=self.morgan,weight=self.weight_column)
        dataset = data.MoleculeDataset(datapoints, featurizer=self.featurizer)
        return dataset
    

    

    def generate(self, df, smiles_column = 'smiles', target_column='docking_score', addH=False, HB = False, morgan = False, weight_column =None,
                 featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()):
        '''Generate chemprop dataset according to a given configuration

        Parameters:
        ----------
        df (Pandas DataFrame): a data frame that contains SMILES code of compounds.
        smiles_column (str): a string that indicates SMILES column in the data frame.
        target_column (str): a string that indicates the target column (i.e. docking_scores, solubility) in the data frame.
        addH (boolean): to incorporate explicit hydrogen atoms into a molecular graph.
        HB (boolean): to incorporate additional HBD/HBA features for each atom in BatchMolGraph.
        morgan (boolean): to incorporate morgan binaray fingerprint for each molecules
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
        self.morgan = morgan
        self.weight_column = weight_column
        

        return self.dataset_generator()
    
