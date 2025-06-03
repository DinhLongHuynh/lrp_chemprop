import pandas as pd
import numpy as np
import torch
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from chemprop import data
from Data_Preprocessor import Data_Preprocessor
import copy
from LRP_Explainer import LRP_Explainer
from Model_Extractor import model_extractor

class Faithfulness_Analyzer(Data_Preprocessor):
    '''A class to perform faithfulness analysis and compute faithfulness score
    
        Parameters:
        ----------
        data_frame (PandasDataFrame): a data frame that contains SMILES code of compounds.
        smiles_column (str): a string that indicates the SMILES column in the data frame.
        target_column (str): a string that indicates the target column (i.e. docking_scores, solubility) in the data frame.
        addH (boolean): to incorporate explicit hydrogen atoms into a molecular graph.
        HB (boolean): to incorporate additional HBD/HBA features for each atom in BatchMolGraph.
        num_drop (int): number of atom dropping rounds.
        model (Chemprop model): Chemprop model.
     '''
    
    def __init__(self,model,
                 data_frame,smiles_column='smiles',target_column='docking_score',
                 addH=False, HB=False,
                 num_drop=10):
        super().__init__()
        self.num_drop = num_drop
        self.data_frame = data_frame
        self.model = model
        self.addH = addH
        self.HB = HB
        self.smiles_column = smiles_column
        self.target_column = target_column


        dataset = self.generate(df=self.data_frame,
                                    smiles_column = self.smiles_column,
                                    target_column = self.target_column,
                                    addH = self.addH,
                                    HB = self.HB)
        data_loader = data.build_dataloader(dataset,batch_size=self.data_frame.shape[0],shuffle=False)
        bmg, *_ = next(iter(data_loader))
        self.bmg = bmg
        self.loader = data_loader


    def manual_control(self):
        self.num_compound = self.data_frame.shape[0]
        
        # Initialized container for predictions of drops atoms
        drop_predictions = {}
        for drop_time in range(self.num_drop):
            drop_predictions['drop_'+str(drop_time)] = []
        
        # Loop through each atom in each compound
        for compound in range(self.num_compound):
            df_compound = self.data_frame.iloc[[compound]]
            dataset_compound = self.generate(df=df_compound,
                                             smiles_column = self.smiles_column,
                                             target_column = self.target_column,
                                             addH = self.addH,
                                             HB = self.HB)
            dataloader_compound = data.build_dataloader(dataset_compound,batch_size=df_compound.shape[0],shuffle=False)
            bmg_compound, V_d, X_d, Y, *_ = next(iter(dataloader_compound))

            # Extract drop_0, aka initial prediction
            initial_prediction = self.model(bmg_compound).detach().cpu().numpy().reshape(-1)
            drop_predictions['drop_0'].append(initial_prediction.item())

            # Try dropping each atom in the molecule and record prediction change (delta)
            atom_index = []
            atom_delta = []
            for atom in range(bmg_compound.V.shape[0]):
                # Drop and generate prediction
                bmg_modified = copy.deepcopy(bmg_compound)
                bmg_modified.V[atom] = 0
                drop_prediction = self.model(bmg_modified).detach().cpu().numpy().reshape(-1)
                # Record atom index and prediction change
                atom_index.append(atom)
                atom_delta.append(root_mean_squared_error(initial_prediction,drop_prediction))
                # Sort
                sorted_atom_delta, sorted_atom_index = zip(*sorted(zip(atom_delta,atom_index),reverse=True))
                sorted_atom_delta = list(sorted_atom_delta)
                sorted_atom_index = list(sorted_atom_index)

            # Generate dropped prediction
            for drop_time in range(1,self.num_drop):
                important_index = sorted_atom_index[:drop_time]
                bmg_modified.V[important_index] = 0
                drop_prediction = self.model(bmg_modified).detach().cpu().numpy().reshape(-1)
                drop_predictions['drop_'+str(drop_time)].append(drop_prediction)

        # Calculate rmse at each drop_time compare to drop_0
        rmse = [0]
        for drop_time in range(1,self.num_drop):
            rmse.append(root_mean_squared_error(drop_predictions['drop_0'],drop_predictions['drop_'+str(drop_time)]))
        
        rmse = np.array(rmse)
        # Append to self for later calculation
        self.rmse_manual = rmse
        return rmse

        

    def shuffle_bmg(self):
        '''Perform random shuffling atom index in bmg in respect of compounds and num_drop.
        
        Returns:
        ----------
        selected_values (torch Tensor): a 1D Tensor with length of num_drop*num_compounds
        
        Whenever take a slice of selected_values[:num_drop*n], there are always n atoms from each compound in the slice.
        '''
        
        tensor = torch.arange(self.bmg.V.shape[0])
        index_tensor = self.bmg.batch
        selected_values = torch.tensor([],dtype=torch.int64)

        # Initialized containers 
        chosen_per_index = {idx.item(): [] for idx in torch.unique(index_tensor)}

        # Perform multiple rounds
        for round_num in range(self.num_drop):
            round_selected = []  
            for idx in torch.unique(index_tensor):
                mask = index_tensor == idx  
                available_values = tensor[mask] 
                remaining_values = available_values[~torch.isin(available_values, torch.tensor(chosen_per_index[idx.item()]))]

                if remaining_values.shape[0] > 0: 
                    selected = remaining_values[torch.randperm(len(remaining_values))[0]] 
                    chosen_per_index[idx.item()].append(selected.item())  
                    round_selected.append(selected)

            if round_selected:
                selected_values = torch.cat([selected_values, torch.tensor(round_selected,dtype=torch.int64)])

        return selected_values


    def random_control(self,seed=0):
        '''Perform random drop atom in the bmg.

        Parameters: 
        ----------
        seed (int): a seed for randomization of self.shuffle_bmg()
        
        Returns:
        ----------
        rmse (numpy array): an array of rmse compared to the initial prediction at each drop round.
        '''
        
        rmse = [0]
        initial_prediction = self.model(self.bmg).detach().cpu().numpy().reshape(-1)

        torch.manual_seed(seed)
        shuffled_indices = self.shuffle_bmg()

        for drop_time in range(1,self.num_drop):
            # Set up random indices to drop
            if self.data_frame.shape[0]*drop_time < self.bmg.V.shape[0]:
                num_atom_drop = self.data_frame.shape[0]*drop_time
            else:
                num_atom_drop = self.bmg.V.shape[0]
            
            # Drop random atom
            random_index  = shuffled_indices[:num_atom_drop]
            bmg_random_modified = copy.deepcopy(self.bmg)
            bmg_random_modified.V[random_index] = 0

            # Generate new prediction
            drop_prediction = self.model(bmg_random_modified).detach().numpy().reshape(-1)
            rmse.append(root_mean_squared_error(initial_prediction,drop_prediction))

        return rmse
        

    def lrp_drop(self,alpha_1,epsilon,alpha_2):
        '''Perform drop atom according to lrp relevance scores.

        Parameters: 
        ----------
        alpha_1 (float): parameter of LRP_Explainer
        epsilon (float): parameter of LRP_Explainer
        alpha_2 (float): parameter of LRP_Explainer
        
        Returns:
        ----------
        rmse (numpy array): an array of rmse compared to the inital prediction at each drop round.
        '''
        
        model_params_cache, activations_cache = model_extractor(self.model,self.loader)

        # Run LRP_Explainer
        relevance_atom_sum = LRP_Explainer(model=self.model,
                                   model_params_cache=model_params_cache,
                                   activations_cache=activations_cache,
                                   bmg = self.bmg,
                                   alpha_1=alpha_1, epsilon=epsilon, alpha_2=alpha_2).explain_all()

  

        #Initial prediction
        initial_prediction = self.model(self.bmg).detach().cpu().numpy().reshape(-1)
        
        #Initialized containers 
        rmse = [0]

        #Determine top important atoms
        important_indices = {}
        for drop_time in range(1, self.num_drop):
            if drop_time > 1:
                relevance_atom_sum[important_indices[str(drop_time - 1)]] = 1000  # Mark previous atoms as dropped
            # Determine the minimum relevance score for each molecule
            min_relevances = torch.zeros((self.bmg.batch.max() + 1,)).scatter_reduce(
                0, self.bmg.batch, relevance_atom_sum, reduce='min', include_self=False
                )

            selected_indices = []
            # Iterate over each molecule in the batch
            for molecule_idx in range(self.bmg.batch.max() + 1):
                atom_indices = torch.nonzero(self.bmg.batch == molecule_idx, as_tuple=True)[0]
                candidate_indices = atom_indices[relevance_atom_sum[atom_indices] == min_relevances[molecule_idx]]
                selected_indices.append(candidate_indices[0])
            important_indices[str(drop_time)] = torch.tensor(selected_indices)


        # Dropping atom and record RMSE
        for drop_time in range(1,self.num_drop):
            drop_indices = torch.concat([important_indices[str(accum_drop)] for accum_drop in range(1,drop_time+1)],dim=0)
            bmg_lrp_modified = copy.deepcopy(self.bmg)
            bmg_lrp_modified.V[drop_indices] = 0
            drop_prediction = self.model(bmg_lrp_modified).detach().numpy().reshape(-1)
            rmse.append(root_mean_squared_error(initial_prediction,drop_prediction))

        self.rmse_lrp = rmse
        return rmse

    
    def analyzer(self,seed=0):
        '''Perform forward pass of Faithfulness Test, calculate all manual, lrp, and random scenarios.

        Parameters: 
        ----------
        alpha_1 (float): parameter of LRP_Explainer
        epsilon (float): parameter of LRP_Explainer
        alpha_2 (float): parameter of LRP_Explainer
        seed (int): a seed for randomization of self.random_control()
        
        Returns:
        ----------
        rmse_manual (numpy array): an array of rmse compared to the initial prediction at each manual drop round.
        rmse_lrp (numpy array): an array of rmse compared to the initial prediction at each lrp drop round.
        rmse_random (numpy array): an array of rmse compared to the initial prediction at each random drop round.
        '''
        
        rmse_random = self.random_control(seed=seed)
        rmse_manual = self.rmse_manual
        rmse_lrp = self.rmse_lrp

        return rmse_manual, rmse_lrp, rmse_random
    
    def faithfulness(self,seed=0):
        '''Calculate faithfulness score.
        
        Parameters: 
        ----------
        alpha_1 (float): parameter of LRP_Explainer
        epsilon (float): parameter of LRP_Explainer
        alpha_2 (float): parameter of LRP_Explainer
        seed (int): a seed for randomization of self.random_control()
        
        Returns:
        ----------
        faithfulness (float): faithfulness score
        '''
        
        rmse_manual, rmse_lrp, rmse_random = self.analyzer(seed=seed)

        faithfulness = (np.trapezoid(y=rmse_lrp,x=range(0,self.num_drop))-np.trapezoid(y=rmse_random,x=range(0,self.num_drop)))/(np.trapezoid(y=rmse_manual,x=range(0,self.num_drop))-np.trapezoid(y=rmse_random,x=range(0,self.num_drop)))
        return faithfulness







