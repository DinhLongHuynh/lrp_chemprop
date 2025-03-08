import pandas as pd
import numpy as np
import torch
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from chemprop import data
from Data_Preprocessor import Data_Preprocessor
import copy
from LRP_Explainer import LRP_Explainer
from Model_Extractor import Model_Extractor

class Faithfulness_Analyzer(Data_Preprocessor):
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
        
        # Investigate each atom in each compound
        for compound in range(self.num_compound):
            # Extract bmg_compound
            df_compound = self.data_frame.iloc[[compound]]
            dataset_compound = self.generate(df=df_compound,
                                             smiles_column = self.smiles_column,
                                             target_column = self.target_column,
                                             addH = self.addH,
                                             HB = self.HB)
            dataloader_compound = data.build_dataloader(dataset_compound,batch_size=df_compound.shape[0],shuffle=False)
            bmg_compound, V_d, X_d, Y, *_ = next(iter(dataloader_compound))

            # Extract drop_0 a.k.a. initial prediction
            initial_prediction = self.model(bmg_compound).detach().cpu().numpy().reshape(-1)
            drop_predictions['drop_0'].append(initial_prediction.item())

            # Try dropping each atom in the molecule
            important_atom_index = []
            for drop_time in range(1,self.num_drop):
                init_rmse = -5000
                for atom in range(bmg_compound.V.shape[0]):
                    if atom in important_atom_index:
                        continue
                    bmg_modified = copy.deepcopy(bmg_compound)
                    bmg_modified.V[atom] = 0
                    bmg_modified.V[important_atom_index] = 0

                    drop_prediction = self.model(bmg_modified).detach().cpu().numpy().reshape(-1)
                    rmse_drop_atom = root_mean_squared_error(initial_prediction,drop_prediction)
                    if rmse_drop_atom > init_rmse:
                        init_rmse = rmse_drop_atom
                        selected_atom = atom
                
                # Store the important atoms
                important_atom_index.append(selected_atom)
                bmg_modified = copy.deepcopy(bmg_compound)
                bmg_modified.V[selected_atom] = 0
                bmg_modified.V[important_atom_index] = 0
                drop_prediction = self.model(bmg_modified).detach().numpy().reshape(-1)
                drop_predictions['drop_'+str(drop_time)].append(drop_prediction.item())

        # Calculate rmse at each drop_time compare to drop_0
        rmse = [0]
        for drop_time in range(1,self.num_drop):
            rmse.append(root_mean_squared_error(drop_predictions['drop_0'],drop_predictions['drop_'+str(drop_time)]))
        
        rmse = np.array(rmse)

        # Save this one for saving time
        self.rmse_manual = rmse
        return rmse
        

    def shuffle_bmg(self):
        tensor = torch.arange(self.bmg.V.shape[0])
        index_tensor = self.bmg.batch
        selected_values = torch.tensor([],dtype=torch.int64)

        # Dictionary to store selected atoms for each compound
        chosen_per_index = {idx.item(): [] for idx in torch.unique(index_tensor)}

        # Perform multiple rounds
        for round_num in range(self.num_drop):
            round_selected = []  

            for idx in torch.unique(index_tensor):
                mask = index_tensor == idx  # Mask to filter atoms by each compound
                available_values = tensor[mask]  # Get values corresponding to atom

                #Exclude already chosen values
                remaining_values = available_values[~torch.isin(available_values, torch.tensor(chosen_per_index[idx.item()]))]

                if remaining_values.shape[0] > 0:  # Ensure values are left for selection
                    selected = remaining_values[torch.randperm(len(remaining_values))[0]]  # Select one random value
                    chosen_per_index[idx.item()].append(selected.item())  # Track selected values
                    round_selected.append(selected)

            # Concatenate the round selection with previous selections
            if round_selected:
                selected_values = torch.cat([selected_values, torch.tensor(round_selected,dtype=torch.int64)])

        return selected_values


    def random_control(self,seed=0):
        rmse = [0]
        initial_prediction = self.model(self.bmg).detach().cpu().numpy().reshape(-1)

        torch.manual_seed(seed)
        shuffled_indices = self.shuffle_bmg()

        for drop_time in range(1,self.num_drop):
            # Set up random indices to drop
            if self.data_frame.shape[0]*drop_time < self.bmg.V.shape[0]:
                num_atom_drop = self.data_frame.shape[0]*drop_time
                #print(num_atom_drop)
            else:
                num_atom_drop = self.bmg.V.shape[0]
                #print(num_atom_drop)
            
            # Drop random atom
            random_index  = shuffled_indices[:num_atom_drop]
            bmg_random_modified = copy.deepcopy(self.bmg)
            bmg_random_modified.V[random_index] = 0

            # Generate new prediction
            drop_prediction = self.model(bmg_random_modified).detach().numpy().reshape(-1)
            rmse.append(root_mean_squared_error(initial_prediction,drop_prediction))

        return rmse
        

    def lrp_drop(self,alpha_1,epsilon,alpha_2):
        extractor = Model_Extractor(self.model,self.loader)
        model_params_cache = extractor.params_cache
        activations_cache = extractor.activations_cache

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
            # Initialize a list to store indices of atoms to drop
            selected_indices = []
            # Iterate over each molecule in the batch
            for molecule_idx in range(self.bmg.batch.max() + 1):
                # Get all atom indices belonging to this molecule
                atom_indices = torch.nonzero(self.bmg.batch == molecule_idx, as_tuple=True)[0]
                # Filter for atoms with the minimum relevance score
                candidate_indices = atom_indices[relevance_atom_sum[atom_indices] == min_relevances[molecule_idx]]
                # Select the first atom from the candidates (or use another criterion like random selection)
                selected_indices.append(candidate_indices[0])
            # Store the selected indices for this drop_time
            important_indices[str(drop_time)] = torch.tensor(selected_indices)


        # Dropping atom and record RMSE
        for drop_time in range(1,self.num_drop):
            drop_indices = torch.concat([important_indices[str(accum_drop)] for accum_drop in range(1,drop_time+1)],dim=0)
            bmg_lrp_modified = copy.deepcopy(self.bmg)
            bmg_lrp_modified.V[drop_indices] = 0
            drop_prediction = self.model(bmg_lrp_modified).detach().numpy().reshape(-1)
            rmse.append(root_mean_squared_error(initial_prediction,drop_prediction))

        return rmse

    
    def analyzer(self,alpha_1,epsilon,alpha_2,seed=0):
        rmse_random = self.random_control(seed=seed)
        rmse_manual = self.rmse_manual
        rmse_lrp = self.lrp_drop(alpha_1=alpha_1,epsilon=epsilon,alpha_2=alpha_2)

        return rmse_manual, rmse_lrp, rmse_random
    
    def faithfulness(self,alpha_1,epsilon,alpha_2,seed=0):
        rmse_manual, rmse_lrp, rmse_random = self.analyzer(alpha_1,epsilon,alpha_2,seed=seed)

        faithfulness = (np.trapezoid(y=rmse_lrp,x=range(0,self.num_drop))-np.trapezoid(y=rmse_random,x=range(0,self.num_drop)))/(np.trapezoid(y=rmse_manual,x=range(0,self.num_drop))-np.trapezoid(y=rmse_random,x=range(0,self.num_drop)))
        return faithfulness







