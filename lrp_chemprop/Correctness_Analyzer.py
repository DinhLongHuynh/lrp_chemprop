import numpy as np
from scipy.stats import kendalltau
import torch

class Correctness_Analyzer():
    '''A class to perform correctness analysis given relevance score and ground truth of each atom.

    Parameters:
    ---------
    relevance_score (dict): a dictionary with keys of 'Compound_id' and values are lists of lrp relevance scores of each atom in the respective compounds.
    ground_truth (dict): a dictionary with keys of 'Compound_id' and values are lists of ground-truth scores of each atom in the respective compounds.
    '''
    
    def __init__(self, relevance_score, ground_truth):
        self.relevance_score = relevance_score
        self.ground_truth = ground_truth
        self.num_compound = len(ground_truth)

    
    def quantile_analyzer(self, quantile_width):
        '''Perform correctness analysis by considering the most important part of the molecule. The 'important part' was defined by the quantile threshold of relevance scores.
        
        Parameters:
        ---------
        quantile_width (float): the granularity of calculation. The quantile range is always from 0 to 1, with a width defined by quantile_width.

        Returns: 
        ----------
        tau_avgs (numpy array): an array that contains the average kendall tau's correlation at each quantile threshold.
        '''
        
        tau_avgs = []
        for i in np.arange(0,1+quantile_width,quantile_width):
            ground_truth_copy = {k: v.clone() for k, v in self.ground_truth.items()}
            relevance_score_copy = {k: v.clone() for k, v in self.relevance_score.items()}

            # Start dropping according to quantiles
            for compound in range(self.num_compound):
                quantile_relevance = relevance_score_copy[f'Compound {compound}'].quantile(i)
                relevance_score_copy[f'Compound {compound}'] = torch.where(
                    relevance_score_copy[f'Compound {compound}']<= quantile_relevance,
                    relevance_score_copy[f'Compound {compound}'],
                    1
                    )
                
                quantile_ground_truth = ground_truth_copy[f'Compound {compound}'].quantile(i)
                ground_truth_copy[f'Compound {compound}'] = torch.where(
                    ground_truth_copy[f'Compound {compound}'] <= quantile_ground_truth,
                    ground_truth_copy[f'Compound {compound}'],
                    1
                )

            # Compute kendal taus at recent quantiles
            taus = []
            for compound in range(0,self.num_compound):
                tau, p_value = kendalltau(ground_truth_copy['Compound '+str(compound)].detach(), relevance_score_copy['Compound '+str(compound)].detach(),nan_policy='omit')
                taus.append(tau)
                if np.isnan(tau):
                    print(f'Compound {compound} has NaN at quantile ({i})')
            
            tau_avg = np.mean(taus)
            tau_avgs.append(tau_avg)

        tau_avgs = np.array(tau_avgs)
        return tau_avgs
    

    def get_minimum_value(self,tensor,order):
        '''Determine the 'order'_th minimum value in a 1D tensor.
        
        Parameters:
        ---------
        tensor (torch Tensor): 1D tensor.
        order (int): the rank of the minimum value to retrieve.

        Returns: 
        ----------
        minimum (float): the 'order'_th minimum value in the tensor.
        '''
        
        tensor_sorted = tensor.sort().values
        
        if order > tensor.shape[-1]:
            return tensor.max().item()
        minimum = tensor_sorted[order-1]
        return minimum

    
    def atom_analyzer(self,num_atoms):
        '''Perform correctness analysis by considering the most important part of the molecule. The 'important part' was defined by the most important atoms according to relevance scores.
        
        Parameters:
        ---------
        num_atoms (int): Number of most important atoms to consider.

        Returns: 
        ----------
        tau_avgs (numpy array): an array that contains the average kendall tau's correlation at each num_atoms threshold.
        '''
        
        tau_avgs = []
        for num_atom in np.arange(1,num_atoms):
            ground_truth_copy = {k: v.clone() for k, v in self.ground_truth.items()}
            relevance_score_copy = {k: v.clone() for k, v in self.relevance_score.items()}

            # Start dropping according to quantiles
            for compound in range(self.num_compound):
                quantile_relevance = self.get_minimum_value(relevance_score_copy[f'Compound {compound}'],num_atom)
                relevance_score_copy[f'Compound {compound}'] = torch.where(
                    relevance_score_copy[f'Compound {compound}']<= quantile_relevance,
                    relevance_score_copy[f'Compound {compound}'],
                    1
                    )
                
                quantile_ground_truth = self.get_minimum_value(ground_truth_copy[f'Compound {compound}'],num_atom)
                ground_truth_copy[f'Compound {compound}'] = torch.where(
                    ground_truth_copy[f'Compound {compound}'] <= quantile_ground_truth,
                    ground_truth_copy[f'Compound {compound}'],
                    1
                )

            # Compute kendal taus at recent quantiles
            taus = []
            for compound in range(0,self.num_compound):
                tau, p_value = kendalltau(ground_truth_copy['Compound '+str(compound)].detach(), relevance_score_copy['Compound '+str(compound)].detach(),nan_policy='omit')
                taus.append(tau)
                if np.isnan(tau):
                    print(f'Compound {compound} has NaN at atom ({num_atom})')
            
            tau_avg = np.mean(taus)
            tau_avgs.append(tau_avg)

        tau_avgs = np.array(tau_avgs)
        return tau_avgs
        
            
        
