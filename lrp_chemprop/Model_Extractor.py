import pandas as pd
import numpy as np
import seaborn as sns
from rdkit import Chem
import torch
from lightning import pytorch as pl
from chemprop import data, featurizers, models, nn
from chemprop.featurizers.molecule import MorganBinaryFeaturizer
from chemprop.nn.metrics import ChempropMetric,MSE,MAE,RMSE, R2Score
from rdkit.Chem import AllChem


class Model_Extractor:
    '''
    A class to extract parameters and activations from model

    Parameters:
    ----------
    model (Chemprop model): a chemprop model checkpoint (.ckpt)
    data_loader (Chemprop dataloader): a chemprop data loader, which can be prepared using 'Data_Prepocessor'.
    '''
    
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader
        self.params_cache = {}
        self.activations_cache = {}

        batch = next(iter(self.data_loader))
        bmg, V_d, X_d, Y, *_ = batch

        self.bmg = self.model.message_passing.graph_transform(bmg)
        self.extract_params()
        self.extract_activations()

        

    
    def extract_params(self):
        ''' To extract model parameters.'''
        
        self.params_cache['W_i'] = self.model.message_passing.W_i.weight
        self.params_cache['W_h'] = self.model.message_passing.W_h.weight
        self.params_cache['W_o'] = self.model.message_passing.W_o.weight
        self.params_cache['b_o'] = self.model.message_passing.W_o.bias
        self.params_cache['ffn_depth'] = len(self.model.predictor.ffn)

        for i in range(1, self.params_cache['ffn_depth']):
            if i == 1:
                self.params_cache[f'W_ffn_{i}'] = self.model.predictor.ffn[i-1][0].weight
                self.params_cache[f'b_ffn_{i}'] = self.model.predictor.ffn[i-1][0].bias
            else:
                self.params_cache[f'W_ffn_{i}'] = self.model.predictor.ffn[i-1][2].weight
                self.params_cache[f'b_ffn_{i}'] = self.model.predictor.ffn[i-1][2].bias

        self.params_cache['W_output'] = self.model.predictor.ffn[-1][2].weight
        self.params_cache['b_output'] = self.model.predictor.ffn[-1][2].bias
        self.params_cache['output_transform_mean'] = self.model.predictor.output_transform.mean
        self.params_cache['output_transform_scale'] = self.model.predictor.output_transform.scale
        self.params_cache['depth'] = self.model.message_passing.depth

    
    def extract_activations(self):
        ''' To extract model activations at different layers'''
        

        # Extract activations for feed forward neural net:
        self.activations_cache['output_scale'] = self.model.encoding(self.bmg, i=self.params_cache['ffn_depth'])
        self.activations_cache['output_unscale'] = (
            self.activations_cache['output_scale'] * self.params_cache['output_transform_scale']
            + self.params_cache['output_transform_mean']
        )

        for i in range(1, self.params_cache['ffn_depth']):
            self.activations_cache[f'ffn{i}'] = torch.clamp(self.model.encoding(self.bmg, i=i), min=0.0)

        self.activations_cache['ffn0'] = self.model.encoding(self.bmg, i=0)
        self.params_cache['d_V'] = self.bmg.V.shape[1]

        # Extract activations for aggregation layer:
        self.activations_cache['agg'] = self.model.message_passing(self.bmg)

        # Extract activations for message-passing process
        H_0 = self.model.message_passing.initialize(self.bmg)
        H = self.model.message_passing.tau(H_0)

        self.activations_cache['H_0_init'] = H_0
        self.activations_cache['H_0'] = H
        for i in range(1, self.model.message_passing.depth):
            if self.model.message_passing.undirected:
                H = (H + H[self.bmg.rev_edge_index]) / 2
            M = self.model.message_passing.message(H, self.bmg)
            H = self.model.message_passing.update(M, H_0)

            self.activations_cache[f'H_{i}'] = H
            self.activations_cache[f'M_{i}'] = M

        index_torch = self.bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])  # Assign edges to atoms
        self.activations_cache[f'M_{self.params_cache["depth"]}'] = torch.zeros(
            len(self.bmg.V), H.shape[1], dtype=H.dtype, device=H.device
        ).scatter_reduce_(0, index_torch, self.activations_cache[f'H_{self.params_cache["depth"]-1}'], reduce='sum', include_self=False)

        self.activations_cache['mp'] = torch.cat((self.bmg.V, self.activations_cache[f'M_{self.params_cache["depth"]}']), dim=1)
        self.activations_cache['bmg'] = torch.cat([self.bmg.V[self.bmg.edge_index[0]], self.bmg.E], dim=1)
