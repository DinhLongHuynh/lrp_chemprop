import torch


class LRP_Rules:
    '''A class that contains all LRP_rules for different kind of layers'''
    
    def __init__(self):
        pass

    def positive_matmul(self, A, B):
        """
        Performs matrix multiplication with positive pairs only.
        """
        A_plus = torch.clamp(A, min=0)
        B_plus = torch.clamp(B, min=0)
        A_minus = torch.clamp(A, max=0)
        B_minus = torch.clamp(B, max=0)

        return torch.matmul(A_plus, B_plus) + torch.matmul(A_minus, B_minus)

    def negative_matmul(self, A, B):
        """
        Performs matrix multiplication with negative pairs only.
        """
        A_plus = torch.clamp(A, min=0)
        B_plus = torch.clamp(B, min=0)
        A_minus = torch.clamp(A, max=0)
        B_minus = torch.clamp(B, max=0)

        return torch.matmul(A_plus, B_minus) + torch.matmul(A_minus, B_plus)

    def lrp_dense_epsilon(self, activation_j, relevance_k, weights_matrix, bias_matrix=None, epsilon=0):
        """
        LRP for dense layers with LRP-epsilon rule
        """

        if bias_matrix is not None:
            zk = torch.matmul(activation_j, weights_matrix.T) + bias_matrix
        else:
            zk = torch.matmul(activation_j, weights_matrix.T)

        zk = zk + torch.where(zk >= 0, epsilon, -epsilon)
        sk = relevance_k / (zk + torch.where(zk >= 0, torch.finfo(zk.dtype).tiny, -torch.finfo(zk.dtype).tiny))
        cj = torch.matmul(sk, weights_matrix)
        relevance_j = activation_j * cj

        return relevance_j

    def lrp_dense_ab(self, activation_j, relevance_k, weights_matrix, alpha=1.0):
        """
        LRP for dense layers with LRP_alpha-beta rule.
        """

        beta = alpha - 1

        zk_plus = self.positive_matmul(activation_j, weights_matrix.T)
        zk_minus = self.negative_matmul(activation_j, weights_matrix.T)

        sk_plus = relevance_k / (zk_plus + torch.finfo(zk_plus.dtype).tiny)
        sk_minus = relevance_k / (zk_minus - torch.finfo(zk_minus.dtype).tiny)

        weights_plus = torch.clamp(weights_matrix, min=0.0)  
        weights_minus = torch.clamp(weights_matrix, max=0.0)  
        activation_plus = torch.clamp(activation_j, min=0.0)  
        activation_minus = torch.clamp(activation_j, max=0.0)

        cj_plus_1 = torch.matmul(sk_plus, weights_plus)
        cj_plus_2 = torch.matmul(sk_plus, weights_minus)
        cj_minus_1 = torch.matmul(sk_minus, weights_plus)
        cj_minus_2 = torch.matmul(sk_minus, weights_minus)

        relevance_plus = activation_plus * cj_plus_1 + activation_minus * cj_plus_2
        relevance_minus = activation_plus * cj_minus_2 + activation_minus * cj_minus_1

        return alpha * relevance_plus - beta * relevance_minus

    def lrp_aggregation_epsilon(self, activation_j, relevance_k, batch_index, epsilon=0, agg_func='sum'):
        """
        LRP for aggregation layers with LRP_epsilon rule.
        """

        num_compounds = batch_index.max().item() + 1
        num_atoms, d_h = activation_j.shape

        # Create sparse mask: [num_compounds, num_atoms]
        indices = torch.stack([batch_index, torch.arange(num_atoms, device=batch_index.device)])
        values = torch.ones(num_atoms, device=batch_index.device)
        mask = torch.sparse_coo_tensor(indices, values, (num_compounds, num_atoms))

        # Compute weights based on aggregation function
        if agg_func == 'sum':
            weights = mask.to_dense()
        elif agg_func == 'mean':
            atom_counts = torch.sparse.sum(mask, dim=1).to_dense().clamp(min=1)
            weights = mask.to_dense() / atom_counts.unsqueeze(1)

        # Apply gamma rule to weights
        adj_weights = weights

        # Aggregate activations with adjusted weights
        adj_weights_sparse = mask * adj_weights
        z_k = torch.sparse.mm(adj_weights_sparse, activation_j)  # [num_compounds, d_h]

        # Stabilize z_k to avoid division by zero
        z_k = z_k + torch.where(z_k >= 0, torch.finfo(z_k.dtype).tiny, -torch.finfo(z_k.dtype).tiny) + torch.where(
            z_k >= 0, epsilon, -epsilon)

        # Compute scaling factor s_k = relevance_k / z_k
        s_k = relevance_k / z_k

        # Distribute relevance back to atoms
        relevance_atoms = adj_weights_sparse.transpose(0, 1).mm(s_k)  # [num_atoms, d_h]

        # Element-wise multiply with original activations
        relevance_j = relevance_atoms * activation_j

        return relevance_j

    def relevance_split(self,relevance,tensor_1,tensor_2):
        s = tensor_1.abs() + tensor_2.abs() 
        safe_s = torch.where(s==0, torch.tensor(1e-20, device=s.device),s)
        relevance_1 = relevance*(tensor_1.abs()/safe_s)
        relevance_2 = relevance*(tensor_2.abs()/safe_s)
        return relevance_1, relevance_2

    def reverse_sort(self,derivative_tensor, index):
        reverse_index = torch.argsort(index)
        original_tensor = derivative_tensor[reverse_index]
        return original_tensor

class LRP_Explainer(LRP_Rules):
    """
    A class that performing Layer-wise Relevance Propagation (LRP) on Chemprop Architecture.
    """

    def __init__(self, model, model_params_cache, activations_cache, bmg, gamma=1.0, epsilon=0, alpha=1.0):
        """
        Initializes the model and LRP object.

        Args:
            model: The Chemprop model.
            model_params_cache (dict): Cached model parameters, which can be extract using Model_Extractor
            activations_cache (dict): Cached activations from forward pass, which can be extrect using Model_Extractor
            bmg: Chemprop's Batch Molecular Graph data structure 
            gamma (float): Gamma parameter for LRP-gamma
            epsilon (float): Epsilon parameter for LRP_epsilon
            alpha (float): Alpha parameter for LRP_alpha-beta
        """
        super().__init__() # Using all the rules in the parent class LRP_rules
        self.model = model
        self.model_params_cache = model_params_cache
        self.activations_cache = activations_cache
        self.bmg = bmg
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.relevances_cache = {}  # Store temporatily relevances



    def explain_all(self):
        """
        Performs the LRP analysis.
        """
        model = self.model
        model_params_cache = self.model_params_cache
        activations_cache = self.activations_cache
        bmg = self.bmg
        epsilon = self.epsilon
        gamma = self.gamma
        alpha = self.alpha

        relevances_cache = self.relevances_cache
        # Predictor
        relevances_cache['ffn' + str(model_params_cache['ffn_depth'] - 1)] = self.lrp_dense_epsilon(
            activation_j=activations_cache['ffn' + str(model_params_cache['ffn_depth'] - 1)],
            relevance_k=activations_cache['output_scale'],
            weights_matrix=model_params_cache['W_output'],
            epsilon=epsilon
        )

        for i in range(model_params_cache['ffn_depth'] - 2, -1, -1):
            relevances_cache['ffn' + str(i)] = self.lrp_dense_epsilon(
                activation_j=activations_cache['ffn' + str(i)],
                relevance_k=relevances_cache['ffn' + str(i + 1)],
                weights_matrix=model_params_cache['W_ffn_' + str(i + 1)],
                epsilon=epsilon
            )

        # Aggregation
        relevances_cache['agg'] = self.lrp_aggregation_epsilon(
            activation_j=activations_cache['agg'],
            relevance_k=relevances_cache['ffn0'],
            batch_index=bmg.batch,
            agg_func='sum',
            epsilon=0
        )

        relevances_cache['mp'] = self.lrp_dense_ab(
            activation_j=activations_cache['mp'],
            relevance_k=relevances_cache['agg'],
            weights_matrix=model_params_cache['W_o'],
            alpha=gamma
        )

        relevance_mp_backprop = relevances_cache['mp'][:, model_params_cache['d_V']:]
        relevance_mp_accummulate = relevances_cache['mp'][:, :model_params_cache['d_V']]
        relevances_cache['M_' + str(model_params_cache['depth'])] = relevance_mp_backprop
        relevances_cache['H_' + str(model_params_cache['depth'] - 1)] = self.lrp_aggregation_epsilon(
            activation_j=activations_cache['H_' + str(model_params_cache['depth'] - 1)],
            relevance_k=relevances_cache['M_' + str(model_params_cache['depth'])],
            batch_index=bmg.edge_index[1],
            agg_func='sum',
            epsilon = 0
        )

        # Message passing
        for i in range(model_params_cache['depth'] - 1, 0, -1):
            z_M = model.message_passing.W_h(activations_cache['M_' + str(i)])
            relevance_backprop, relevance_accumulate = self.relevance_split(relevances_cache['H_' + str(i)], z_M, activations_cache['H_0_init'])
            relevances_cache['H_' + str(i) + '_accumulate'] = relevance_accumulate
            relevances_cache['M_' + str(i)] = self.lrp_dense_ab(
                activation_j=activations_cache['M_' + str(i)],
                relevance_k=relevance_backprop,
                weights_matrix=model_params_cache['W_h'],
                alpha=alpha
            )

            index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, model_params_cache['W_h'].shape[1])
            M_atom = torch.zeros(len(bmg.V), model_params_cache['W_h'].shape[1],
                                 dtype=activations_cache['H_1'].dtype, device=activations_cache['H_1'].device). \
                scatter_reduce_(0, index_torch, activations_cache['H_' + str(i - 1)], reduce='sum',
                                include_self=False)

            M_all = M_atom[bmg.edge_index[0]]
            M_rev = activations_cache['H_' + str(i - 1)][bmg.rev_edge_index]

            relevance_M_all, relevance_M_rev = self.relevance_split(relevances_cache['M_' + str(i)], M_all, -M_rev)
            relevance_H_rev = self.reverse_sort(relevance_M_rev, bmg.rev_edge_index)  # re-arrange M_2_rev

            index_torch = bmg.edge_index[0].unsqueeze(1).repeat(1, model_params_cache['W_h'].shape[1])
            relevance_M_atom = torch.zeros(len(bmg.V), model_params_cache['W_h'].shape[1],
                                           dtype=activations_cache['H_1'].dtype,
                                           device=activations_cache['H_1'].device). \
                scatter_reduce_(0, index_torch, relevance_M_all, reduce='sum', include_self=False)

            relevance_H_all = self.lrp_aggregation_epsilon(
                activation_j=activations_cache['H_' + str(i - 1)],
                relevance_k=relevance_M_atom,
                batch_index=bmg.edge_index[1],
                agg_func='sum',
                epsilon = 0
            )
            relevances_cache['H_' + str(i - 1)] = relevance_H_all + relevance_H_rev

        # Plus accumulate relevances
        result = relevances_cache['H_0']
        for i in range(1, model_params_cache['depth']):
            result += relevances_cache['H_' + str(i) + '_accumulate']
        relevances_cache['H_0_all'] = result

        # Initialize layer
        relevances_cache['bmg'] = self.lrp_dense_ab(activation_j=activations_cache['bmg'],
                                                    relevance_k=relevances_cache['H_0_all'],
                                                    weights_matrix=model_params_cache['W_i'],
                                                    alpha=alpha)
        relevance_starting_atom = relevances_cache['bmg'][:, :model_params_cache['d_V']]
        self.relevance_bond = relevances_cache['bmg'][:, model_params_cache['d_V']:]

        index_torch = bmg.edge_index[0].unsqueeze(1).repeat(1, bmg.V.shape[1])
        relevance_atom = torch.zeros(len(bmg.V), bmg.V.shape[1], dtype=activations_cache['H_1'].dtype,
                                     device=activations_cache['H_1'].device). \
            scatter_reduce_(0, index_torch, relevance_starting_atom, reduce='sum', include_self=False)
        self.relevance_atom = relevance_atom + relevance_mp_accummulate

        # relevance_atom_sum
        relevances_H_0 = relevances_cache['H_0_all']
        index_torch = bmg.edge_index[0].unsqueeze(1).repeat(1, relevances_H_0.shape[1])
        relevances_H_0_starting_atom = torch.zeros(bmg.V.shape[0], relevances_H_0.shape[1],
                                                    dtype=activations_cache['H_1'].dtype,
                                                    device=activations_cache['H_1'].device). \
            scatter_reduce_(0, index_torch, relevances_H_0, reduce='sum', include_self=False)
        relevances_H_0_all = relevances_H_0_starting_atom.sum(dim=1)
        relevance_accumulate_all = relevance_mp_accummulate.sum(dim=1)

        relevance_atom_sum = relevances_H_0_all + relevance_accumulate_all

        return relevance_atom_sum
    
    def explain_atom(self):
        pass
        return self.relevance_atom
    
    def explain_bond(self):
        pass
        return self.relevance_bond
