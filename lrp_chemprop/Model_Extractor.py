import torch

def extract_params(model):
    '''To extracts model parameters.

    Parameters:
    ----------
    model (Chemprop model): a chemprop model checkpoint (.ckpt)

    Returns:
    --------
    params_cache (dict): a dictionary containing model parameters.
    '''

    params_cache = {
        'W_i': model.message_passing.W_i.weight,
        'W_h': model.message_passing.W_h.weight,
        'W_o': model.message_passing.W_o.weight,
        'b_o': model.message_passing.W_o.bias,
        'ffn_depth': len(model.predictor.ffn)
    }
    
    for i in range(1, params_cache['ffn_depth']):
        if i == 1:
            params_cache[f'W_ffn_{i}'] = model.predictor.ffn[i-1][0].weight
            params_cache[f'b_ffn_{i}'] = model.predictor.ffn[i-1][0].bias
        else:
            params_cache[f'W_ffn_{i}'] = model.predictor.ffn[i-1][2].weight
            params_cache[f'b_ffn_{i}'] = model.predictor.ffn[i-1][2].bias
    
    params_cache['W_output'] = model.predictor.ffn[-1][2].weight
    params_cache['b_output'] = model.predictor.ffn[-1][2].bias
    params_cache['output_transform_mean'] = model.predictor.output_transform.mean
    params_cache['output_transform_scale'] = model.predictor.output_transform.scale
    params_cache['depth'] = model.message_passing.depth
    
    return params_cache


def extract_activations(model, bmg, params_cache):
    '''To extracts activations of model.

    Parameters:
    ----------
    model (Chemprop model): a chemprop model checkpoint (.ckpt)
    bmg (BatchMolGraph): a chemprop batch of molecule graphs.
    params_cache (dict): a dictionary containing model parameters.

    Returns:
    --------
    activations_cache (dict): a dictionary containing activations.
    '''

    activations_cache = {}
    activations_cache['output_scale'] = model.encoding(bmg, i=params_cache['ffn_depth'])
    activations_cache['output_unscale'] = (
        activations_cache['output_scale'] * params_cache['output_transform_scale']
        + params_cache['output_transform_mean']
    )
    
    for i in range(1, params_cache['ffn_depth']):
        activations_cache[f'ffn{i}'] = torch.clamp(model.encoding(bmg, i=i), min=0.0)
    
    activations_cache['ffn0'] = model.encoding(bmg, i=0)
    params_cache['d_V'] = bmg.V.shape[1]
    
    activations_cache['agg'] = model.message_passing(bmg)
    
    H_0 = model.message_passing.initialize(bmg)
    H = model.message_passing.tau(H_0)
    
    activations_cache['H_0_init'] = H_0
    activations_cache['H_0'] = H
    
    for i in range(1, model.message_passing.depth):
        if model.message_passing.undirected:
            H = (H + H[bmg.rev_edge_index]) / 2
        M = model.message_passing.message(H, bmg)
        H = model.message_passing.update(M, H_0)
        
        activations_cache[f'H_{i}'] = H
        activations_cache[f'M_{i}'] = M
    
    index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
    activations_cache[f'M_{params_cache["depth"]}'] = torch.zeros(
        len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device
    ).scatter_reduce_(0, index_torch, activations_cache[f'H_{params_cache["depth"]-1}'], reduce='sum', include_self=False)
    
    activations_cache['mp'] = torch.cat((bmg.V, activations_cache[f'M_{params_cache["depth"]}']), dim=1)
    activations_cache['bmg'] = torch.cat([bmg.V[bmg.edge_index[0]], bmg.E], dim=1)
    
    return activations_cache


def model_extractor(model, data_loader):
    '''To extracts model parameters and activations.

    Parameters:
    ----------
    model (Chemprop model): a chemprop model checkpoint (.ckpt)
    dataloader (DataLoader): a chemprop data loader.

    Returns:
    --------
    params_cache (dict): a dictionary containing model parameters.
    activations_cache (dict): a dictionary containing activations.
    '''

    batch = next(iter(data_loader))
    bmg, *_ = batch
    bmg = model.message_passing.graph_transform(bmg)
    
    params_cache = extract_params(model)
    activations_cache = extract_activations(model, bmg, params_cache)
    
    return params_cache, activations_cache
