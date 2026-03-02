from src.old_sampler.random_sampling import random_sampling
from src.old_sampler.uniformly_sampling import uniform_sampling

def run_sampling(nodes, sample_num, sampling_method, dim, type=None, require_grad=True):
    """
    Run sampling method
    Input:
    - nodes: input nodes
    - sample_num: number of samples
    - sampling_method: sampling method (Random, Uniform)
    - dim: dimension of the input nodes
    - type: type of sampling ("line", "triangle", "rectangle", "polygon")
    """
    if sampling_method == "Random":
        return random_sampling(nodes, sample_num, dim, type, requires_grad=require_grad)
    elif sampling_method == "Uniform":
        return uniform_sampling(nodes, sample_num, dim, type, requires_grad=require_grad)
    else:
        raise ValueError("Invalid sampling method")