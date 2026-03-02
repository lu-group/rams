import torch
import numpy as np
from scipy.stats import qmc
def sampling(x_min, x_max, y_min, y_max, num_samples, requires_grad=True):
    """
    Randomly sample points from a 2D domain
    Input:
    - x_min: minimum value of x
    - x_max: maximum value of x
    - y_min: minimum value of y
    - y_max: maximum value of y
    - num_samples: number of samples
    """
    x = torch.rand(num_samples, 1) * (x_max - x_min) + x_min
    y = torch.rand(num_samples, 1) * (y_max - y_min) + y_min
    samples = torch.cat((x, y), 1)
    samples.requires_grad = requires_grad
    return samples

def sampler_random_uniform_high_dimen(x_min, x_max, num_samples, dim, requires_grad=True):
    """
    Randomly sample points from a 2D domain
    Input:
    - x_min: minimum value of x
    - x_max: maximum value of x
    - num_samples: number of samples
    """
    samples = torch.rand(num_samples, dim) * (x_max - x_min) + x_min
    samples.requires_grad = requires_grad
    return samples

def quasirandom_sampling(x_min, x_max, y_min, y_max, num_samples, sampler_name, requires_grad=True):
    bounds = np.array([[x_min, x_max], [y_min, y_max]])

    if sampler_name == "lhs":
        # Use Latin Hypercube Sampling to generate 1000 points
        lhs_sampler = qmc.LatinHypercube(d=2)
        sample = lhs_sampler.random(n=num_samples)
    elif sampler_name == "halton":
        halton_sampler = qmc.Halton(d=2)
        sample = halton_sampler.random(n=num_samples)
    elif sampler_name == "sobol":
        # m = int(np.ceil(np.log2(num_samples))) - 1
        sobol_sampler = qmc.Sobol(d=2)
        sample = sobol_sampler.random(n=num_samples)
        # new_sample_num = num_samples - sample0.shape[0]
        # m = int(np.ceil(np.log2(new_sample_num))) - 1
        # sample0 = np.concatenate((sample0, sobol_sampler.random_base2(m=m)))
        # new_sample_num = num_samples - sample0.shape[0]
        # m = int(np.ceil(np.log2(new_sample_num))) - 1
        # sample0 = np.concatenate((sample0, sobol_sampler.random_base2(m=m)))
        # sample = sample0

    else:
        raise ValueError("Invalid sampler name")
    # Scale the samples to the given bounds
    scaled_sample = qmc.scale(sample, bounds[:, 0], bounds[:, 1])
    # Transform the numpy array to a torch tensor
    scaled_sample = torch.tensor(scaled_sample, dtype=torch.float32, requires_grad=requires_grad)
    return scaled_sample
