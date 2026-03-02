import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from scipy.stats import multivariate_normal

def grf_2D(node, l, sam_num, std=1, L=None, is_torch=False, device=None):
    # node: a list of tuples containing all the nodes (x, y) coordinates
    # l: correlation length
    # std: standard deviation of the field
    # return: a list of random field values at each node
    is_return_L = False
    num = len(node)
    if L is None:
        # Unzip the x and y coordinates
        x = np.array([n[0] for n in node])
        y = np.array([n[1] for n in node])

        # Covariance matrix using the Gaussian kernel
        def gaussian_kernel(x, y, l, std):
            """Generate a Gaussian kernel matrix with correlation length l."""
            # Efficiently calculate the squared exponential kernel
            dx = x[:, np.newaxis] - x[np.newaxis, :]
            dy = y[:, np.newaxis] - y[np.newaxis, :]
            cov_matrix = (std ** 2) * np.exp(-0.5 * (dx ** 2 + dy ** 2) / l ** 2)
            return cov_matrix

        # Compute the covariance matrix
        covariance_matrix = gaussian_kernel(x, y, l, std)
        epsilon = 1e-8
        covariance_matrix += epsilon * np.eye(covariance_matrix.shape[0])
        # Cholesky decomposition of the covariance matrix
        L = np.linalg.cholesky(covariance_matrix)
        is_return_L = True
    if not is_torch:
        results = []
        # Transform the standard normal variables using the Cholesky factor L
        for i in range(sam_num):
            z = np.random.normal(0, 1, num)
            random_field = np.dot(L, z)
            results.append(random_field)
        results = np.array(results)
    else:
        z = torch.normal(0, 1, (sam_num, num))
        if device is not None:
            z = z.to(device)
            # L = torch.tensor(L, dtype=torch.float32).to(device)
        results = L @ z.T
        results = results.T
    if is_return_L:
        return results, L
    else:
        return results



if __name__ == '__main__':
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
    node = np.array([grid_x.flatten(), grid_y.flatten()]).T
    random_field = grf_2D(node, 0.2, 100, 1)



