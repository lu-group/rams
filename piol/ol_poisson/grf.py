import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from scipy.stats import multivariate_normal

def grf_1Dv2(x, l):
    # Covariance matrix using the Gaussian kernel
    def gaussian_kernel(x, l):
        """ Generate a Gaussian kernel matrix with correlation length l. """
        x = x[:, np.newaxis]  # Convert to column vector
        # Squared exponential kernel
        return np.exp(-0.5 * (x - x.T) ** 2 / l ** 2)

    covariance_matrix = gaussian_kernel(x, l)

    # Sample from the multivariate normal distribution
    mean = np.zeros(len(x))  # Zero mean
    random_field = multivariate_normal.rvs(mean=mean, cov=covariance_matrix)
    # max_val = np.max(np.abs(random_field))  # Find the maximum absolute value
    # random_field = 0.9 * random_field / max_val
    return x, random_field

def grf_1D_lognormal(lb, ub, num, l, mean, std):
    # Generate the points at which to sample
    x = np.linspace(lb, ub, num)

    # Calculate the parameters for the underlying normal distribution
    mean_normal = np.log(mean ** 2 / np.sqrt(std ** 2 + mean ** 2))
    std_normal = np.sqrt(np.log(1 + (std ** 2 / mean ** 2)))
    l_normal = l * np.log(1 + (std_normal ** 2 / mean_normal ** 2)) / (std_normal ** 2 / mean_normal ** 2)
    print("mean_normal: ", mean_normal)
    print("std_normal: ", std_normal)
    print("l_normal: ", l_normal)

    # Create the covariance matrix based on the exponential decay correlation function
    covariance_matrix = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            distance = np.abs(x[i] - x[j])
            covariance_matrix[i, j] = (std_normal ** 2) * np.exp(-0.5 * (distance / l_normal) ** 2)

    # Generate the correlated normal random variables
    normal_random_field = np.random.multivariate_normal(mean_normal * np.ones(num), covariance_matrix)

    # Convert the normal distribution to log-normal
    log_normal_random_field = np.exp(normal_random_field)

    return x, log_normal_random_field


def grf_2D_lognormal(node_loc, l, mean, std):
    # Extract the number of nodes
    num_nodes = node_loc.shape[0]

    # Calculate the parameters for the underlying normal distribution
    mean_normal = np.log(mean ** 2 / np.sqrt(std ** 2 + mean ** 2))
    std_normal = np.sqrt(np.log(1 + (std ** 2 / mean ** 2)))
    l_normal = l * np.log(1 + (std_normal ** 2 / mean_normal ** 2)) / (std_normal ** 2 / mean_normal ** 2)

    # Create the covariance matrix based on the exponential decay correlation function
    distance_matrix = np.linalg.norm(node_loc[:, np.newaxis, :] - node_loc[np.newaxis, :, :], axis=2)
    covariance_matrix = (std_normal ** 2) * np.exp(-0.5 * (distance_matrix / l_normal) ** 2)

    # Generate the correlated normal random variables
    normal_random_field = np.random.multivariate_normal(mean_normal * np.ones(num_nodes), covariance_matrix)

    # Convert the normal distribution to log-normal
    log_normal_random_field = np.exp(normal_random_field)

    return log_normal_random_field

def grf_2D_normal(node_loc, l, mean, std):
    # Extract the number of nodes
    if type(node_loc) == list:
        node_loc = np.array(node_loc)
    num_nodes = node_loc.shape[0]

    # Create the covariance matrix based on the exponential decay correlation function
    distance_matrix = np.linalg.norm(node_loc[:, np.newaxis, :] - node_loc[np.newaxis, :, :], axis=2)
    covariance_matrix = std ** 2 * np.exp(-0.5 * (distance_matrix / l) ** 2)

    # Generate the correlated normal random variables
    normal_random_field = np.random.multivariate_normal(mean * np.ones(num_nodes), covariance_matrix)

    return normal_random_field

def grf_1D(lb, up, num, l):
    x = np.linspace(lb, up, num)  # Discretize the interval [0, 1]

    # Covariance matrix using the Gaussian kernel
    def gaussian_kernel(x, l):
        """ Generate a Gaussian kernel matrix with correlation length l. """
        x = x[:, np.newaxis]  # Convert to column vector
        # Squared exponential kernel
        return np.exp(-0.5 * (x - x.T) ** 2 / l ** 2)

    covariance_matrix = gaussian_kernel(x, l)

    # Sample from the multivariate normal distribution
    mean = np.zeros(num)  # Zero mean
    random_field = multivariate_normal.rvs(mean=mean, cov=covariance_matrix)
    # max_val = np.max(np.abs(random_field))  # Find the maximum absolute value
    # random_field = 0.9 * random_field / max_val
    return x, random_field

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



