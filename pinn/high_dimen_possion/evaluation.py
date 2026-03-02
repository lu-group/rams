import torch
import numpy as np
import matplotlib.pyplot as plt
class Evaluation_HignDimPossion():
    def __init__(self, a, D, mesh_size, device=None, freq=1000, is_net_transformed=False):
        if device is None:
            # Determine the device to be used for the training
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.frequency = freq
        # Solution: u = exp(−a * ||x||^2)
        self.a = a
        self.D = D
        self.mesh_size = mesh_size
        # self.sam, self.u = get_solution(a, D, mesh_size)
        # self.sam = self.sam.detach().numpy()
        # self.u = self.u.detach().numpy()
        # self.sam = torch.tensor(self.sam).to(self.device)
        # self.u = self.u.to(self.device)
        # self.results = []
        self.is_net_transformed = is_net_transformed

    def get_u(self, net, sam):
        if self.is_net_transformed:
            raise NotImplementedError
        else:
            u = net(sam)
        return u

    def evaluate(self, net, epoch):
        pass

    # def get_results(self, net):
    #     sam_tensor = torch.tensor(get_samples(), dtype=torch.float32).to(self.device)
    #     u_pre = self.get_u(net, sam_tensor).detach().cpu().numpy()
    #     u_accu = self.u
    #     u_diff = np.abs(u_pre - u_accu)
    #     error = np.linalg.norm(u_diff) / np.linalg.norm(u_accu)
    #     return error
    #
    # def get_samples(self, D, san_num=10000):
    #     sam1 = sample_n_ball(D, 1, int(0.5 * san_num))
    #     sam2 = torch.rand(int(0.5 * san_num), D)
    #     sam = torch.cat([sam1, sam2], dim=0)
    #     return sam


    # def output(self):
    #     print("Epoch, Relative L2 Error")
    #     # Print the results in two effective digits
    #     for result in self.results:
    #         print(f"{result[0]}, {result[1]:.2e}")
    #     return self.results

def get_solution(a, D, mesh_size):
    # Creat grid points at [-0.1,0.1] ^ D, each dimension has mesh_size points
    mesh = torch.linspace(-0.1, 0.1, mesh_size)
    sams = torch.meshgrid([mesh for _ in range(D)])
    sams = torch.stack(sams, 2).view(-1, D)
    # Calculate the solution
    sams_norm = torch.norm(sams, dim=1)
    u = torch.exp(-a * sams_norm ** 2)
    return sams, u

def sample_n_ball(n, R, num_samples, is_shuffled=True):
    # Sample radius: r
    r = R * torch.rand(num_samples)

    # Sample angles: theta_1 to theta_{n-1}
    theta = torch.empty(num_samples, n - 1)
    theta[:, :-1] = torch.acos(1 - 2 * torch.rand(num_samples, n - 2))  # For angles from 0 to pi
    theta[:, -1] = 2 * torch.pi * torch.rand(num_samples)  # Last angle from 0 to 2*pi

    # Convert spherical to Cartesian coordinates
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    x = torch.empty(num_samples, n)
    x[:, 0] = r * cos_theta[:, 0]
    for i in range(1, n - 1):
        x[:, i] = r * torch.prod(sin_theta[:, :i], dim=1) * cos_theta[:, i]
    x[:, n - 1] = r * torch.prod(sin_theta, dim=1)

    if is_shuffled:
        x_shuffled = torch.empty_like(x)
        # Loop through each row and shuffle its columns
        for i in range(num_samples):
            # Generate a random permutation for the current row
            perm = torch.randperm(n)
            # Apply the permutation to the current row
            x_shuffled[i] = x[i, perm]
        x = x_shuffled
    return x


def get_results(net, D, sam_num=10000, device=None, a=10):
    if device is None:
        # Determine the device to be used for the training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = device
    sam_tensor = torch.tensor(get_samples(D=D, san_num=sam_num), dtype=torch.float32).to(device)
    net = net.to(device)
    u_pre = net(sam_tensor).cpu().detach().numpy()
    # Compute the ground truth
    sam_norm = torch.norm(sam_tensor, dim=1).view(-1, 1)
    sam_norm = sam_norm.cpu().detach().numpy()
    u_accu = np.exp(-a * sam_norm ** 2)
    u_diff = np.abs(u_pre - u_accu)
    error = (np.linalg.norm(u_diff) / np.linalg.norm(u_accu)) ** 2
    return error

def get_samples(D, san_num=10000):
    sam1 = sample_n_ball(D, 1, int(0.5 * san_num))
    sam2 = torch.rand(int(0.5 * san_num), D)
    sam = torch.cat([sam1, sam2], dim=0)
    return sam

if __name__ == '__main__':
    model_name = r"results\burgers_random_trainable_D=5.pth"
    net = torch.load(model_name)
    D = 5
    error = get_results(net, D)
    print(f"The relative L2 error is {error:.2e}.")
