from loss_random import Loss_Random
import matplotlib.pyplot as plt
from src.sampler.random_uniform_2D import sampler_random_uniform_high_dimen
import numpy as np
import torch
import grf

class Loss_RAR(Loss_Random):
    def __init__(self, device, max_epochs, initial_sampling_info, sample_update_interval, n_x, n_y,
                 updated_sample_num, kept_sample_num, is_net_transformed, is_maximized=False, proj_l=0.3, k_min=0.5, b=0.3, k_max=1,
                 is_trainable=False, training_sample_info=None, is_visualized=False):
        #######################################################################
        # param net: The network to be trained with the input [x, y] and the output [w] (w: the deflection)
        # model: The model information
        # sample_num ([nD, nBC1, nBC2, nBC3, nBC4]): The number of sampling points at different regions;
        # nD is the sampling number for the domain D, nBCi is the sampling number for the boundary i
        # sampling_method: The sampling method used in the loss function
        #######################################################################
        super().__init__(device, max_epochs, initial_sampling_info, sample_update_interval, proj_l=proj_l, n_x=n_x, n_y=n_y, k_min=k_min, b=b, k_max=k_max,
                         is_net_transformed=is_net_transformed, is_trainable=False, training_sample_info=None, is_visualized=False)
        self.updated_sample_num = updated_sample_num
        self.kept_sample_num = kept_sample_num
        self.is_maximized = is_maximized
        self.is_trainable = is_trainable
        self.training_sample_info = training_sample_info
        self.is_visualized = is_visualized
        self.train_sample_interval = training_sample_info["train_sample_interval"]
        if is_trainable and training_sample_info is not None:
            self.train_sample_interval = training_sample_info["train_sample_interval"]
            self.train_sample_iter = training_sample_info["train_sample_iter"]
            self.opt_type = training_sample_info["opt_type"]
            self.opt_lr = training_sample_info["opt_lr"]

    def update_losses(self, net, epoch):
        # if epoch % 2500 == 0 and epoch < self.max_epochs:
        #     self.update_trunk_input(self.branch_input)
        if self.is_trainable and epoch % self.train_sample_interval == 0 and epoch > 0 and self.current_epoch <= self.max_epochs:
            self.update_samples(net)
        elif epoch % 2500 == 0 and epoch < self.max_epochs:
            self.update_trunk_input(self.branch_input)
        # Employ the MSE loss
        loss_measurement = torch.nn.MSELoss()
        if not self.is_net_transformed:
            self.loss_D = self.update_loss_D(net, self.branch_input, self.trunk_input, loss_measurement)
            self.loss_BC = self.update_loss_BC(net, self.branch_input, self.trunk_input_BC, loss_measurement)
            self.losses = [self.loss_D, self.loss_BC]
        else:
            self.loss_D = self.update_loss_D(net, self.branch_input, self.trunk_input, loss_measurement)
            self.losses = [self.loss_D]
        self.current_epoch += 1
    # Loss term corresponding to the governing PDE in the domain D

    def update_samples(self, net):
        updated_sample_num = self.updated_sample_num
        n_x = self.n_x
        n_y = self.n_y
        # branch_input = []
        # grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, n_x), np.linspace(-1, 1, n_y))
        node = self.branch_sample_node.detach().cpu().numpy()
        branch_input = grf.grf_2D(node, l=self.proj_l, sam_num=updated_sample_num, L=self.L, is_torch=True, device=self.device)
        branch_input = torch.tensor(branch_input, dtype=torch.float32).to(self.device)
        branch_input_norm = torch.norm(branch_input, dim=1) / (n_x * n_y) ** 0.5 / 2
        branch_input = branch_input / branch_input_norm[:, None]
        new_branch_input = torch.tensor(branch_input, dtype=torch.float32).to(self.device)
        residuals = self.compute_residuals(net, new_branch_input, self.trunk_input).view(-1)
        _, topk_indices = torch.topk(residuals, self.kept_sample_num)
        new_branch_input = new_branch_input[topk_indices]
        if self.is_trainable:
            new_branch_input = new_branch_input.clone().detach().requires_grad_(True)
            new_branch_input = self.training_samples(net, new_branch_input)
        self.branch_input = torch.cat((self.branch_input, new_branch_input), 0)
        self.f = self.get_f(self.branch_input, self.branch_sample_node, self.trunk_input)
        if self.is_visualized:
            self.visualize_samples()

    def visualize_samples(self):
        sam_D = self.sam_D.detach().cpu().numpy()
        init_sample_num = self.initial_sampling_info["nD"]
        init_samples = sam_D[:init_sample_num]
        updated_sample_num = sam_D.shape[0] - init_sample_num
        if not self.is_maximized:
            updated_samples = sam_D[init_sample_num:]
        else:
            updated_samples = self.sam_D_max.detach().cpu().numpy()
        plt.scatter(init_samples[:, 0], init_samples[:, 1], alpha=0.1, color='blue')
        plt.scatter(updated_samples[:, 0], updated_samples[:, 1], alpha=0.1, color='red')
        plt.xlabel("x")
        plt.ylabel("t")
        plt.grid(True)
        plt.text(0.45, 0.02, "Epoch: " + str(self.current_epoch), horizontalalignment='center', verticalalignment='bottom',
                 bbox=dict(facecolor='white', alpha=0.5))
        plt.show()
if __name__ == '__main__':
    pass

