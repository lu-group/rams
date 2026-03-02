import numpy as np
import torch
from src.loss.loss_base import LossBase
import torch
from torch.autograd.functional import jacobian
from src.util.gradient import gradients
from src.sampler.random_uniform_2D import sampling as sampler_random_uniform_2D, quasirandom_sampling
from src.sampler.random_uniform_2D import sampler_random_uniform_high_dimen
from che_poly import Chebyshev_poly

import matplotlib.pyplot as plt

class Loss_Random(LossBase):
    def __init__(self, device, max_epochs, initial_sampling_info, sample_update_interval,
                 is_net_transformed=True, is_trainable=False, training_sample_info=None, is_visualized=False):
        #######################################################################
        # param net: The network to be trained with the input [x, y] and the output [w] (w: the deflection)
        # model: The model information
        # sample_num ([nD, nBC1, nBC2, nBC3, nBC4]): The number of sampling points at different regions;
        # nD is the sampling number for the domain D, nBCi is the sampling number for the boundary i
        # sampling_method: The sampling method used in the loss function
        #######################################################################
        super().__init__()
        # self.net = net # Input: [x, t]; Output: [u]
        self.max_epochs = max_epochs
        self.is_trainable = is_trainable
        self.current_epoch = 0
        self.sample_update_interval = sample_update_interval
        if device is None:
            # Determine the device to be used for the training
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.initial_sampling_info = initial_sampling_info
        self.is_net_transformed = is_net_transformed
        self.training_sample_info = training_sample_info
        if is_trainable and training_sample_info is not None:
            self.train_sample_interval = training_sample_info["train_sample_interval"]
            self.train_sample_iter = training_sample_info["train_sample_iter"]
            self.sample_ratio = training_sample_info["sample_ratio"]
            self.kept_ratio = training_sample_info["kept_ratio"]
            self.opt_type = training_sample_info["opt_type"]
            self.opt_lr = training_sample_info["opt_lr"]
        self.is_visualized = is_visualized
        self.init_samplingpoints(self.initial_sampling_info)
        self.cal_chebyshev_poly = Chebyshev_poly(x_span=[0, 1], n=100, device=self.device)
    def get_u(self, net, branch_input, trunk_input):
        is_net_transformed = self.is_net_transformed
        if is_net_transformed:
            raise NotImplementedError
        else:
            u = net.forward_branch_trunk_fixed(branch_input, trunk_input)
        return u
    def update_losses(self, net, epoch):
        if self.is_trainable and epoch % self.train_sample_interval == 0 and epoch > 0 and self.current_epoch <= self.max_epochs:
            self.update_samples(net)
            if self.is_visualized:
                self.visual_sam(epoch)
        # Employ the MSE loss
        loss_measurement = torch.nn.MSELoss()
        if not self.is_net_transformed:
            self.loss_D = self.update_loss_D(net, self.branch_input, self.trunk_input, loss_measurement)
            self.loss_BC = self.update_loss_BC(net, self.branch_input, self.trunk_input_x0, loss_measurement)
            self.losses = [self.loss_D, self.loss_BC]
        else:
            raise NotImplementedError
        self.current_epoch += 1
    # Loss term corresponding to the governing PDE in the domain D


    def update_loss_D(self, net, branch_input, trunk_input, loss_measurement):
        trunk_output = net.trunk_net(trunk_input)
        dx = 1e-5
        dt = net.trunk_net(trunk_input + dx) - trunk_output
        dt_dx = dt.transpose(0, 1) / dx
        # dt_dx = jacobian(net.trunk_net, trunk_input, create_graph=True)
        branch_output = net.branch_net(branch_input)
        du_dx = torch.matmul(branch_output, dt_dx)
        f = self.cal_chebyshev_poly.get_results(branch_input)
        norm_branch_input = torch.norm(branch_input - 0.5, dim=1) ** 2
        exp_coef = torch.diag(torch.exp(-6 * norm_branch_input))
        res = du_dx - exp_coef @ f
        loss = loss_measurement(res, torch.zeros_like(res))
        return loss
    # Loss term corresponding to the boundary conditions
    def update_loss_BC(self, net, branch_input, trunk_input, loss_measurement):
        u = self.get_u(net, branch_input, trunk_input)
        loss = loss_measurement(u, torch.zeros_like(u))
        return loss
    # boundary condition
    def init_samplingpoints(self, initial_sampling_info):
        sam_num = initial_sampling_info["n"]
        requires_grad = False
        self.branch_input = sampler_random_uniform_high_dimen(x_min=-1, x_max=1, dim=8, num_samples=sam_num, requires_grad=requires_grad)
        self.branch_input = self.branch_input.to(self.device)
        self.trunk_input = torch.linspace(0, 1, 100).view(-1, 1).to(self.device) #.requires_grad_(True)
        self.trunk_input_x0 = torch.zeros(1, 1).to(self.device)
    def update_samples(self, net):
        # Randomly divided sam_D into two parts
        self.branch_input = self.branch_input.detach()
        training_sample_num = int(self.branch_input.shape[0] * self.sample_ratio)
        kept_sample_num = int(self.kept_ratio * self.branch_input.shape[0])
        unkept_sample_num = self.branch_input.shape[0] - kept_sample_num
        unkept_samples = self.branch_input[:unkept_sample_num]
        kept_samples = self.branch_input[unkept_sample_num:]

        sam1 = unkept_samples[:training_sample_num]
        sam2 = unkept_samples[training_sample_num:]
        new_sam = sam1.clone().detach().requires_grad_(True)
        new_sam = self.training_samples(net, new_sam)
        new_sam = torch.concatenate((sam2, new_sam), 0)
        new_sam = torch.cat((new_sam, kept_samples), 0)
        self.branch_input.data = new_sam.detach()  # Efficiently update the original data

    def training_samples(self, net, new_sam):
        if self.opt_type == "Adam":
            opt = torch.optim.Adam(params=[new_sam], lr=self.opt_lr)
        elif self.opt_type == "SGD":
            opt = torch.optim.SGD(params=[new_sam], lr=self.opt_lr)
        loss_func = torch.nn.MSELoss()
        net.eval()
        for _ in range(self.train_sample_iter):
            opt.zero_grad()
            loss = -1 * self.update_loss_D_for_training_sample(net, branch_input=new_sam, trunk_input=self.trunk_input, loss_measurement=loss_func)
            loss.backward()
            opt.step()
            new_sam.requires_grad_(False)
            new_sam.clamp_(-1, 1)  # projection
            new_sam.requires_grad_(True)
        net.train()
        new_sam.requires_grad_(False)
        new_sam.clamp_(min=-1 * torch.ones(8).to(self.device),
                       max=torch.ones(8).to(self.device))
        return new_sam

    def update_loss_D_for_training_sample(self, net, branch_input, trunk_input, loss_measurement):
        return self.update_loss_D(net, branch_input, trunk_input, loss_measurement)
    def compute_residuals(self, net, branch_input, trunk_input):
        # x, t = samp[:, 0], samp[:, 1]
        # x.requires_grad = True
        # t.requires_grad = True
        # net_input = torch.stack((x, t), 1)
        # u = self.get_u(net, net_input)
        # u_t = gradients(u, t)
        # u_x = gradients(u, x)
        # u_xx = gradients(u_x, x)
        # cond = torch.exp(-t) * (-torch.sin(torch.pi * x) + torch.pi ** 2 * torch.sin(torch.pi * x))
        # res = (u_t - u_xx - cond)
        # res = res ** 2
        trunk_output = net.trunk_net(trunk_input)
        dx = 1e-5
        dt = net.trunk_net(trunk_input + dx) - trunk_output
        dt_dx = dt.transpose(0, 1) / dx
        # dt_dx = jacobian(net.trunk_net, trunk_input, create_graph=True)
        branch_output = net.branch_net(branch_input)
        du_dx = torch.matmul(branch_output, dt_dx)
        f = self.cal_chebyshev_poly.get_results(branch_input)
        norm_branch_input = torch.norm(branch_input - 0.5, dim=1) ** 2
        exp_coef = torch.diag(torch.exp(-6 * norm_branch_input))
        res = du_dx - exp_coef @ f
        # For each row, calculate the L2 norm
        res_norm = torch.norm(res, dim=1)
        return res_norm

    def visual_sam(self, epoch, vis_dim=[1,2]):
        sample = self.branch_input.cpu().detach().numpy()
        sample_x = sample[:, vis_dim[0]]
        sample_y = sample[:, vis_dim[1]]
        sample = np.stack((sample_x, sample_y), axis=1)
        plt.scatter(sample[:, 0], sample[:, 1], alpha=0.1, color='blue')
        plt.xlabel("a"+str(vis_dim[0]))
        plt.ylabel("a"+str(vis_dim[1]))
        plt.grid(True)
        plt.text(0.02, 0.02, "Epoch: " + str(epoch), horizontalalignment='center', verticalalignment='bottom',
                 bbox=dict(facecolor='white', alpha=0.5))
        plt.show()
if __name__ == '__main__':
    pass

