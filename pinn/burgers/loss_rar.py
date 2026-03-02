import torch
from src.loss.loss_base import LossBase
import torch
from src.util.gradient import gradients
from src.sampler.random_uniform_2D import sampling as sampler_random_uniform_2D
from loss_random import LossBurgers
import matplotlib.pyplot as plt

class LossBurgers_RAR(LossBurgers):
    def __init__(self, device, max_epochs, mu, initial_sampling_info, sample_update_interval,
                 updated_sample_num, kept_sample_num, is_net_transformed, is_maximized=False,
                 is_trainable=False, training_sample_info=None, is_visualized=False):
        #######################################################################
        # param net: The network to be trained with the input [x, y] and the output [w] (w: the deflection)
        # model: The model information
        # sample_num ([nD, nBC1, nBC2, nBC3, nBC4]): The number of sampling points at different regions;
        # nD is the sampling number for the domain D, nBCi is the sampling number for the boundary i
        # sampling_method: The sampling method used in the loss function
        #######################################################################
        super().__init__(device, max_epochs, mu, initial_sampling_info, sample_update_interval,
                         is_net_transformed=is_net_transformed, is_trainable=False, training_sample_info=None, is_visualized=False)
        self.updated_sample_num = updated_sample_num
        self.kept_sample_num = kept_sample_num
        self.is_maximized = is_maximized
        self.is_trainable = is_trainable
        self.training_sample_info = training_sample_info
        self.is_visualized = is_visualized
        if is_trainable and training_sample_info is not None:
            self.train_sample_iter = training_sample_info["train_sample_iter"]
            self.opt_type = training_sample_info["opt_type"]
            self.opt_lr = training_sample_info["opt_lr"]
            self.den_coef = training_sample_info["den_coef"] # Used to penalize the density of the training samples
        if self.is_maximized:
            self.sam_D_max = torch.tensor([]).to(self.device)

    def update_losses(self, net, epoch):
        if epoch % self.sample_update_interval == 0 and epoch > 0 and self.current_epoch < self.max_epochs:
            self.update_samples(net)
        # Employ the MSE loss
        loss_measurement = torch.nn.MSELoss()
        if not self.is_net_transformed:
            # For the governing pde
            self.loss_D = self.update_loss_D(net, self.sam_D, loss_measurement)
            # For the boundary conditions
            self.loss_BC1 = self.update_loss_BC(net, self.sam_BC1, loss_measurement)
            self.loss_BC2 = self.update_loss_BC(net, self.sam_BC2, loss_measurement)
            self.loss_IC = self.update_loss_IC(net, self.sam_IC, loss_measurement)
            # Add the losses to the list
            self.losses = [self.loss_D, self.loss_BC1, self.loss_BC2, self.loss_IC]
        else:
            self.loss_D = self.update_loss_D(net, self.sam_D, loss_measurement)
            self.losses = [self.loss_D]
        self.current_epoch += 1
        if self.is_maximized and self.current_epoch > self.sample_update_interval:
            loss_measurement = torch.nn.MSELoss()
            if not self.is_net_transformed:
                # For the governing pde
                self.loss_D_max = self.update_loss_D(net, self.sam_D_max, loss_measurement)
            else:
                self.loss_D_max = self.update_loss_D(net, self.sam_D_max, loss_measurement)
            self.losses.append(self.loss_D_max)

    def update_samples(self, net):
        # self.init_samplingpoints(self.initial_sampling_info)
        updated_sample_num = self.updated_sample_num
        new_sample_points = sampler_random_uniform_2D(x_min=-1, x_max=1, y_min=0, y_max=1, num_samples=updated_sample_num, requires_grad=False)
        new_sample_points = new_sample_points.to(self.device)
        residuals = self.compute_residuals(net, new_sample_points).view(-1)
        _, topk_indices = torch.topk(residuals, self.kept_sample_num)
        new_sample_points = new_sample_points[topk_indices]
        if self.is_trainable:
            new_sample_points = new_sample_points.clone().detach().requires_grad_(True)
            new_sample_points = self.training_samples(net, new_sample_points)
        if not self.is_maximized:
            self.sam_D = torch.cat((self.sam_D, new_sample_points), 0)
        else:
            # self.sam_D_max = new_sample_points #
            self.sam_D_max = torch.cat((self.sam_D_max, new_sample_points), 0)
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

    def compute_residuals(self, net, samp):
        x, t = samp[:, 0], samp[:, 1]
        x.requires_grad = True
        t.requires_grad = True
        net_input = torch.stack((x, t), 1)
        u = self.get_u(net, net_input)
        u_t = gradients(u, t)
        u_x = gradients(u, x)
        u_xx = gradients(u_x, x)
        res = u_t + u.view(u_x.shape) * u_x - self.mu * u_xx
        res = res ** 2
        return res
if __name__ == '__main__':
    pass

