import torch
from src.loss.loss_base import LossBase
import torch
from src.util.gradient import gradients
from src.sampler.random_uniform_2D import sampling as sampler_random_uniform_2D
from loss_random import Loss_Random
import matplotlib.pyplot as plt
from src.sampler.random_uniform_2D import sampler_random_uniform_high_dimen

class Loss_RAR(Loss_Random):
    def __init__(self, device, max_epochs, initial_sampling_info, sample_update_interval,
                 updated_sample_num, kept_sample_num, is_net_transformed, is_maximized=False,
                 is_trainable=False, training_sample_info=None, is_visualized=False):
        #######################################################################
        # param net: The network to be trained with the input [x, y] and the output [w] (w: the deflection)
        # model: The model information
        # sample_num ([nD, nBC1, nBC2, nBC3, nBC4]): The number of sampling points at different regions;
        # nD is the sampling number for the domain D, nBCi is the sampling number for the boundary i
        # sampling_method: The sampling method used in the loss function
        #######################################################################
        super().__init__(device, max_epochs, initial_sampling_info, sample_update_interval,
                         is_net_transformed=is_net_transformed, is_trainable=False, training_sample_info=None, is_visualized=False)
        self.updated_sample_num = updated_sample_num
        self.kept_sample_num = kept_sample_num
        self.is_maximized = is_maximized
        self.is_trainable = is_trainable
        self.training_sample_info = training_sample_info
        self.is_visualized = is_visualized
        if is_trainable and training_sample_info is not None:
            self.train_sample_interval = training_sample_info["train_sample_interval"]
            self.train_sample_iter = training_sample_info["train_sample_iter"]
            self.opt_type = training_sample_info["opt_type"]
            self.opt_lr = training_sample_info["opt_lr"]

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
    def update_samples(self, net):
        updated_sample_num = self.updated_sample_num
        new_branch_input = sampler_random_uniform_high_dimen(x_min=-1, x_max=1, dim=8, num_samples=updated_sample_num,
                                                         requires_grad=False).to(self.device)
        residuals = self.compute_residuals(net, new_branch_input, self.trunk_input).view(-1)
        _, topk_indices = torch.topk(residuals, self.kept_sample_num)
        new_branch_input = new_branch_input[topk_indices]
        if self.is_trainable:
            new_branch_input = new_branch_input.clone().detach().requires_grad_(True)
            new_branch_input = self.training_samples(net, new_branch_input)
            self.branch_input = torch.cat((self.branch_input, new_branch_input), 0)
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

