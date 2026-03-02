import torch
import numpy as np
from src.loss.loss_base import LossBase
import torch
import matplotlib.pyplot as plt
from src.util.gradient import gradients
from src.sampler.random_uniform_2D import sampling as sampler_random_uniform_2D
from loss_random import LossBurgers
from loss_rar import LossBurgers_RAR
class LossBurgers_RAD(LossBurgers_RAR):
    def __init__(self, device, max_epochs, mu, initial_sampling_info,sample_update_interval, updated_sample_num, kept_sample_num,
                 is_net_transformed, is_maximized=False, is_trainable=False, training_sample_info=None, is_visualized=False):
        #######################################################################
        # param net: The network to be trained with the input [x, y] and the output [w] (w: the deflection)
        # model: The model information
        # sample_num ([nD, nBC1, nBC2, nBC3, nBC4]): The number of sampling points at different regions;
        # nD is the sampling number for the domain D, nBCi is the sampling number for the boundary i
        # sampling_method: The sampling method used in the loss function
        #######################################################################
        super().__init__(device=device, max_epochs=max_epochs, mu=mu, initial_sampling_info=initial_sampling_info, sample_update_interval=sample_update_interval,
                 updated_sample_num=updated_sample_num, kept_sample_num=kept_sample_num, is_net_transformed=is_net_transformed, is_maximized=is_maximized,
                 is_trainable=is_trainable, training_sample_info=training_sample_info, is_visualized=is_visualized)
    def update_samples(self, net):
        # self.init_samplingpoints(self.initial_sampling_info)
        updated_sample_num = self.updated_sample_num
        new_sample_points = sampler_random_uniform_2D(x_min=-1, x_max=1, y_min=0, y_max=1,
                                                      num_samples=updated_sample_num, requires_grad=False)
        new_sample_points = new_sample_points.to(self.device)
        residuals = self.compute_residuals(net, new_sample_points).view(-1)
        new_sample_points = new_sample_points.detach().cpu().numpy()
        residuals = residuals.detach().cpu().numpy()
        residuals = np.abs(residuals) / np.sum(np.abs(residuals))
        new_sample_points_idx = np.random.choice(a=len(new_sample_points), size=self.kept_sample_num, replace=False, p=residuals)
        new_sample_points = torch.tensor(new_sample_points[new_sample_points_idx], dtype=torch.float32).to(self.device)
        if self.is_trainable:
            new_sample_points = new_sample_points.clone().detach().requires_grad_(True)
            new_sample_points = self.training_samples(net, new_sample_points)
        if not self.is_maximized:
            self.sam_D = torch.cat((self.sam_D, new_sample_points), 0)
        else:
            self.sam_D_max = torch.cat((self.sam_D_max, new_sample_points), 0)
        if self.is_visualized:
            self.visualize_samples()


if __name__ == '__main__':
    pass

