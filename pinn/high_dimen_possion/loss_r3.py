import torch
import numpy as np
from src.sampler.random_uniform_2D import sampling as sampler_random_uniform_2D
from loss_rar import Loss_RAR
class Loss_R3(Loss_RAR):
    def __init__(self, a, r, device, max_epochs, initial_sampling_info,sample_update_interval,
                 is_net_transformed, is_maximized=False, is_trainable=False, training_sample_info=None, is_visualized=False):
        #######################################################################
        # param net: The network to be trained with the input [x, y] and the output [w] (w: the deflection)
        # model: The model information
        # sample_num ([nD, nBC1, nBC2, nBC3, nBC4]): The number of sampling points at different regions;
        # nD is the sampling number for the domain D, nBCi is the sampling number for the boundary i
        # sampling_method: The sampling method used in the loss function
        #######################################################################
        super().__init__(a=a, r=r, device=device, max_epochs=max_epochs, initial_sampling_info=initial_sampling_info, sample_update_interval=sample_update_interval,
                 updated_sample_num=None, kept_sample_num=None, is_net_transformed=is_net_transformed, is_maximized=is_maximized,
                 is_trainable=is_trainable, training_sample_info=training_sample_info, is_visualized=is_visualized)
    def update_samples(self, net):
        if self.is_maximized:
            raise Exception("The maximization operation is not supported in the R3 loss function")
        sam_D_residuals = self.compute_residuals(net, self.sam_D).view(-1)
        avg_residual = torch.mean(sam_D_residuals)
        # Delete the sampling points with residuals smaller than the average
        sam_idx = torch.where(sam_D_residuals > avg_residual)
        self.sam_D = self.sam_D[sam_idx]
        # Shuffle the sampling points
        sam_idx = torch.randperm(self.sam_D.shape[0])
        self.sam_D = self.sam_D[sam_idx]
        if self.is_trainable:
            # new_sample_points = self.training_sample_points(net, new_sample_points)
            # trainable_sample_ratio = self.training_sample_info["sample_ratio"]
            # training_sample_num = int(self.sam_D.shape[0] * trainable_sample_ratio)
            # trainable_sample = self.sam_D[:training_sample_num].detach().clone().requires_grad_(True)
            # self.sam_D[:training_sample_num] = self.training_samples(net, trainable_sample)

            trainable_sample_ratio = self.training_sample_info["sample_ratio"]
            training_sample_num = int(self.sam_D.shape[0] * trainable_sample_ratio)
            residuals = self.compute_residuals(net, self.sam_D).view(-1)
            self.sam_D = self.sam_D.detach().cpu().numpy()
            residuals = residuals.detach().cpu().numpy()
            residuals = np.abs(residuals) / np.sum(np.abs(residuals))
            new_sample_points_idx = np.random.choice(a=len(residuals), size=training_sample_num, replace=False,
                                                     p=residuals)
            sam_D1 = torch.tensor(self.sam_D[new_sample_points_idx], dtype=torch.float32).to(self.device)
            sam_D2 = self.sam_D[np.setdiff1d(np.arange(len(self.sam_D)), new_sample_points_idx)]
            sam_D2 = torch.tensor(sam_D2, dtype=torch.float32).to(self.device)
            new_sam_D = sam_D1.clone().detach().requires_grad_(True)
            new_sam_D = self.training_samples(net, new_sam_D)
            self.sam_D = torch.cat((new_sam_D, sam_D2), 0)
        resampling_num = self.initial_sampling_info["nD"] - self.sam_D.shape[0]
        new_sample_points = sampler_random_uniform_2D(x_min=-1, x_max=1, y_min=0, y_max=1, num_samples=resampling_num, requires_grad=False)
        new_sample_points = new_sample_points.to(self.device)
        self.sam_D = torch.cat((self.sam_D, new_sample_points), 0)


if __name__ == '__main__':
    pass

