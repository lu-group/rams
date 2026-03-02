import torch
import numpy as np
from gradient_compute import gradients
import matplotlib.pyplot as plt
import torch.nn as nn

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

class Loss_Random():
    def __init__(self, a, D, device, max_epochs, initial_sampling_info, sample_update_interval, is_net_transformed=True,
                 is_trainable=False, training_sample_info=None, is_visualized=False, samBC_update_interval=None):
        #######################################################################
        # param net: The network to be trained with the input [x, y] and the output [w] (w: the deflection);
        # model: The model information;
        # sample_num ([nD, nBC1, nBC2, nBC3, nBC4]): The number of sampling points at different regions;
        # nD is the sampling number for the domain D, nBCi is the sampling number for the boundary i;
        # sampling_method: The sampling method used in the loss function;
        #######################################################################
        # super().__init__()
        self.a = a
        self.D = D
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
            self.den_coef = training_sample_info["den_coef"]  # Used to penalize the density of the training samples
        self.is_visualized = is_visualized
        self.init_samplingpoints(self.initial_sampling_info)
        self.samBC_update_interval = samBC_update_interval
        self.loss_value = 0

    def get_u(self, net, sam):
        is_net_transformed = self.is_net_transformed
        if is_net_transformed:
            raise NotImplementedError
        else:
            u = net(sam)
        return u

    def update_losses(self, net, epoch):
        if self.samBC_update_interval is not None and epoch % self.samBC_update_interval == 0 and epoch > 0:
            self.sam_BC = self.get_samBC(D=self.D, sam_num=self.initial_sampling_info["nBC"])

        if self.is_trainable and epoch % self.train_sample_interval == 0 and self.current_epoch <= self.max_epochs: # and  epoch > 0:
            self.update_samples(net)
            if self.is_visualized:
                self.visual_sam(epoch)

        if epoch == 0 and self.is_visualized:
            self.visual_sam(epoch)
        # Employ the MSE loss
        loss_measurement = torch.nn.MSELoss()
        if not self.is_net_transformed:
            # For the governing pde
            self.loss_D = self.update_loss_D(net, self.sam_D, loss_measurement)
            # For the boundary conditions
            self.loss_BC = self.update_loss_BC(net, self.sam_BC, loss_measurement)
            # Add the losses to the list
            self.losses = [self.loss_D, self.loss_BC]
        else:
            raise NotImplementedError
        self.current_epoch += 1
        self.loss_value = self.loss_D.item()

    # Loss term corresponding to the governing PDE in the domain D
    def update_loss_D(self, net, sam_D, loss_measurement):
        x_list = []
        for i in range(self.D):
            x_list.append(sam_D[:, i].requires_grad_(True))
        net_input = torch.stack(x_list, 1)
        u = self.get_u(net, net_input)
        u_xx_list = []
        for i in range(self.D):
            tu_x = gradients(u, x_list[i])
            u_xx_list.append(gradients(tu_x, x_list[i]))
        a = self.a
        x_norm = torch.norm(net_input, dim=1)
        tu = torch.exp(-a * x_norm ** 2)
        res_list = []
        for i in range(self.D):
            res = (-2 * a + 4 * a ** 2 * x_list[i] ** 2) * tu - u_xx_list[i]
            res_list.append(res)
        res = torch.sum(torch.stack(res_list, 1), 1)
        loss_D = loss_measurement(res, torch.zeros_like(res))
        return loss_D

    # Loss term corresponding to the boundary conditions
    def update_loss_BC(self, net, sam_BC, loss_measurement):
        u = self.get_u(net, sam_BC)
        a = self.a
        x_norm = torch.norm(sam_BC, dim=1)
        u_acc = torch.exp(-a * x_norm ** 2)
        loss_BC = loss_measurement(u, u_acc)
        return loss_BC

    def init_samplingpoints(self, initial_sampling_info):
        sam_num_D = initial_sampling_info["nD"]
        sam_num_BC = initial_sampling_info["nBC"]
        sampler_type = initial_sampling_info["SamplingMethod"]
        if sampler_type == "Uniform":
            requires_grad = False
            self.sam_D = sampler_random_uniform_high_dimen(x_min=-1, x_max=1, dim=self.D,
                                                           num_samples=sam_num_D, requires_grad=requires_grad)
            self.sam_D = self.sam_D.to(self.device)
        else:
            raise NotImplementedError
        self.sam_BC = self.get_samBC(D=self.D, sam_num=sam_num_BC)

    def get_samBC(self, D, sam_num):
        sam_BC = sampler_random_uniform_high_dimen(x_min=-1, x_max=1, dim=D, num_samples=sam_num, requires_grad=False)
        # Pick the column idx for the max value in each row
        max_idx = torch.argmax(torch.abs(sam_BC), dim=1).view(-1).tolist()
        # Set the max value to 1 and the others to 0
        row_idx = torch.arange(sam_num).tolist()
        sam_BC[row_idx[:int(0.5 * sam_num)], max_idx[:int(0.5 * sam_num)]] = 1
        sam_BC[row_idx[int(0.5 * sam_num):], max_idx[int(0.5 * sam_num):]] = -1
        # sam_BC = torch.where(sam_BC < 0, torch.tensor(-1.0), torch.tensor(1.0))
        return torch.tensor(sam_BC, dtype=torch.float32).to(self.device)

    def update_samples(self, net):
        # Randomly divided sam_D into two parts
        self.sam_D = self.sam_D.detach()
        training_sample_num = int(self.sam_D.shape[0] * self.sample_ratio)
        kept_sample_num = int(self.kept_ratio * self.sam_D.shape[0])
        unkept_sample_num = self.sam_D.shape[0] - kept_sample_num
        unkept_samples = self.sam_D[:unkept_sample_num]
        kept_samples = self.sam_D[unkept_sample_num:]
        residuals = self.compute_residuals(net, unkept_samples).view(-1)
        unkept_samples = unkept_samples.detach().cpu().numpy()
        residuals = residuals.detach().cpu().numpy()
        residuals = np.abs(residuals) / np.sum(np.abs(residuals))
        new_sample_points_idx = np.random.choice(a=len(residuals), size=training_sample_num, replace=False, p=residuals)
        sam_D1 = torch.tensor(unkept_samples[new_sample_points_idx], dtype=torch.float32).to(self.device)
        sam_D2 = unkept_samples[np.setdiff1d(np.arange(len(unkept_samples)), new_sample_points_idx)]
        sam_D2 = torch.tensor(sam_D2, dtype=torch.float32).to(self.device)
        new_sam_D = sam_D1.clone().detach().requires_grad_(True)
        new_sam_D = self.training_samples(net, new_sam_D)
        new_sam_D = torch.cat((sam_D2, new_sam_D), 0)

        new_sam_D = torch.cat((new_sam_D, kept_samples), 0)
        self.sam_D.data = new_sam_D.detach()  # Efficiently update the original data

    def training_samples(self, net, new_sam_D):
        if self.opt_type == "Adam":
            opt = torch.optim.Adam(params=[new_sam_D], lr=self.opt_lr)
        elif self.opt_type == "SGD":
            opt = torch.optim.SGD(params=[new_sam_D], lr=self.opt_lr)
        loss_func = torch.nn.MSELoss()
        net.eval()
        max_iter = self.train_sample_iter
        if self.loss_value > 1e-1:
            max_iter = int(max_iter/5)
        elif self.loss_value > 1e-2:
            max_iter = int(max_iter/3)
        for _ in range(max_iter):
            opt.zero_grad()
            loss = -1 * self.update_loss_D_for_training_sample(net, new_sam_D, loss_func)
            if self.den_coef > 0:
                mean_new_sam_D = torch.mean(new_sam_D, dim=0)
                variance_sam_D = torch.mean(torch.sum((new_sam_D - mean_new_sam_D) ** 2, dim=1))
                normed_variance = self.den_coef * torch.log(1 + variance_sam_D)
                print("Epoch: ", self.current_epoch, "Loss: ", loss.item(), "Det(Sigma): ", normed_variance.item())
                loss -= normed_variance
            loss.backward()
            opt.step()
            new_sam_D.requires_grad_(False)
            new_sam_D.clamp_(-1, 1)  # projection
            new_sam_D.requires_grad_(True)
        net.train()
        new_sam_D.requires_grad_(False)
        new_sam_D.clamp_(min=-1 * torch.ones(self.D, device=self.device), max=torch.ones(self.D, device=self.device))
        return new_sam_D

    def update_loss_D_for_training_sample(self, net, sam_D, loss_measurement):
        x_list = []
        for i in range(self.D):
            x_list.append(sam_D[:, i])
        net_input = torch.stack(x_list, 1)
        u = self.get_u(net, net_input)
        u_xx_list = []
        for i in range(self.D):
            tu_x = gradients(u, x_list[i])
            u_xx_list.append(gradients(tu_x, x_list[i]))
        a = self.a
        x_norm = torch.norm(net_input, dim=1)
        tu = torch.exp(-a * x_norm ** 2)
        res_list = []
        for i in range(self.D):
            res = (-2 * a + 4 * a ** 2 * x_list[i] ** 2) * tu - u_xx_list[i]
            res_list.append(res)
        res = torch.sum(torch.stack(res_list, 1), 1)
        loss_D = loss_measurement(res, torch.zeros_like(res))
        return loss_D

    def compute_residuals(self, net, samp):
        x_list = []
        for i in range(self.D):
            x_list.append(samp[:, i].requires_grad_(True))
        net_input = torch.stack(x_list, 1)
        u = self.get_u(net, net_input)
        u_xx_list = []
        for i in range(self.D):
            tu_x = gradients(u, x_list[i])
            u_xx_list.append(gradients(tu_x, x_list[i]))
        a = self.a
        x_norm = torch.norm(net_input, dim=1)
        tu = torch.exp(-a * x_norm ** 2)
        res_list = []
        for i in range(self.D):
            res = (-2 * a + 4 * a ** 2 * x_list[i] ** 2) * tu - u_xx_list[i]
            res_list.append(res)
        res = torch.sum(torch.stack(res_list, 1), 1)
        res = res ** 2
        return res

    def visual_sam(self, epoch):
        raise NotImplementedError
        sample = self.sam_D.cpu().detach().numpy()
        if self.is_trainable and self.current_epoch != 0:
            training_sample_num = int(self.sam_D.shape[0] * self.sample_ratio)
            sample_updated = sample[:training_sample_num]
            sample_not_updated = sample[training_sample_num:]
            plt.scatter(sample_not_updated[:, 0], sample_not_updated[:, 1], alpha=0.1, color='blue')
            plt.scatter(sample_updated[:, 0], sample_updated[:, 1], alpha=0.1, color='red')
            plt.xlabel("x")
            plt.ylabel("t")
            plt.grid(True)
            plt.text(0.45, 0.02, "Epoch: " + str(epoch), horizontalalignment='center', verticalalignment='bottom',
                    bbox=dict(facecolor='white', alpha=0.5))
            plt.show()
        else:
            sample = self.sam_D.cpu().detach().numpy()
            plt.scatter(sample[:, 0], sample[:, 1], alpha=0.1, color='blue')
            plt.xlabel("x")
            plt.ylabel("t")
            plt.grid(True)
            plt.text(0.45, 0.02, "Epoch: " + str(epoch), horizontalalignment='center', verticalalignment='bottom',
                     bbox=dict(facecolor='white', alpha=0.5))
            plt.show()

if __name__ == '__main__':
    pass

