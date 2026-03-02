import torch
from src.loss.loss_base import LossBase
import torch
from src.util.gradient import gradients
from src.sampler.random_uniform_2D import sampling as sampler_random_uniform_2D, quasirandom_sampling
import matplotlib.pyplot as plt
import numpy as np

class LossBurgers(LossBase):
    def __init__(self, device, max_epochs, mu, initial_sampling_info, sample_update_interval,
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
        self.mu = mu
        self.sample_update_interval = sample_update_interval
        if device is None:
            # Determine the device to be used for the training
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        # self.net.to(self.device)
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
    def get_u(self, net, sam):
        is_net_transformed = self.is_net_transformed
        if is_net_transformed:
            output = net(sam)
            u = (-torch.sin(torch.pi * sam[:, 0].view(-1, 1))
                 + (1 - (sam[:, 0] ** 2).view(-1, 1)) * sam[:, 1].view(-1, 1) * output)
        else:
            u = net(sam)
        return u
    def update_losses(self, net, epoch):
        if self.is_trainable and epoch % self.train_sample_interval == 0 and epoch > 0 and self.current_epoch <= self.max_epochs:
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
            self.loss_BC1 = self.update_loss_BC(net, self.sam_BC1, loss_measurement)
            self.loss_BC2 = self.update_loss_BC(net, self.sam_BC2, loss_measurement)
            self.loss_IC = self.update_loss_IC(net, self.sam_IC, loss_measurement)
            # Add the losses to the list
            self.losses = [self.loss_D, self.loss_BC1, self.loss_BC2, self.loss_IC]
        else:
            self.loss_D = self.update_loss_D(net, self.sam_D, loss_measurement)
            self.losses = [self.loss_D]
        self.current_epoch += 1
    # Loss term corresponding to the governing PDE in the domain D
    def update_loss_D(self, net, sam_D, loss_measurement):
        x, t = sam_D[:, 0], sam_D[:, 1]
        x.requires_grad = True
        t.requires_grad = True
        net_input = torch.stack((x, t), 1)
        u = self.get_u(net, net_input)
        u_t = gradients(u, t)
        u_x = gradients(u, x)
        u_xx = gradients(u_x, x)
        res = u_t + u.view(u_x.shape) * u_x - self.mu * u_xx
        loss_D = loss_measurement(res, torch.zeros_like(res))
        return loss_D
    # Loss term corresponding to the boundary conditions
    def update_loss_BC(self, net, sam_BC, loss_measurement):
        u = self.get_u(net, sam_BC)
        loss_BC = loss_measurement(u, torch.zeros_like(u))
        return loss_BC
    # Initial condition
    def update_loss_IC(self, net, sam_IC, loss_measurement):
        u = self.get_u(net, sam_IC)
        x = sam_IC[:, 0]
        x = x.detach().cpu().numpy()
        import numpy as np
        u_0 = -np.sin(np.pi * x)
        u_0 = torch.tensor(u_0, dtype=torch.float32, device=self.device).view(-1, 1)
        # u_0 = -torch.sin(torch.pi * sam_IC[:, 0])
        res = u - u_0
        loss_IC = loss_measurement(res, torch.zeros_like(res))
        return loss_IC
    # Initialize the sampling point
    def init_samplingpoints(self, initial_sampling_info):
        sam_num_D = initial_sampling_info["nD"]
        sam_num_BC1 = initial_sampling_info["nBC1"]
        sam_num_BC2 = initial_sampling_info["nBC2"]
        sam_num_IC = initial_sampling_info["nIC"]
        sampler_type = initial_sampling_info["SamplingMethod"]
        if sampler_type == "Uniform":
            sampler = sampler_random_uniform_2D
            requires_grad = False
            self.sam_D = sampler(x_min=-1, x_max=1, y_min=0, y_max=1, num_samples=sam_num_D, requires_grad=requires_grad)
            self.sam_D = self.sam_D.to(self.device)

            self.sam_BC1 = sampler(x_min=-1, x_max=-1, y_min=0, y_max=1, num_samples=sam_num_BC1, requires_grad=requires_grad)
            self.sam_BC1 = self.sam_BC1.to(self.device)

            self.sam_BC2 = sampler(x_min=1, x_max=1, y_min=0, y_max=1, num_samples=sam_num_BC2, requires_grad=requires_grad)
            self.sam_BC2 = self.sam_BC2.to(self.device)

            self.sam_IC = sampler(x_min=-1, x_max=1, y_min=0, y_max=0, num_samples=sam_num_IC, requires_grad=requires_grad)
            self.sam_IC = self.sam_IC.to(self.device)
        else:
            requires_grad = False
            if self.is_trainable:
                kept_sample_num = int(self.kept_ratio * sam_num_D)
                unkept_sample_num = sam_num_D - kept_sample_num
                sam_D1 = quasirandom_sampling(x_min=-1, x_max=1, y_min=0, y_max=1, num_samples=unkept_sample_num,
                                 sampler_name=sampler_type, requires_grad=requires_grad)
                sam_D2 = quasirandom_sampling(x_min=-1, x_max=1, y_min=0, y_max=1, num_samples=kept_sample_num,
                                 sampler_name=sampler_type, requires_grad=requires_grad)
                self.sam_D = torch.cat((sam_D1, sam_D2), 0)
            else:
                self.sam_D = quasirandom_sampling(x_min=-1, x_max=1, y_min=0, y_max=1, num_samples=sam_num_D,
                                                  sampler_name=sampler_type, requires_grad=requires_grad)
            self.sam_D = self.sam_D.to(self.device)

            sampler = sampler_random_uniform_2D
            self.sam_BC1 = sampler(x_min=-1, x_max=-1, y_min=0, y_max=1, num_samples=sam_num_BC1,
                                   requires_grad=requires_grad)
            self.sam_BC1 = self.sam_BC1.to(self.device)

            self.sam_BC2 = sampler(x_min=1, x_max=1, y_min=0, y_max=1, num_samples=sam_num_BC2,
                                   requires_grad=requires_grad)
            self.sam_BC2 = self.sam_BC2.to(self.device)

            self.sam_IC = sampler(x_min=-1, x_max=1, y_min=0, y_max=0, num_samples=sam_num_IC,
                                  requires_grad=requires_grad)
            self.sam_IC = self.sam_IC.to(self.device)
    def update_samples(self, net):
        self.sam_D = self.sam_D.detach()
        training_sample_num = int(self.sam_D.shape[0] * self.sample_ratio)
        kept_sample_num = int(self.kept_ratio * training_sample_num)
        unkept_sample_num = self.sam_D.shape[0] - kept_sample_num
        unkept_samples = self.sam_D[:unkept_sample_num]
        kept_samples = self.sam_D[unkept_sample_num:]

        residuals = self.compute_residuals(net, unkept_samples).view(-1)
        unkept_samples = unkept_samples.detach().cpu().numpy()
        residuals = residuals.detach().cpu().numpy()
        residuals = np.abs(residuals) / np.sum(np.abs(residuals))
        new_sample_points_idx = np.random.choice(a=len(residuals), size=kept_sample_num, replace=False, p=residuals)
        sam_D1 = torch.tensor(unkept_samples[new_sample_points_idx], dtype=torch.float32).to(self.device)
        sam_D2 = unkept_samples[np.setdiff1d(np.arange(len(unkept_samples)), new_sample_points_idx)]
        sam_D2 = torch.tensor(sam_D2, dtype=torch.float32).to(self.device)
        new_sam_D = sam_D1.clone().detach().requires_grad_(True)
        new_sam_D = self.training_samples(net, new_sam_D)
        new_sam_D = torch.cat((sam_D2, new_sam_D), 0)

        new_sam_D = torch.cat((new_sam_D, kept_samples), 0)
        self.sam_D.data = new_sam_D.detach()

    def training_samples(self, net, new_sam_D):
        if self.opt_type == "Adam":
            opt = torch.optim.Adam(params=[new_sam_D], lr=self.opt_lr)
        elif self.opt_type == "SGD":
            opt = torch.optim.SGD(params=[new_sam_D], lr=self.opt_lr)
        loss_func = torch.nn.MSELoss()
        net.eval()
        for _ in range(self.train_sample_iter):
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
        new_sam_D.clamp_(min=torch.tensor([-1, 0], device=self.device), max=torch.tensor([1, 1], device=self.device))
        return new_sam_D

    def update_loss_D_for_training_sample(self, net, sam_D, loss_measurement):
        x, t = sam_D[:, 0], sam_D[:, 1]
        net_input = torch.stack((x, t), 1)
        u = self.get_u(net, net_input)
        u_t = gradients(u, t)
        u_x = gradients(u, x)
        u_xx = gradients(u_x, x)
        res = u_t + u.view(u_x.shape) * u_x - self.mu * u_xx
        loss_D = loss_measurement(res, torch.zeros_like(res))
        return loss_D
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
    def visual_sam(self, epoch):
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
    from src.main import run
    from src.network.createnet import createnet

    # para network (dict): The information about the network architecture
    net_info = {"Name": "test", "Type": "FCNN",
                "ActivationFunc": "Tanh",
                "InputSize": 2, "OutputSize": 1, "HiddenSizes": [50, 50, 50]}
    net = createnet(net_info)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    max_epochs = 1000
    mu = 0.01
    initial_sampling_info = {"nD": 2500, "nBC1": 50, "nBC2": 50, "nIC": 50, "SamplingMethod": "Uniform"}
    sample_update_interval = 100
    loss = LossBurgers(device, max_epochs, mu, initial_sampling_info, sample_update_interval)
    print("Initialization of the loss function is successful!")
    loss.update_losses(net)
    print("The loss function is updated successfully!")

