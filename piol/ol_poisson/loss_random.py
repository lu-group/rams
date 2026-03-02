import numpy as np
from src.loss.loss_base import LossBase
import torch
import grf
from src.util.gradient import gradients
import matplotlib.pyplot as plt

class Loss_Random(LossBase):
    def __init__(self, device, max_epochs, initial_sampling_info, sample_update_interval, proj_l=0.3, n_x=100, n_y=100,
                 is_net_transformed=True, is_trainable=False, training_sample_info=None, is_visualized=False, k_min=0.5, b=0.3, k_max=1,
                 resampling_collocation_interval=2500):
        super().__init__()
        self.n_x = n_x
        self.n_y = n_y
        self.k_max = k_max
        self.k_min = k_min
        self.b = b # Width of the central area
        self.proj_l = proj_l
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
        self.resampling_collocation_interval = resampling_collocation_interval

    def get_k(self, trunk_input, is_retain_graph=False):
        k_max = self.k_max
        k_min = self.k_min
        b = self.b
        k = torch.ones(trunk_input.shape[0], dtype=torch.float32) * k_max
        # for those -b <x < b, -b < y < b, k = k_min
        k[(trunk_input[:, 0] > -b) & (trunk_input[:, 0] < b)
          & (trunk_input[:, 1] > -b) & (trunk_input[:, 1] < b)] = k_min
        k = k.to(self.device)
        if not is_retain_graph:
            k = k.detach()
        return k

    def update_trunk_input(self, branch_input):
        num_trunk_input = self.n_x * self.n_y * 4
        grid_x = torch.rand(num_trunk_input).to(self.device)
        grid_y = torch.rand(num_trunk_input).to(self.device)
        # Scale to the domain [-1, 1] ^ 2
        grid_x = 2 * grid_x - 1
        grid_y = 2 * grid_y - 1
        trunk_input = torch.stack((grid_x, grid_y), 1)
        # Compute the source term via the 2D linear interpolation using branchnet
        grid_x0, grid_y0 = np.meshgrid(np.linspace(-1, 1, self.n_x), np.linspace(-1, 1, self.n_y))
        node0 = np.array([grid_x0.flatten(), grid_y0.flatten()]).T
        node0 = torch.tensor(node0, dtype=torch.float32).to(self.device)
        self.trunk_input = trunk_input
        self.f = self.get_f(branch_input, node0, self.trunk_input)
        self.k = self.get_k(self.trunk_input)

    def get_f(self, branch_input, node0, trunk_input, is_retain_graph=False):
        x = trunk_input[:, 0]
        y = trunk_input[:, 1]
        seg_length_x = 2 / (self.n_x - 1)
        seg_length_y = 2 / (self.n_y - 1)
        index_x = (x.detach() + 1) // seg_length_x
        index_y = (y.detach() + 1) // seg_length_y
        u1_index = index_x + index_y * self.n_x
        u2_index = index_x + 1 + index_y * self.n_x
        u3_index = index_x + 1 + (index_y + 1) * self.n_x
        u4_index = index_x + (index_y + 1) * self.n_x
        u1_index = u1_index.long()
        u2_index = u2_index.long()
        u3_index = u3_index.long()
        u4_index = u4_index.long()
        node1 = node0[u1_index]
        u1 = branch_input[:, u1_index]
        u2 = branch_input[:, u2_index]
        u3 = branch_input[:, u3_index]
        u4 = branch_input[:, u4_index]
        x_prime = x - node1[:, 0]
        y_prime = y - node1[:, 1]
        # f = a * x' + b * y' + c*x'*y' + d
        a = (u2 - u1) / seg_length_x
        b = (u4 - u1) / seg_length_y
        c = (u1 - u2 - u4 + u3) / (seg_length_x * seg_length_y)
        d = u1
        f = a * x_prime + b * y_prime + c * x_prime * y_prime + d
        if not is_retain_graph:
            f = f.detach()
        return f

    def update_losses(self, net, epoch):
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


    def update_loss_D(self, net, branch_input, trunk_input, loss_measurement, is_residual=False):
        is_transformed = self.is_net_transformed
        def get_trunk_output(net, trunk_input_tensor):
            trunk_output = net.trunk_net(trunk_input_tensor)
            channel_size = net.model_channel_size[0]
            if is_transformed:
                x = trunk_input_tensor[:, 0].view(-1,1)
                y = trunk_input_tensor[:, 1].view(-1,1)
                x = x.repeat(1, channel_size)
                y = y.repeat(1, channel_size)
                trunk_output = trunk_output * (1 - x**2) * (1 - y**2)
            return trunk_output

        dx = 5e-3
        dy = 5e-3
        trunk_output0 = get_trunk_output(net, trunk_input)
        trunk_output_pdx = get_trunk_output(net, trunk_input + torch.tensor([dx, 0]).to(self.device))
        trunk_output_rdx = get_trunk_output(net, trunk_input - torch.tensor([dx, 0]).to(self.device))
        trunk_output_dxx = (trunk_output_pdx - 2 * trunk_output0 + trunk_output_rdx) / (dx ** 2)
        trunk_output_pdy = get_trunk_output(net, trunk_input + torch.tensor([0, dy]).to(self.device))
        trunk_output_rdy = get_trunk_output(net, trunk_input - torch.tensor([0, dy]).to(self.device))
        trunk_output_dyy = (trunk_output_pdy - 2 * trunk_output0 + trunk_output_rdy) / (dy ** 2)
        trunk_output_laplacian = trunk_output_dxx + trunk_output_dyy
        trunk_output_laplacian = trunk_output_laplacian.transpose(0, 1)
        trunk_output_laplacian = trunk_output_laplacian * self.k

        branch_output = net.branch_net(branch_input)
        res = torch.matmul(branch_output, trunk_output_laplacian) - self.f
        if is_residual:
            return res
        else:
            loss = loss_measurement(res, torch.zeros_like(res))
            return loss
    # Loss term corresponding to the boundary conditions
    def update_loss_BC(self, net, branch_input, trunk_input, loss_measurement):
        # raise NotImplementedError
        branch_output = net.branch_net(branch_input)
        trunk_output = net.trunk_net(trunk_input).transpose(0, 1)
        pred = torch.matmul(branch_output, trunk_output)
        loss = loss_measurement(pred, torch.zeros_like(pred))
        return loss
    # boundary condition
    def init_samplingpoints(self, initial_sampling_info):
        sam_num = initial_sampling_info["n"]
        proj_l = self.proj_l
        n_x = self.n_x
        n_y = self.n_y
        grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, n_x), np.linspace(-1, 1, n_y))
        node = np.array([grid_x.flatten(), grid_y.flatten()]).T
        branch_input, L = grf.grf_2D(node, l=proj_l, sam_num=sam_num)
        self.L = torch.tensor(L, dtype=torch.float32).to(self.device)
        self.branch_sample_node = torch.tensor(node, dtype=torch.float32).to(self.device)
        branch_input = torch.tensor(branch_input, dtype=torch.float32).to(self.device)
        branch_input_norm = torch.norm(branch_input, dim=1) / (n_x * n_y) ** 0.5 / 2
        branch_input = branch_input / branch_input_norm[:, None]
        # branch_input = branch_input * 0 + 1
        self.branch_input = branch_input

        def gaussian_kernel(x, y, l):
            """Generate a Gaussian kernel matrix with correlation length l."""
            # Efficiently calculate the squared exponential kernel
            dx = x[:, np.newaxis] - x[np.newaxis, :]
            dy = y[:, np.newaxis] - y[np.newaxis, :]
            cov_matrix = torch.exp(-0.5 * (dx ** 2 + dy ** 2) / l ** 2)
            return cov_matrix

        self.kernel_matrix = gaussian_kernel(self.branch_sample_node[:, 0], self.branch_sample_node[:, 1], proj_l)
        self.update_trunk_input(self.branch_input)

        if not self.is_net_transformed:
            x = torch.linspace(-0.995, 0.995, 100)
            y = torch.linspace(-0.995, 0.995, 100)
            node1 = torch.stack((x, torch.ones_like(x) * -1), dim=0)
            node2 = torch.stack((x, torch.ones_like(x) * 1), dim=0)
            node3 = torch.stack((torch.ones_like(y) * -1, y), dim=0)
            node4 = torch.stack((torch.ones_like(y) * 1, y), dim=0)
            trunk_input_x0 = torch.cat((node1, node2, node3, node4), dim=1).transpose(0, 1)
            self.trunk_input_BC = trunk_input_x0.to(self.device)

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
        self.f = self.get_f(self.branch_input, self.branch_sample_node, self.trunk_input)
        # self.k = self.get_k(self.trunk_input)

    def training_samples(self, net, new_sam):
        if self.opt_type == "Adam":
            opt = torch.optim.Adam(params=[new_sam], lr=self.opt_lr)
        elif self.opt_type == "SGD":
            opt = torch.optim.SGD(params=[new_sam], lr=self.opt_lr)
        loss_func = torch.nn.MSELoss()
        net.eval()
        for _ in range(self.train_sample_iter):
            tsam = (self.kernel_matrix @ new_sam.T / self.kernel_matrix.sum(dim=1, keepdim=True)).T
            tsam_norm = torch.norm(tsam, dim=1) / (self.n_x * self.n_y) ** 0.5 / 2
            tsam = tsam / tsam_norm[:, None]
            opt.zero_grad()
            loss = -1 * self.update_loss_D_for_training_sample(net, branch_input=tsam, trunk_input=self.trunk_input,
                                                               loss_measurement=loss_func)
            loss.backward()
            opt.step()
            # Project the samples back to the original space with the function norm = 1
        # new_sam.requires_grad_(False)
        new_sam = (self.kernel_matrix @ new_sam.T / self.kernel_matrix.sum(dim=1, keepdim=True)).T
        new_sam_norm = torch.norm(new_sam, dim=1) / (self.n_x * self.n_y) ** 0.5 / 2
        new_sam = new_sam / new_sam_norm[:, None]
        # new_sam.requires_grad_(True)
        # tsam = new_sam
        net.train()
        new_sam = new_sam.detach()
        # new_sam.requires_grad_(False)
        return new_sam

    def update_loss_D_for_training_sample(self, net, branch_input, trunk_input, loss_measurement, is_residual=False):
        is_transformed = self.is_net_transformed

        def get_trunk_output(net, trunk_input_tensor):
            trunk_output = net.trunk_net(trunk_input_tensor)
            channel_size = net.model_channel_size[0]
            if is_transformed:
                x = trunk_input_tensor[:, 0].view(-1, 1)
                y = trunk_input_tensor[:, 1].view(-1, 1)
                x = x.repeat(1, channel_size)
                y = y.repeat(1, channel_size)
                trunk_output = trunk_output * (1 - x ** 2) * (1 - y ** 2)
            return trunk_output

        dx = 1e-3
        dy = 1e-3
        trunk_output0 = get_trunk_output(net, trunk_input)
        trunk_output_pdx = get_trunk_output(net, trunk_input + torch.tensor([dx, 0]).to(self.device))
        trunk_output_rdx = get_trunk_output(net, trunk_input - torch.tensor([dx, 0]).to(self.device))
        trunk_output_dxx = (trunk_output_pdx - 2 * trunk_output0 + trunk_output_rdx) / (dx ** 2)
        trunk_output_pdy = get_trunk_output(net, trunk_input + torch.tensor([0, dy]).to(self.device))
        trunk_output_rdy = get_trunk_output(net, trunk_input - torch.tensor([0, dy]).to(self.device))
        trunk_output_dyy = (trunk_output_pdy - 2 * trunk_output0 + trunk_output_rdy) / (dy ** 2)
        trunk_output_laplacian = trunk_output_dxx + trunk_output_dyy
        trunk_output_laplacian = trunk_output_laplacian.transpose(0, 1)
        trunk_output_laplacian = trunk_output_laplacian * self.k

        branch_output = net.branch_net(branch_input)
        f = self.get_f(branch_input, self.branch_sample_node, trunk_input, is_retain_graph=True)
        res = torch.matmul(branch_output, trunk_output_laplacian) - f
        if is_residual:
            return res
        else:
            loss = loss_measurement(res, torch.zeros_like(res))
            return loss
    def compute_residuals(self, net, branch_input, trunk_input):
        res = self.update_loss_D_for_training_sample(net, branch_input, trunk_input, None, is_residual=True).detach()
        res_norm = torch.norm(res, dim=1)
        return res_norm

    def visual_sam(self, epoch, vis_num=4):
        samples = self.branch_input.cpu().detach().numpy()
        x = np.linspace(0, 1, len(samples[0]))
        for i in range(vis_num):
            sample = samples[i]
            plt.plot(x, sample)
        plt.xlabel("x")
        plt.ylabel("v")
        plt.grid(True)
        plt.text(0.02, 0.02, "Epoch: " + str(epoch), horizontalalignment='center', verticalalignment='bottom',
                 bbox=dict(facecolor='white', alpha=0.5))
        plt.show()
if __name__ == '__main__':
    pass

