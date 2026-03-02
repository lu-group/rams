# Training process for Multiple-input DeepONet with the fixed branch input for each epoch
import numpy as np
import os
from tqdm import *
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from loss_visual import LossEvaluation
from src.network.deeponet import DeepONet
# from src.util.mideeponet import MIDeepONet
# from torch.cuda.amp import GradScaler, autocast


class creatDataSet(Dataset):
    def __init__(self, data_path, sample_name, sample_num=None, trunk_sample_num=None,
                 branch_input_min=None, branch_input_max=None,
                 trunk_input_min=None, trunk_input_max=None,
                 output_min=None, output_max=None, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load data from .npz file
        dataset = np.load(data_path + sample_name + ".npz")
        branch_input = dataset["branch_input"][:sample_num,:]
        trunk_input = dataset["trunk_input"]
        label = dataset["results"][:sample_num,:]
        self.node = dataset["node"] # This is the node locations for resampling in the propsoed method
        self.branch_input = torch.tensor(branch_input, dtype=torch.float32).to(device)
        self.trunk_input = torch.tensor(trunk_input, dtype=torch.float32).to(device)
        self.label = torch.tensor(label, dtype=torch.float32).to(device)
        self.label_norm = torch.norm(self.label, dim=1)

    def __getitem__(self, idx):
        # idx is the index of the branch input data
        # return: the branch input tensor; the trunk input tensor; the label tensor
        return self.branch_input[idx], self.label[idx], self.label_norm[idx]
        # return self.branch_input_tensor[idx], self.label_tensor[idx * self.trunk_sample_num: (idx + 1) * self.trunk_sample_num], self.label_norm[idx]

    def __len__(self):
        return self.branch_input.size(0)

def get_predict(net, branch_input, trunk_input):
    branch_output = net.branch_net(branch_input)
    trunk_output = net.trunk_net(trunk_input)
    pred = torch.matmul(branch_output, trunk_output.T)
    return pred

def train(net, batch_size, max_epoch, TOL,
          data_path, training_sample_name, test_sample_name,
          model_path, model_name, sample_num,
          device=None, lr=0.001, weight_decay=0, early_stopping_epoch=500,
          is_loss_plot=False, loss_record_interval=50):

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("The device is %s." % device)

    train_dataset = creatDataSet(data_path, training_sample_name, sample_num=sample_num)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = creatDataSet(data_path, sample_name=test_sample_name, sample_num=None)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    net = net.to(device)

    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    # opt = torch.optim.SGD(net.parameters(), lr=lr)

    loss_fn = torch.nn.MSELoss()
    # loss_fn = mape
    loss_MIN = 1e10

    desc = "start training..."
    pbar = tqdm(range(max_epoch), desc=desc)
    iterCounter = 0

    if is_loss_plot:
        LossEvaluation.init_plot(max_epoch)

    train_trunk_input = train_dataset.trunk_input
    test_trunk_input = test_dataset.trunk_input
    for epoch in pbar:
        train_loss_list = []
        for idx, (train_branch_input, train_label, train_label_norm) in enumerate(train_loader):
            opt.zero_grad()
            prediction = get_predict(net, train_branch_input, train_trunk_input)
            opt.zero_grad()
            loss = loss_fn(prediction, train_label)
            # loss = torch.norm(prediction - train_label, dim=1) / train_label_norm
            # loss = torch.mean(loss)
            train_loss_list.append(loss.item())
            loss.backward()
            opt.step()

        net.eval()
        train_loss_value = np.mean(train_loss_list)
        if is_loss_plot and epoch % loss_record_interval == 0:
            LossEvaluation.update_loss_train(train_loss_value, epoch)

        for idx, (test_branch_input, test_label, test_label_norm) in enumerate(test_loader):
            prediction = get_predict(net, test_branch_input, test_trunk_input)
            opt.zero_grad()
            loss = loss_fn(prediction, test_label)
            # loss = torch.norm(prediction - test_label, dim=1) / test_label_norm
            # loss = torch.mean(loss)

        test_loss_value = loss.item()
        net.train()

        if is_loss_plot and epoch % loss_record_interval == 0:
            LossEvaluation.update_loss_test(loss.item(), epoch)

        pbar.set_description("Epoch %d, iterCounter %d, Train Loss %.2e, Test Loss %.2e, Minimum Loss %.2e" % (
            epoch, iterCounter, train_loss_value, test_loss_value, loss_MIN))

        if epoch % loss_record_interval == 0 and is_loss_plot and epoch > 0:
            LossEvaluation.update_plot()

        iterCounter += 1
        if epoch > 100 and epoch % 1000 == 0:
            loss_str = "%.2e" % loss_MIN
            cache_model_path = r"cache_models\\"
            import os
            if not os.path.exists(cache_model_path):
                os.makedirs(cache_model_path)
            torch.save(net, cache_model_path + model_name + "_" + "epoch_" + str(epoch) + "_loss_" + loss_str + ".pth")
        if loss.item() < loss_MIN:
            iterCounter = 0
            loss_MIN = loss.item()
            if epoch > 1000:
                torch.save(net, model_path + model_name + ".pth")

        if early_stopping_epoch is not None:
            if iterCounter > early_stopping_epoch:
                print("The loss value is not decreasing. Stop training.")
                return

        if loss.item() < TOL:
            print("Tolerance is satisfied! Stop training.")
            torch.save(net, model_path + model_name + ".pth")
            break

    print("The current loss value is %.2e. Stop training." % loss_MIN)
    # NetworkSL.save_model(model, model_path, name)

    return

def run(sample_num=50):
    import os
    project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_path = r"datasets\\"
    model_path = r"nnmodels_50sam\\"
    import os
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    model_name = "random_sample"  # "ex5.3_helmholtz_40mesh_3000sam_2pi"
    training_sample_name = "training_dataset2"
    test_sample_name = "validation_dataset2"
    batch_size = 16
    TOL = 1e-9

    width = 100
    depth = 4
    bracnh_input_size = 201
    branch_depth = depth
    branch_hidden_size = width
    branch_output_size = width
    trunk_input_size = 2
    trunk_depth = depth
    trunk_hidden_size = width
    trunk_output_size = width
    act_fn = torch.nn.Tanh()
    branchinfo = {'act_fn': [act_fn] * branch_hidden_size, 'input_size': bracnh_input_size,
                  'output_size': branch_output_size, 'hidden_sizes': [branch_hidden_size] * branch_depth}
    trunkinfo = {'act_fn': [act_fn] * trunk_hidden_size, 'input_size': trunk_input_size,
                 'output_size': trunk_output_size, 'hidden_sizes': [trunk_hidden_size] * trunk_depth}
    channel_size = [branch_output_size]

    lr = 0.0005

    max_epoch = 100000
    for i in range(3):
        net = DeepONet(branchinfo, trunkinfo, channel_size)
        model_name = "trandom_sample_rep_" + str(i)
        train(net=net, batch_size=batch_size, max_epoch=max_epoch, TOL=TOL,
              data_path=data_path, training_sample_name=training_sample_name, test_sample_name=test_sample_name,
              model_path=model_path, model_name=model_name, sample_num=sample_num,
              device=None, lr=0.001, early_stopping_epoch=5000, is_loss_plot=False)


if __name__ == '__main__':
    run()
