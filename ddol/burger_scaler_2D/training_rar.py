# Training process for Multiple-input DeepONet with the fixed branch input for each epoch
import numpy as np
import os
from tqdm import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from loss_visual import LossEvaluation


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
        label = dataset["label"][:sample_num,:]
        self.node = dataset["node"] # This is the node locations for resampling in the propsoed method
        self.branch_input = torch.tensor(branch_input, dtype=torch.float32).to(device)
        self.trunk_input = torch.tensor(trunk_input, dtype=torch.float32).to(device)
        self.label = torch.tensor(label, dtype=torch.float32).to(device)

    def move_to(self, device):
        self.branch_input = self.branch_input.to(device)
        self.trunk_input = self.trunk_input.to(device)
        self.label = self.label.to(device)

    def __getitem__(self, idx):
        # idx is the index of the branch input data
        # return: the branch input tensor; the trunk input tensor; the label tensor
        return self.branch_input[idx], self.label[idx]
        # return self.branch_input_tensor[idx], self.label_tensor[idx * self.trunk_sample_num: (idx + 1) * self.trunk_sample_num], self.label_norm[idx]

    def __len__(self):
        return self.branch_input.size(0)

def train(net, batch_size, max_epoch, TOL,
          data_path, training_sample_name, test_sample_name,
          model_path, model_name, sample_num,
          device=None, lr=0.001, weight_decay=0, early_stopping_epoch=500, lbfgs_max_epochs=150,
          is_loss_plot=False, loss_record_interval=50, rar_update_info=None, PhyLoss=None, sampling_iter=200):

    rar_update_start_epoch = rar_update_info['start_epoch']
    rar_update_interval = rar_update_info['interval']
    rar_update_num = rar_update_info['num']  # Number of  samples to update in each time
    rar_update_time = rar_update_info['update_time']

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

    loss_fn = torch.nn.MSELoss()
    # loss_fn = mape
    loss_MIN = 1e10

    desc = "start training..."
    pbar = tqdm(range(max_epoch), desc=desc)
    iterCounter = 0
    current_updated_time = 0

    if is_loss_plot:
        LossEvaluation.init_plot(max_epoch)

    train_trunk_input = train_dataset.trunk_input
    test_trunk_input = test_dataset.trunk_input
    for epoch in pbar:

        if current_updated_time < rar_update_time:
            if iterCounter >= rar_update_start_epoch:
                if current_updated_time == 0:
                    early_stopping_epoch = rar_update_interval
                    rar_update_start_epoch = rar_update_interval
                print("Updating training samples for the %d-th time." % current_updated_time + "The current epoch is %d." % current_updated_time)
                net = torch.load(model_path + model_name + ".pth")
                opt = torch.optim.Adam(net.parameters(), lr=lr/3, weight_decay=weight_decay)
                phyloss = PhyLoss(branch_sam_num=1000, trunk_sam_num=10000, device=device, max_iter=sampling_iter)
                train_dataset, train_loader = phyloss.update_training_samples(net, kept_num=rar_update_num,
                                                                              train_dataset=train_dataset,
                                                                              batch_size=batch_size)
                del phyloss
                current_updated_time += 1
                torch.cuda.empty_cache()
                iterCounter = 0

        train_loss_list = []
        for idx, (train_branch_input, train_label) in enumerate(train_loader):
            opt.zero_grad()
            predict_u = net(branch_input=train_branch_input, trunk_input=train_trunk_input)
            opt.zero_grad()
            loss = loss_fn(predict_u, train_label)
            train_loss_list.append(loss.item())
            loss.backward()
            opt.step()

        net.eval()
        train_loss_value = np.mean(train_loss_list)
        if is_loss_plot and epoch % loss_record_interval == 0:
            LossEvaluation.update_loss_train(train_loss_value, epoch)

        for idx, (test_branch_input, test_label) in enumerate(test_loader):
            predict_u = net(branch_input=test_branch_input, trunk_input=test_trunk_input)
            opt.zero_grad()
            loss = loss_fn(predict_u, test_label)

        test_loss_value = loss.item()
        net.train()

        if is_loss_plot and epoch % loss_record_interval == 0:
            LossEvaluation.update_loss_test(loss.item(), epoch)

        pbar.set_description("Epoch %d, iterCounter %d, Train Loss %.2e, Val. Loss %.2e, Minimum Loss %.2e" % (
            epoch, iterCounter, train_loss_value, test_loss_value, loss_MIN))

        if epoch % loss_record_interval == 0 and is_loss_plot and epoch > 0:
            LossEvaluation.update_plot()

        iterCounter += 1
        if epoch > 100 and epoch % 1000 == 0:
            loss_str = "%.2e" % loss_MIN
            cache_model_path = r"cache_models/"
            if not os.path.exists(cache_model_path):
                os.makedirs(cache_model_path)
            torch.save(net, cache_model_path + model_name + "_" + "epoch_" + str(epoch) + "_loss_" + loss_str + ".pth")

        if loss.item() < loss_MIN:
            iterCounter = 0
            loss_MIN = loss.item()
            if epoch > 0:
                torch.save(net, model_path + model_name + ".pth")

        if early_stopping_epoch is not None:
            if iterCounter > early_stopping_epoch:
                print("The loss value is not decreasing. Stop training.")
                break

        if loss.item() < TOL:
            print("Tolerance is satisfied! Stop training.")
            torch.save(net, model_path + model_name + ".pth")
            break

    # print("The current loss value is %.2e. Stop training." % loss_MIN)
    # NetworkSL.save_model(model, model_path, name)
    iterCounter = 0
    desc = "L-BFGS fine-tuning..."
    net = torch.load(model_path + model_name + ".pth")
    pbar = tqdm(range(lbfgs_max_epochs), desc=desc)
    opt = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=20, history_size=10)
    for epoch in pbar:
        train_loss_list = []
        # Loop through the training data loader
        for idx, (train_branch_input, train_label) in enumerate(train_loader):
            def closure():
                opt.zero_grad()
                # Recompute prediction and loss within the closure
                predict_u = net(branch_input=train_branch_input, trunk_input=train_trunk_input)
                loss = loss_fn(predict_u, train_label)
                loss.backward(retain_graph=True)  # retain_graph=True to allow multiple backward passes
                return loss

            # Record the loss and update the model parameters
            train_loss_list.append(loss.item())
            opt.step(closure)

        net.eval()
        train_loss_value = np.mean(train_loss_list)

        for idx, (test_branch_input, test_label) in enumerate(test_loader):
            predict_u = net(branch_input=test_branch_input, trunk_input=test_trunk_input)
            loss = loss_fn(predict_u, test_label)

        test_loss_value = loss.item()
        net.train()

        pbar.set_description("Epoch %d, iterCounter %d, Train Loss %.2e, Val. Loss %.2e, Minimum Loss %.2e" % (
            epoch, iterCounter, train_loss_value, test_loss_value, loss_MIN))

        iterCounter += 1

        if iterCounter > 20:
            break

        if epoch > 100 and epoch % 1000 == 0:
            loss_str = "%.2e" % loss_MIN
            cache_model_path = r"cache_models/"
            if not os.path.exists(cache_model_path):
                os.makedirs(cache_model_path)
            torch.save(net,
                       cache_model_path + model_name + "_" + "lbgfs_epoch_" + str(epoch) + "_loss_" + loss_str + ".pth")

        if loss.item() < loss_MIN:
            iterCounter = 0
            loss_MIN = loss.item()
            if epoch > 0:
                torch.save(net, model_path + model_name + ".pth")

        if loss.item() < TOL:
            print("Tolerance is satisfied! Stop training.")
            torch.save(net, model_path + model_name + ".pth")
            break


    torch.save(net, model_path + model_name + "_final.pth")
    return

if __name__ == '__main__':
    pass