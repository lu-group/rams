import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def add_results_to_file(file_path, content):
    with open(file_path, "a") as file:
        file.write("\n" + content)

class Evaluation():
    def __init__(self, device=None, freq=2, filename="results"):
        if device is None:
            # Determine the device to be used for the training
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.frequency = freq
        self.filename = filename
        add_results_to_file(self.filename, "Epoch, MSE error1, MSE error2")

    def evaluate(self, net, epoch):
        mse_error1, mse_error2 = self.plot_results(net, is_print=False)
        add_results_to_file(self.filename, str(epoch) + ", " + str(mse_error1.item()) + ", " + str(mse_error2.item()))

    def plot_results(self, net, is_print=True):
        x = torch.linspace(0, 1, 100).view(-1,1).to(self.device)
        # Evaluate the results on dataset1_input.csv and dataset1_output.csv
        branch_input = pd.read_csv('dataset1_input.csv')
        acc_results = pd.read_csv('dataset1_output.csv')
        branch_input = torch.tensor(branch_input.values, dtype=torch.float32).to(self.device)
        acc_results = torch.tensor(acc_results.values, dtype=torch.float32).to(self.device)
        acc_results = acc_results.reshape(-1,1)
        pred_results = net.forward_branch_trunk_fixed(branch_input, x)
        mse_error1 = torch.nn.MSELoss()(pred_results, acc_results)
        if is_print:
            print('MSE error1: ', mse_error1)

        # Evaluate the results on dataset2_input.csv and dataset2_output.csv
        branch_input = pd.read_csv('dataset2_input.csv')
        acc_results = pd.read_csv('dataset2_output.csv')
        branch_input = torch.tensor(branch_input.values, dtype=torch.float32).to(self.device)
        acc_results = torch.tensor(acc_results.values, dtype=torch.float32).to(self.device)
        acc_results = acc_results.reshape(-1, 1)
        pred_results = net.forward_branch_trunk_fixed(branch_input, x)
        mse_error2 = torch.nn.MSELoss()(pred_results, acc_results)
        if is_print:
            print('MSE error2: ', mse_error2)
        return mse_error1, mse_error2


if __name__ == '__main__':
    # Evaluation()
    # Evaluation().plot_results(None)
    acc_results = pd.read_csv('dataset2_output.csv')
    acc_results = torch.tensor(acc_results.values, dtype=torch.float32)
    acc_results = acc_results.reshape(-1, 1)
    # Calculate the mean square value
    print(torch.nn.MSELoss()(acc_results, torch.zeros_like(acc_results)))
