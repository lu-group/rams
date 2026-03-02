import torch.random
from src.main import run
from src.network.deeponet import DeepONet
from loss_random import Loss_Random
from loss_rar import Loss_RAR
from evaluation import Evaluation

is_visualized = False
is_net_transformed = False
max_epochs = 16001
training = {"Type": "StandardPI", "MaxEpochs": max_epochs+1000, "LearningRate": 0.001,
            "isLBFGS": False, "LBFGSMaxEpochs": 5000}
device = None

# Test for resampling method
sample_update_interval = 50
sample_num = 1000
initial_sampling_info = {"n": sample_num}
train_sample_iter = 400
training_sample_info_random = {"train_sample_interval": sample_update_interval, "train_sample_iter": train_sample_iter,
                        "sample_ratio": 0.05, "kept_ratio": 0.5,"opt_type": "Adam", "opt_lr": 0.01, "den_coef": 0.0}

width = 50
depth = 5
bracnh_input_size = 8
branch_depth = depth
branch_hidden_size = width
branch_output_size = width
trunk_input_size = 1
trunk_depth = depth
trunk_hidden_size = width
trunk_output_size = width
act_fn = torch.nn.Tanh()
branchinfo = {'act_fn': [act_fn] * branch_hidden_size, 'input_size': bracnh_input_size,
              'output_size': branch_output_size, 'hidden_sizes': [branch_hidden_size] * branch_depth}
trunkinfo = {'act_fn': [act_fn] * trunk_hidden_size, 'input_size': trunk_input_size,
             'output_size': trunk_output_size, 'hidden_sizes': [trunk_hidden_size] * trunk_depth}
channel_size = [branch_output_size]
#
net = DeepONet(branchinfo, trunkinfo, channel_size)
loss = Loss_Random(device=device, max_epochs=max_epochs, initial_sampling_info=initial_sampling_info,
                   sample_update_interval=None, is_net_transformed=is_net_transformed,
                   is_trainable=True, training_sample_info=training_sample_info_random, is_visualized=is_visualized)
evaluation = Evaluation(freq=500, filename="results_random")
net = run(training, loss, net, device=device, evaluation=evaluation)

updated_sample_num = 1000
kept_sample_num = 40
train_sample_iter = 400
training_sample_info_random = {"train_sample_interval": sample_update_interval, "train_sample_iter": train_sample_iter,
                        "sample_ratio": 0.05, "kept_ratio": 0.5,"opt_type": "Adam", "opt_lr": 0.01, "den_coef": 0.0}

net = DeepONet(branchinfo, trunkinfo, channel_size)
loss = Loss_RAR(device=device, max_epochs=max_epochs, initial_sampling_info=initial_sampling_info,
                   sample_update_interval=None, is_net_transformed=is_net_transformed, updated_sample_num=updated_sample_num, kept_sample_num=kept_sample_num,
                   is_trainable=True, training_sample_info=training_sample_info_random, is_visualized=is_visualized)
evaluation = Evaluation(freq=500, filename="results_rar")
net = run(training, loss, net, device=device, evaluation=evaluation)

