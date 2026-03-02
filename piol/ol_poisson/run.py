import torch.random
from src.main import run
from src.network.deeponet import DeepONet
from loss_random import Loss_Random
from loss_rar import Loss_RAR
from evaluation import Evaluation
from src.network.cnn import CNN

is_visualized = False
is_net_transformed = True
max_epochs = 75000
training = {"Type": "StandardPI", "MaxEpochs": max_epochs+25000, "LearningRate": 0.0001,
            "isLBFGS": True, "LBFGSMaxEpochs": 2000}
device = None

# Test for resampling method
sample_update_interval = 5000

width = 350
depth = 3
n_x = 50
n_y = 50
bracnh_input_size = n_x * n_y
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for sample_num in [100, 200, 300, 400, 600, 800]:
    initial_sampling_info = {"n": sample_num}
    train_sample_iter = 300
    training_sample_info_random = {"train_sample_interval": sample_update_interval,
                                   "train_sample_iter": train_sample_iter,
                                   "sample_ratio": 0.05, "kept_ratio": 0.5, "opt_type": "Adam", "opt_lr": 0.01,
                                   "den_coef": 0.0}

    net = DeepONet(branchinfo, trunkinfo, channel_size)
    branch_net = CNN(input_size=n_x * n_y, outputsize=width).to(device)
    net.branch_net = branch_net
    print("Random sampling with sample number: ", sample_num)
    loss = Loss_Random(device=device, max_epochs=max_epochs, initial_sampling_info=initial_sampling_info,
                       sample_update_interval=None, is_net_transformed=is_net_transformed, n_x=n_x, n_y=n_y,
                       is_trainable=False, training_sample_info=training_sample_info_random,
                       is_visualized=is_visualized)
    ls_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    evaluation = Evaluation(freq=50, filename="random", ls_list=ls_list)
    net = run(training, loss, net, device=device, evaluation=evaluation)
    net_name = r"models\\" + str(int(sample_num)) + "sam_random.pth"
    torch.save(net, net_name)

    net = DeepONet(branchinfo, trunkinfo, channel_size)
    branch_net = CNN(input_size=n_x * n_y, outputsize=width).to(device)
    net.branch_net = branch_net
    print("Trainable random sampling with sample number: ", sample_num)
    loss = Loss_Random(device=device, max_epochs=max_epochs, initial_sampling_info=initial_sampling_info,
                       sample_update_interval=None, is_net_transformed=is_net_transformed, n_x=n_x, n_y=n_y,
                       is_trainable=True, training_sample_info=training_sample_info_random, is_visualized=is_visualized)
    ls_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    evaluation = Evaluation(freq=50, filename="150width_trainable_random", ls_list=ls_list)
    net = run(training, loss, net, device=device, evaluation=evaluation)
    net_name = r"models\\" + str(int(sample_num)) + "sam_random_trainable_" + str(train_sample_iter) + "iter.pth"
    torch.save(net, net_name)

    updated_sample_num = int(sample_num * 0.3)
    kept_sample_num = int(sample_num * 0.02)
    initial_sampling_info = {"n": int(0.7 * sample_num)}
    train_sample_iter = 300
    training_sample_info_random = {"train_sample_interval": sample_update_interval,
                                   "train_sample_iter": train_sample_iter,
                                   "sample_ratio": 0.05, "kept_ratio": 0.5, "opt_type": "Adam", "opt_lr": 0.01,
                                   "den_coef": 0.0}
    net = DeepONet(branchinfo, trunkinfo, channel_size)
    branch_net = CNN(input_size=n_x * n_y, outputsize=width).to(device)
    net.branch_net = branch_net
    print("RAR sampling with sample number: ", sample_num)
    loss = Loss_RAR(device=device, max_epochs=max_epochs, initial_sampling_info=initial_sampling_info, n_x=n_x, n_y=n_y,
                    sample_update_interval=None, is_net_transformed=is_net_transformed,
                    updated_sample_num=updated_sample_num, kept_sample_num=kept_sample_num,
                    is_trainable=False, training_sample_info=training_sample_info_random, is_visualized=is_visualized)
    ls_list = None
    evaluation = Evaluation(freq=50, filename="150width_results_rar", ls_list=ls_list)
    net = run(training, loss, net, device=device, evaluation=evaluation)
    net_name = r"models\\" + str(int(sample_num)) + "sam_rar.pth"
    torch.save(net, net_name)

    net = DeepONet(branchinfo, trunkinfo, channel_size)
    branch_net = CNN(input_size=n_x * n_y, outputsize=width).to(device)
    net.branch_net = branch_net
    print("Trainable RAR sampling with sample number: ", sample_num)
    loss = Loss_RAR(device=device, max_epochs=max_epochs, initial_sampling_info=initial_sampling_info, n_x=n_x, n_y=n_y,
                    sample_update_interval=None, is_net_transformed=is_net_transformed,
                    updated_sample_num=updated_sample_num, kept_sample_num=kept_sample_num,
                    is_trainable=True, training_sample_info=training_sample_info_random, is_visualized=is_visualized)
    ls_list = None
    evaluation = Evaluation(freq=50, filename="150width_results_rar", ls_list=ls_list)
    net = run(training, loss, net, device=device, evaluation=evaluation)
    # evaluation.get_results(net)
    net_name = r"models\\" + str(int(sample_num)) + "sam_rar_trainable_" + str(train_sample_iter) + "iter.pth"
    torch.save(net, net_name)

import get_results
get_results.run_eval()