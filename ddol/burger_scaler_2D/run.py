import torch
import torch.nn as nn
from training import train as random_train
from training_rar import train as rar_train
from phy_loss_trainable import PhyLoss
import os
import networks_model

# Please generate the dataset using create_dataset.py before running the code below.

training_sample_name = "training_dataset"
test_sample_name = "testing_dataset"
TOL = 1e-9
batch_size = 300
max_epoch = 100000
output_num = 150
trunk_hidden_layer_num = 4
trunk_neuron_num = output_num
trunk_hidden_sizes = [trunk_neuron_num] * trunk_hidden_layer_num
trunk_act_fn = [nn.Tanh()] * trunk_hidden_layer_num

project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
data_path = r"datasets/"

sam_num_list = [1500,2000,2500, 3000]
resampling_portion = [0.1, 0.2, 0.4]
resampling_time = 1 #or 2
model_path = "resampling_time_" + str(resampling_time) + r"results_nnmodels/"
if not os.path.exists(model_path):
    os.makedirs(model_path)
iteration_for_sampling = 100

# Correlation length for kernel smoothing method
proj_l = 0.5
PhyLoss.proj_l = proj_l
# Viscosity for Burgers
nu = 0.1
PhyLoss.mu = nu

for sample_num in sam_num_list:
    for i in range(3):
        # Baseline using the random training
        print("Random training: sample_num: {}, rep: {}".format(sample_num, i))
        model_name = "random_samnum_{}_rep_{}".format(sample_num, i)
        trunk_net = networks_model.ResidualFCNN(act_fn=trunk_act_fn, input_size=3, output_size=output_num,
                                                hidden_sizes=trunk_hidden_sizes)
        mesh_size = 64
        branch_net = networks_model.CNN(input_size=mesh_size * mesh_size, outputsize=output_num)
        net = networks_model.DeepONet(trunk_net=trunk_net, branch_net=branch_net)
        random_train(net=net, batch_size=batch_size, max_epoch=max_epoch, TOL=TOL,
              data_path=data_path, training_sample_name=training_sample_name, test_sample_name=test_sample_name,
              model_path=model_path, model_name=model_name, sample_num=sample_num,
              device=None, lr=0.001, early_stopping_epoch=500, is_loss_plot=False)

    for portion in resampling_portion:
        for i in range(3):
            print("RAR training: sample_num: {}, portion: {}, rep: {}".format(sample_num, portion, i))
            first_sampling_num = int(sample_num - sample_num * portion)
            rar_update_info = {'start_epoch': 500, 'interval': 1000, 'num': int(sample_num * portion / resampling_time),
                               'update_time': resampling_time}
            model_name = "rar_samnum_{}_portion_{}_rep_{}".format(sample_num, portion, i)
            trunk_net = networks_model.ResidualFCNN(act_fn=trunk_act_fn, input_size=3, output_size=output_num,
                                                    hidden_sizes=trunk_hidden_sizes)
            mesh_size = 64
            branch_net = networks_model.CNN(input_size=mesh_size * mesh_size, outputsize=output_num)
            net = networks_model.DeepONet(trunk_net=trunk_net, branch_net=branch_net)
            rar_train(net=net, batch_size=batch_size, max_epoch=max_epoch, TOL=TOL,
                      data_path=data_path, training_sample_name=training_sample_name, test_sample_name=test_sample_name,
                      model_path=model_path, model_name=model_name, sample_num=first_sampling_num, device=None, lr=0.001,
                      early_stopping_epoch=500, is_loss_plot=False,
                      rar_update_info=rar_update_info, PhyLoss=PhyLoss, sampling_iter=iteration_for_sampling)

from evaluation import evaluation
evaluation(resampling_time=resampling_time, sam_num_list=[1500, 2000, 2500, 3000])