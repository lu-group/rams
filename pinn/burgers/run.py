import torch.random

from src.main import run
from src.network.createnet import createnet
from loss_random import LossBurgers
from loss_rar import LossBurgers_RAR
from loss_r3 import LossBurgers_R3
from loss_rad import LossBurgers_RAD
from evaluation import Evaluation_Burgers
import math
import numpy as np

def add_results_to_file(file_path, content):
    with open(file_path, "a") as file:
        file.write("\n" + content)

"""========================================================================================================
Comparison of the results with and without training
========================================================================================================"""
import time
current_time = time.strftime("%Y-%m-%d", time.localtime())
filename = str(current_time) + "_results"
add_results_to_file(file_path=filename, content="*"*75)

test_num = 1
mu = 0.01 / math.pi

is_visualized = False
net_info = {"Name": "burgers_baseline", "Type": "FCNN", "ActivationFunc": "Tanh",
            "InputSize": 2, "OutputSize": 1, "HiddenSizes": [100,100,100]}
is_net_transformed = True
max_epochs = 15000
training = {"Type": "StandardPI", "MaxEpochs": max_epochs+5000, "LearningRate": 0.001,
            "isLBFGS": True, "LBFGSMaxEpochs": 1000}
device = None

# For resampling method
sample_update_interval = 200
sample_num = 2500
# Initial sampling information
initial_sampling_info = {"nD": sample_num, "nBC1": 150, "nBC2": 150, "nIC": 150, "SamplingMethod": "Uniform"}
train_sample_iter = 5
training_sample_info_rar = {"train_sample_iter": train_sample_iter, "opt_type": "Adam", "opt_lr": 0.001, "den_coef": 0.0}
updated_sample_num = 1000
kept_sample_num = 40
rar_wo_results = [] # Without RAMS
for _ in range(test_num):
    net = createnet(net_info)
    loss = LossBurgers_RAR(device=device, max_epochs=max_epochs, mu=mu, initial_sampling_info=initial_sampling_info,
                           sample_update_interval=sample_update_interval, updated_sample_num=updated_sample_num,
                           kept_sample_num=kept_sample_num, is_net_transformed=is_net_transformed, is_maximized=False,
                           is_trainable=False, training_sample_info=training_sample_info_rar, is_visualized=is_visualized)
    evaluation = Evaluation_Burgers(mu=mu, freq=1000, is_net_transformed=is_net_transformed)
    net = run(training, loss, net, device=device, evaluation=evaluation)
    error = evaluation.plot_results(net, is_show=False)
    print("RAR w/o Training Error: ", error)
    rar_wo_results.append(error)
    add_results_to_file(file_path=filename, content="RAR w/o Training Error: " + str(error))

rar_w_results = []
for _ in range(test_num):
    net = createnet(net_info)
    loss = LossBurgers_RAR(device=device, max_epochs=max_epochs, mu=mu, initial_sampling_info=initial_sampling_info,
                           sample_update_interval=sample_update_interval, updated_sample_num=updated_sample_num,
                           kept_sample_num=kept_sample_num, is_net_transformed=is_net_transformed, is_maximized=False,
                           is_trainable=True, training_sample_info=training_sample_info_rar, is_visualized=is_visualized)
    evaluation = Evaluation_Burgers(mu=mu, freq=1000, is_net_transformed=is_net_transformed)
    net = run(training, loss, net, device=device, evaluation=evaluation)
    error = evaluation.plot_results(net, is_show=False)
    print("RAR with Training Error: ", error)
    rar_w_results.append(error)
    add_results_to_file(file_path=filename, content="RAR with Training Error: " + str(error))

# Initial sampling information
train_sample_iter = 5
training_sample_info_rad = {"train_sample_iter": train_sample_iter, "opt_type": "Adam", "opt_lr": 0.001, "den_coef": 0.0}
updated_sample_num = 1000
kept_sample_num = 40
rad_wo_results = [] # With RAMS
for _ in range(test_num):
    net = createnet(net_info)
    loss = LossBurgers_RAD(device=device, max_epochs=max_epochs, mu=mu, initial_sampling_info=initial_sampling_info,
                           sample_update_interval=sample_update_interval, updated_sample_num=updated_sample_num,
                           kept_sample_num=kept_sample_num, is_net_transformed=is_net_transformed, is_maximized=False,
                           is_trainable=False, training_sample_info=training_sample_info_rad, is_visualized=is_visualized)
    evaluation = Evaluation_Burgers(mu=mu, freq=1000, is_net_transformed=is_net_transformed)
    net = run(training, loss, net, device=device, evaluation=evaluation)
    error = evaluation.plot_results(net, is_show=False)
    print("RAD w/o Training Error: ", error)
    rad_wo_results.append(error)
    add_results_to_file(file_path=filename, content="RAD w/o Training Error: " + str(error))

rad_w_results = []
for _ in range(test_num):
    net = createnet(net_info)
    loss = LossBurgers_RAD(device=device, max_epochs=max_epochs, mu=mu, initial_sampling_info=initial_sampling_info,
                           sample_update_interval=sample_update_interval, updated_sample_num=updated_sample_num,
                           kept_sample_num=kept_sample_num, is_net_transformed=is_net_transformed, is_maximized=False,
                           is_trainable=True, training_sample_info=training_sample_info_rad, is_visualized=is_visualized)
    evaluation = Evaluation_Burgers(mu=mu, freq=1000, is_net_transformed=is_net_transformed)
    net = run(training, loss, net, device=device, evaluation=evaluation)
    error = evaluation.plot_results(net, is_show=False)
    print("RAD with Training Error: ", error)
    rad_w_results.append(error)
    add_results_to_file(file_path=filename, content="RAD with Training Error: " + str(error))


initial_sampling_info = {"nD": sample_num, "nBC1": 150, "nBC2": 150, "nIC": 150, "SamplingMethod": "Uniform"}
train_sample_iter = 5
training_sample_info_rar = {"train_sample_iter": train_sample_iter, "opt_type": "Adam", "opt_lr": 0.001,
                            "den_coef": 0.0, "sample_ratio": 0.02}


r3_wo_results = []
for _ in range(test_num):
    net = createnet(net_info)
    loss = LossBurgers_R3(device=device, max_epochs=max_epochs, mu=mu, initial_sampling_info=initial_sampling_info,
                           is_net_transformed=is_net_transformed, is_maximized=False, is_trainable=False, sample_update_interval=sample_update_interval,
                          training_sample_info=training_sample_info_rar, is_visualized=is_visualized)
    evaluation = Evaluation_Burgers(mu=mu, freq=1000, is_net_transformed=is_net_transformed)
    net = run(training, loss, net, device=device, evaluation=evaluation)
    error = evaluation.plot_results(net, is_show=False)
    print("R3 w/o Training Error: ", error)
    r3_wo_results.append(error)
    add_results_to_file(file_path=filename, content="R3 w/o Training Error: " + str(error))

r3_w_results = []
for _ in range(test_num):
    net = createnet(net_info)
    loss = LossBurgers_R3(device=device, max_epochs=max_epochs, mu=mu, initial_sampling_info=initial_sampling_info,
                           is_net_transformed=is_net_transformed, is_maximized=False, is_trainable=True, sample_update_interval=sample_update_interval,
                          training_sample_info=training_sample_info_rar, is_visualized=is_visualized)
    evaluation = Evaluation_Burgers(mu=mu, freq=1000, is_net_transformed=is_net_transformed)
    net = run(training, loss, net, device=device, evaluation=evaluation)
    error = evaluation.plot_results(net, is_show=False)
    print("R3 with Training Error: ", error)
    r3_w_results.append(error)
    add_results_to_file(file_path=filename, content="R3 with Training Error: " + str(error))

# For sampling method
sample_num = 5000
sample_update_interval = 200
# Initial sampling information
initial_sampling_info = {"nD": sample_num, "nBC1": 150, "nBC2": 150, "nIC": 150, "SamplingMethod": "Uniform"}
# Optimizer information for sampling
train_sample_iter = 15
training_sample_info = {"train_sample_interval": sample_update_interval, "train_sample_iter": train_sample_iter,
                        "sample_ratio": 0.02, "kept_ratio": 0.9,"opt_type": "Adam", "opt_lr": 0.001, "den_coef": 0.0}

random_wo_results = []
initial_sampling_info["SamplingMethod"] = "Uniform"
for _ in range(test_num):
    net = createnet(net_info)
    loss = LossBurgers(device=device, max_epochs=max_epochs, mu=mu, initial_sampling_info=initial_sampling_info,
                       sample_update_interval=None, is_net_transformed=True,
                       is_trainable=False, training_sample_info=training_sample_info, is_visualized=is_visualized)
    evaluation = Evaluation_Burgers(mu=mu, freq=1000)
    net = run(training, loss, net, device=device, evaluation=evaluation)
    error = evaluation.plot_results(net, is_show=False)
    print("Random w/o Training Error: ", error)
    random_wo_results.append(error)
    add_results_to_file(file_path=filename, content="Random w/o Training Error: " + str(error))

random_w_results = []
initial_sampling_info["SamplingMethod"] = "Uniform"
for _ in range(test_num):
    net = createnet(net_info)
    loss = LossBurgers(device=device, max_epochs=max_epochs, mu=mu, initial_sampling_info=initial_sampling_info,
                       sample_update_interval=None, is_net_transformed=True,
                       is_trainable=True, training_sample_info=training_sample_info, is_visualized=is_visualized)
    evaluation = Evaluation_Burgers(mu=mu, freq=1000)
    net = run(training, loss, net, device=device, evaluation=evaluation)
    error = evaluation.plot_results(net, is_show=False)
    print("Random with Training Error: ", error)
    random_w_results.append(error)
    add_results_to_file(file_path=filename, content="Random with Training Error: " + str(error))

# Optimizer information for sampling
train_sample_iter = 10
training_sample_info = {"train_sample_interval": sample_update_interval, "train_sample_iter": train_sample_iter,
                        "sample_ratio": 0.02, "kept_ratio": 0.9,"opt_type": "Adam", "opt_lr": 0.001, "den_coef": 0.0}
lhs_wo_results = []
initial_sampling_info["SamplingMethod"] = "lhs"
for _ in range(test_num):
    net = createnet(net_info)
    loss = LossBurgers(device=device, max_epochs=max_epochs, mu=mu, initial_sampling_info=initial_sampling_info,
                       sample_update_interval=None, is_net_transformed=True,
                       is_trainable=False, training_sample_info=training_sample_info, is_visualized=is_visualized)
    evaluation = Evaluation_Burgers(mu=mu, freq=1000)
    net = run(training, loss, net, device=device, evaluation=evaluation)
    error = evaluation.plot_results(net, is_show=False)
    print("LHS w/o Training Error: ", error)
    lhs_wo_results.append(error)
    add_results_to_file(file_path=filename, content="LHS w/o Training Error: " + str(error))

lhs_w_results = []
initial_sampling_info["SamplingMethod"] = "lhs"
for _ in range(test_num):
    net = createnet(net_info)
    loss = LossBurgers(device=device, max_epochs=max_epochs, mu=mu, initial_sampling_info=initial_sampling_info,
                       sample_update_interval=None, is_net_transformed=True,
                       is_trainable=True, training_sample_info=training_sample_info, is_visualized=is_visualized)
    evaluation = Evaluation_Burgers(mu=mu, freq=1000)
    net = run(training, loss, net, device=device, evaluation=evaluation)
    error = evaluation.plot_results(net, is_show=False)
    print("LHS with Training Error: ", error)
    lhs_w_results.append(error)
    add_results_to_file(file_path=filename, content="LHS with Training Error: " + str(error))

train_sample_iter = 10
training_sample_info = {"train_sample_interval": sample_update_interval, "train_sample_iter": train_sample_iter,
                        "sample_ratio": 0.02, "kept_ratio": 0.9,"opt_type": "Adam", "opt_lr": 0.001, "den_coef": 0.0}
halton_wo_results = []
initial_sampling_info["SamplingMethod"] = "halton"
for _ in range(test_num):
    net = createnet(net_info)
    loss = LossBurgers(device=device, max_epochs=max_epochs, mu=mu, initial_sampling_info=initial_sampling_info,
                       sample_update_interval=None, is_net_transformed=True,
                       is_trainable=False, training_sample_info=training_sample_info, is_visualized=is_visualized)
    evaluation = Evaluation_Burgers(mu=mu, freq=1000)
    net = run(training, loss, net, device=device, evaluation=evaluation)
    error = evaluation.plot_results(net, is_show=False)
    print("Halton w/o Training Error: ", error)
    halton_wo_results.append(error)
    add_results_to_file(file_path=filename, content="Halton w/o Training Error: " + str(error))

halton_w_results = []
initial_sampling_info["SamplingMethod"] = "halton"
for _ in range(test_num):
    net = createnet(net_info)
    loss = LossBurgers(device=device, max_epochs=max_epochs, mu=mu, initial_sampling_info=initial_sampling_info,
                       sample_update_interval=None, is_net_transformed=True,
                       is_trainable=True, training_sample_info=training_sample_info, is_visualized=is_visualized)
    evaluation = Evaluation_Burgers(mu=mu, freq=1000)
    net = run(training, loss, net, device=device, evaluation=evaluation)
    error = evaluation.plot_results(net, is_show=False)
    print("Halton with Training Error: ", error)
    halton_w_results.append(error)
    add_results_to_file(file_path=filename, content="Halton with Training Error: " + str(error))

train_sample_iter = 10
training_sample_info = {"train_sample_interval": sample_update_interval, "train_sample_iter": train_sample_iter,
                        "sample_ratio": 0.02, "kept_ratio": 0.9,"opt_type": "Adam", "opt_lr": 0.001, "den_coef": 0.0}
sobol_wo_results = []
initial_sampling_info["SamplingMethod"] = "sobol"
for _ in range(test_num):
    net = createnet(net_info)
    loss = LossBurgers(device=device, max_epochs=max_epochs, mu=mu, initial_sampling_info=initial_sampling_info,
                       sample_update_interval=None, is_net_transformed=True,
                       is_trainable=False, training_sample_info=training_sample_info, is_visualized=is_visualized)
    evaluation = Evaluation_Burgers(mu=mu, freq=1000)
    net = run(training, loss, net, device=device, evaluation=evaluation)
    error = evaluation.plot_results(net, is_show=False)
    print("Sobol w/o Training Error: ", error)
    sobol_wo_results.append(error)
    add_results_to_file(file_path=filename, content="Sobol w/o Training Error: " + str(error))

sobol_w_results = []
initial_sampling_info["SamplingMethod"] = "sobol"
for _ in range(test_num):
    net = createnet(net_info)
    loss = LossBurgers(device=device, max_epochs=max_epochs, mu=mu, initial_sampling_info=initial_sampling_info,
                       sample_update_interval=None, is_net_transformed=True,
                       is_trainable=True, training_sample_info=training_sample_info, is_visualized=is_visualized)
    evaluation = Evaluation_Burgers(mu=mu, freq=1000)
    net = run(training, loss, net, device=device, evaluation=evaluation)
    error = evaluation.plot_results(net, is_show=False)
    print("Sobol with Training Error: ", error)
    sobol_w_results.append(error)
    add_results_to_file(file_path=filename, content="Sobol with Training Error: " + str(error))

"""==============================================================================================================
                                        Summary of the results
=============================================================================================================="""
try:
    print("=" * 40)
    print("RAR w/o Training Results: ", rar_wo_results)
    print("Average RAR w/o Training Results: ", sum(rar_wo_results) / len(rar_wo_results))
    print("RAR w/o Training Results Std.: ", np.std(rar_wo_results))
    print("RAR with Training Results: ", rar_w_results)
    print("Average RAR with Training Results: ", sum(rar_w_results) / len(rar_w_results))
    print("RAR with Training Results Std.: ", np.std(rar_w_results))
except:
    pass

try:
    print("="*40)
    print("RAD w/o Training Results: ", rad_wo_results)
    print("Average RAD w/o Training Results: ", sum(rad_wo_results) / len(rad_wo_results))
    print("RAD w/o Training Results Std.: ", np.std(rad_wo_results))
    print("RAD with Training Results: ", rad_w_results)
    print("Average RAD with Training Results: ", sum(rad_w_results) / len(rad_w_results))
    print("RAD with Training Results Std.: ", np.std(rad_w_results))
except:
    pass

try:
    print("=" * 40)
    print("R3 w/o Training Results: ", r3_wo_results)
    print("Average R3 w/o Training Results: ", sum(r3_wo_results) / len(r3_wo_results))
    print("R3 w/o Training Results Std.: ", np.std(r3_wo_results))
    print("R3 with Training Results: ", r3_w_results)
    print("Average R3 with Training Results: ", sum(r3_w_results) / len(r3_w_results))
    print("R3 with Training Results Std.: ", np.std(r3_w_results))
except:
    pass

try:
    print("=" * 40)
    print("Random w/o Training Results: ", random_wo_results)
    print("Average Random w/o Training Results: ", sum(random_wo_results) / len(random_wo_results))
    print("Random w/o Training Results Std.: ", np.std(random_wo_results))
    print("Random with Training Results: ", random_w_results)
    print("Average Random with Training Results: ", sum(random_w_results) / len(random_w_results))
    print("Random with Training Results Std.: ", np.std(random_w_results))
except:
    pass

try:
    print("=" * 40)
    print("LHS w/o Training Results: ", lhs_wo_results)
    print("Average LHS w/o Training Results: ", sum(lhs_wo_results) / len(lhs_wo_results))
    print("LHS w/o Training Results Std.: ", np.std(lhs_wo_results))
    print("LHS with Training Results: ", lhs_w_results)
    print("Average LHS with Training Results: ", sum(lhs_w_results) / len(lhs_w_results))
    print("LHS with Training Results Std.: ", np.std(lhs_w_results))
except:
    pass

try:
    print("="*40)
    print("Halton w/o Training Results: ", halton_wo_results)
    print("Average Halton w/o Training Results: ", sum(halton_wo_results) / len(halton_wo_results))
    print("Halton w/o Training Results Std.: ", np.std(halton_wo_results))
    print("Halton with Training Results: ", halton_w_results)
    print("Average Halton with Training Results: ", sum(halton_w_results) / len(halton_w_results))
    print("Halton with Training Results Std.: ", np.std(halton_w_results))
except:
    pass

try:
    print("="*40)
    print("Sobol w/o Training Results: ", sobol_wo_results)
    print("Average Sobol w/o Training Results: ", sum(sobol_wo_results) / len(sobol_wo_results))
    print("Sobol w/o Training Results Std.: ", np.std(sobol_wo_results))
    print("Sobol with Training Results: ", sobol_w_results)
    print("Average Sobol with Training Results: ", sum(sobol_w_results) / len(sobol_w_results))
    print("Sobol with Training Results Std.: ", np.std(sobol_w_results))
except:
    pass






