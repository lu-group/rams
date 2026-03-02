import torch.nn
from training import train
from fcnn import FCNN
from loss_random import Loss_Random
from evaluation import Evaluation_HignDimPossion, get_results

def add_results_to_file(file_path, content):
    with open(file_path, "a") as file:
        file.write("\n" + content)

def random_training(net_name, D, a=10, max_epochs=40000, sample_num=10000, file_path="result_sum.txt"):
    is_visualized = True
    # net_info = {"Name": net_name, "Type": "FCNN", "ActivationFunc": "Tanh",
    #             "InputSize": D, "OutputSize": 1, "HiddenSizes": [100, 100, 100, 100, 100]}
    is_net_transformed = False
    training = {"Type": "StandardPI", "MaxEpochs": max_epochs + 10000, "LearningRate": 0.001,
                "isLBFGS": False, "LBFGSMaxEpochs": 500}
    device = None
    # Test for resampling method
    sample_update_interval = 500
    nBC = 1000
    initial_sampling_info = {"nD": sample_num, "nBC": nBC, "SamplingMethod": "Uniform"}
    train_sample_iter = 100
    training_sample_info_random = {"train_sample_interval": sample_update_interval,
                                   "train_sample_iter": train_sample_iter,
                                   "sample_ratio": 0.05, "kept_ratio": 0.5, "opt_type": "Adam", "opt_lr": 0.01,
                                   "den_coef": 0.0}
    # net = createnet(net_info)
    net = FCNN(act_fn=torch.nn.Tanh(), input_size=D, output_size=1, hidden_sizes=[100, 100, 100], name=net_name)
    loss = Loss_Random(a=a, D=D, device=device, max_epochs=max_epochs, initial_sampling_info=initial_sampling_info,
                       sample_update_interval=None, is_net_transformed=is_net_transformed, samBC_update_interval=1000,
                       is_trainable=False, training_sample_info=training_sample_info_random,
                       is_visualized=is_visualized)
    evaluation = Evaluation_HignDimPossion(a=a, D=D, mesh_size=3, freq=1000, is_net_transformed=is_net_transformed)
    net = train(training, loss, net, device=device, evaluation=evaluation)
    error = get_results(net, D)
    added_content = f"Sampling: Random, D: {D}, Error: {error}"
    add_results_to_file(file_path, added_content)
    return error

def random_trainable(net_name, D, train_sample_iter=100, a=10, max_epochs=1, sample_num=10000, file_path="result_sum.txt"):
    is_visualized = False
    # net_info = {"Name": net_name, "Type": "FCNN", "ActivationFunc": "Tanh",
    #             "InputSize": D, "OutputSize": 1, "HiddenSizes": [100, 100, 100, 100, 100]}
    is_net_transformed = False
    training = {"Type": "StandardPI", "MaxEpochs": max_epochs + 10000, "LearningRate": 0.001,
                "isLBFGS": True, "LBFGSMaxEpochs": 1}
    device = None
    # Test for resampling method
    sample_update_interval = 500
    nBC = 10000
    initial_sampling_info = {"nD": sample_num, "nBC": nBC, "SamplingMethod": "Uniform"}
    training_sample_info_random = {"train_sample_interval": sample_update_interval,
                                   "train_sample_iter": train_sample_iter,
                                   "sample_ratio": 0.1, "kept_ratio": 0.5, "opt_type": "Adam", "opt_lr": 0.01,
                                   "den_coef": 0.0}
    # net = createnet(net_info)
    net = FCNN(act_fn=torch.nn.Tanh(), input_size=D, output_size=1, hidden_sizes=[100, 100, 100],
               name=net_name)
    loss = Loss_Random(a=a, D=D, device=device, max_epochs=max_epochs, initial_sampling_info=initial_sampling_info,
                       sample_update_interval=None, is_net_transformed=is_net_transformed, samBC_update_interval=500,
                       is_trainable=True, training_sample_info=training_sample_info_random,
                       is_visualized=is_visualized)
    evaluation = Evaluation_HignDimPossion(a=a, D=D, mesh_size=3, freq=1000, is_net_transformed=is_net_transformed)
    net = train(training, loss, net, device=device, evaluation=evaluation)
    torch.save(net, net_name + ".pt")
    error = get_results(net, D)
    added_content = f"Trainable Sampling: Random, D: {D}, Error: {error}"
    add_results_to_file(file_path, added_content)
    return error

def run():
    sample_num = 20000
    for D in range(2, 10):
        net_name = "burgers_random_D=" + str(D)
        error = random_training(net_name, D)
        print(f"Random, D={D}, error={error}")
        net_name = "burgers_random_trainable_D=" + str(D)
        train_sample_iter = 100
        error = random_trainable(net_name, D, sample_num=sample_num, train_sample_iter=train_sample_iter)
        print(f"Random (with RAMS), D={D}, error={error}")
if __name__ == '__main__':
    run()

