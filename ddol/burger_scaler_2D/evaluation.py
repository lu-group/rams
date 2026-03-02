import torch
import numpy as np
import os

def logging_results(file_path, content, end=" "):
    with open(file_path, "a") as file:
        file.write("\n" + content + end)

def evaluation_val(net, dataset, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    branch_input = dataset["branch_input"]
    trunk_input = dataset["trunk_input"]
    label = dataset["label"]
    branch_input = torch.tensor(branch_input, dtype=torch.float32).to(device)
    trunk_input = torch.tensor(trunk_input, dtype=torch.float32).to(device)
    label = torch.tensor(label, dtype=torch.float32).to(device)
    net = net.to(device)
    pred = net(branch_input, trunk_input)
    diff = pred - label
    diff = diff.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    diff_norm = np.linalg.norm(diff, axis=1)
    label_norm = np.linalg.norm(label, axis=1)
    mean_error = np.mean(diff_norm / label_norm)
    return mean_error

def evaluation(resampling_time=1, sam_num_list=[3000]):
    data_path = r"datasets/"
    dataset_name = 'validation_dataset.npz'
    dataset = np.load(data_path + dataset_name)
    model_path = "resampling_time_" + str(resampling_time) + r"results_nnmodels/"
    logging_file_name = "resampling_time_" + str(resampling_time) + "_results.txt"

    resampling_portion = [0.1, 0.2, 0.4]

    for sample_num in sam_num_list:
        print("Sample num: ", sample_num)
        logging_results(logging_file_name, "=" * 40)
        logging_results(logging_file_name, "Sample num: " + str(sample_num))
        print("Base line, ", end="")
        logging_results(logging_file_name, "Base line: ", end="")
        for i in range(3):
            model_name = "random_samnum_{}_rep_{}".format(sample_num, i)
            try:
                net = torch.load(model_path + model_name + ".pth")
                result = evaluation_val(net, dataset)
            except:
                result = None
            print(result, end=" ,")
            logging_results(logging_file_name, str(result))
        print()
        for portion in resampling_portion:
            print("Trainable (p= " + str(portion) + "), ", end="")
            logging_results(logging_file_name, "Trainable (p= " + str(portion) + "), ", end="")
            for i in range(3):
                model_name = "rar_samnum_{}_portion_{}_rep_{}".format(sample_num, portion, i)
                try:
                    net = torch.load(model_path + model_name + ".pth")
                    result = evaluation_val(net, dataset)
                except:
                    result = None
                print(result, end=" ,")
                logging_results(logging_file_name, str(result))
            print()


if __name__ == '__main__':
    evaluation(2, [1500, 2000, 2500, 3000])