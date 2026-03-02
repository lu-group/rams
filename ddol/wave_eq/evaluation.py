import torch
import numpy as np

def evaluation_val(net, dataset):
    branch_input = dataset["branch_input"]
    trunk_input = dataset["trunk_input"]
    label = dataset["results"]
    branch_output = net.branch_net(torch.tensor(branch_input, dtype=torch.float32))
    trunk_output = net.trunk_net(torch.tensor(trunk_input, dtype=torch.float32))
    pred = torch.matmul(branch_output, trunk_output.T).detach().numpy()
    diff_norm = np.linalg.norm(pred - label, axis=1) / np.linalg.norm(label, axis=1)
    mean_error = np.mean(diff_norm)
    # loss = np.mean((pred - label) ** 2) / np.mean(label ** 2)
    # print("The loss is %.4e" % mean_error)
    return mean_error

def run():
    data_path = r"datasets\\"
    dataset_name = 'testing_dataset2.npz'
    dataset = np.load(data_path + dataset_name)
    model_path = r"nnmodels_50sam\\"
    # model_name = "trainable_rar_sample_iter_800.pth"
    # model_name = "random_sample.pth"
    for iter in [0, 100, 200, 300, 400, 500, 600]:
        results = []
        print("For iteration %d:" % iter, end=' ')
        for i in range(3):
            model_name = "test_trainable_rar_sample_iter_" + str(iter) + "_repeat_" + str(i) + ".pth"
            file_name = model_path + model_name
            # file_name = r"cache_models/trainable_rar_sample_epoch_10000_loss_1.22e-02.pth"
            try:
                net = torch.load(file_name, map_location=torch.device('cpu'))
                test_loss = evaluation_val(net, dataset)
            except:
                test_loss = None
            results.append(test_loss)
            print(test_loss, end=' ')
        print()
    results = []
    print("Random samling:", end=' ')
    for i in range(3):
        model_name = "trandom_sample_rep_" + str(i) + ".pth"
        # model_name = "random_sample_rep_0.pth"
        file_name = model_path + model_name
        # net = torch.load(file_name, map_location=torch.device('cpu'))
        try:
            net = torch.load(file_name, map_location=torch.device('cpu'))
            test_loss = evaluation_val(net, dataset)
        except:
            test_loss = None
        results.append(test_loss)
        print(test_loss, end=' ')
if __name__ == '__main__':
    pass
    # data_path = r"datasets\\"
    # dataset_name = 'testing_dataset2.npz'
    # dataset = np.load(data_path + dataset_name)
    # model_path = r"nnmodels_50sam\\"
    # # model_name = "trainable_rar_sample_iter_800.pth"
    # # model_name = "random_sample.pth"
    # for iter in [0,100,200,300,400,500,600]:
    #     results = []
    #     print("For iteration %d:" % iter, end=' ')
    #     for i in range(3):
    #         model_name = "test_trainable_rar_sample_iter_" + str(iter) + "_repeat_" + str(i) + ".pth"
    #         file_name = model_path + model_name
    #         # file_name = r"cache_models/trainable_rar_sample_epoch_10000_loss_1.22e-02.pth"
    #         try:
    #             net = torch.load(file_name, map_location=torch.device('cpu'))
    #             test_loss = evaluation_val(net, dataset)
    #         except:
    #             test_loss = None
    #         results.append(test_loss)
    #         print(test_loss, end=' ')
    #     print()
    # results = []
    # print("Random samling:", end=' ')
    # for i in range(3):
    #     model_name = "trandom_sample_rep_" + str(i) + ".pth"
    #     # model_name = "random_sample_rep_0.pth"
    #     file_name = model_path + model_name
    #     # net = torch.load(file_name, map_location=torch.device('cpu'))
    #     try:
    #         net = torch.load(file_name, map_location=torch.device('cpu'))
    #         test_loss = evaluation_val(net, dataset)
    #     except:
    #         test_loss = None
    #     results.append(test_loss)
    #     print(test_loss, end=' ')