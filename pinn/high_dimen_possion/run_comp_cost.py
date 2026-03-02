from run_rel_err import random_training, random_trainable
import time

def add_results_to_file(file_path, content):
    with open(file_path, "a") as file:
        file.write("\n" + content)

def get_comp_cost_random(net_name, start_sam_num, max_sam_num, D, max_epochs=30000, file_path="resultv2",
                         result_sum_file="comp_cost_sum"):
    file_path = file_path + "_random"
    result_sum_file = result_sum_file + "_random"
    sam_num = start_sam_num
    add_results_to_file(file_path, f"Sampling: Random, D: {D}")
    add_results_to_file(file_path, "=" * 30)
    while True:
        startT = time.time()
        error = random_training(net_name, D, sample_num=sam_num, max_epochs=max_epochs, file_path=file_path)
        endT = time.time()
        if error > 1e-3:
            sam_num = int(sam_num * 2)
            add_results_to_file(file_path, f"Sample number: {sam_num}, Error: {error}")
        else:
            comp_time = endT - startT
            add_results_to_file(file_path, f"Sample number: {sam_num}, Error: {error}, Time: {comp_time}")
            add_results_to_file(result_sum_file, f"Sampling: Random, D: {D}, Sample number: {sam_num}, Error: {error}, Time: {comp_time}")
            break
        if sam_num > max_sam_num:
            comp_time = None
            add_results_to_file(file_path, f"D: {D}, Failed to converge.")
            add_results_to_file(result_sum_file, f"Sampling: Random, D: {D}, Failed to converge.")
            break
    return comp_time, sam_num

def get_comp_cost_trainable(net_name, D, start_samp_iter,  start_training_epoch, current_indicator, max_indicator=20, file_path="resultv2",
                            result_sum_file="comp_cost_sum", iter_step=20, training_epoch_step=5000):
    sam_iter = start_samp_iter
    training_epoch = start_training_epoch
    file_path = file_path + "_random_trainable"
    result_sum_file = result_sum_file + "_random_trainable"
    add_results_to_file(file_path, f"Sampling: Trainable Random, D: {D}")
    add_results_to_file(file_path, "=" * 30)
    while True:
        startT = time.time()
        error = random_trainable(net_name, D, sample_num=20000, train_sample_iter=sam_iter, max_epochs=training_epoch, file_path=file_path)
        endT = time.time()
        if error > 1e-3:
            sam_iter += iter_step
            training_epoch += training_epoch_step
            current_indicator += 1
            add_results_to_file(file_path, f"Sample Iter: {sam_iter}, Training Iter: {training_epoch},  Error: {error}")
        else:
            comp_time = endT - startT
            add_results_to_file(file_path, f"Sample Iter: {sam_iter}, Error: {error}, Time: {comp_time}")
            add_results_to_file(result_sum_file, f"Sampling: Trainable Random, D: {D}, Sample Iter: {sam_iter}, Training Iter: {training_epoch}, Error: {error}, Time: {comp_time}")
            break
        if current_indicator > max_indicator:
            comp_time = None
            add_results_to_file(file_path, f"D: {D}, Failed to converge.")
            add_results_to_file(result_sum_file, f"Sampling: Trainable Random, D: {D}, Failed to converge.")
            break
    return comp_time, sam_iter, training_epoch, current_indicator

def run():
    start_sam_num = 100
    print("Start sampling number: ", start_sam_num)
    max_sam_num = start_sam_num * 2 ** 16
    for D in range(2, 7):
        net_name = f"results/comp_cost_random_D={D}.pth"
        comp_time, sam_num = get_comp_cost_random(net_name, start_sam_num, max_sam_num, D)
        print(f"Random sampling, D: {D}, Sample number: {sam_num}, Time: {comp_time}")
        start_sam_num = sam_num
        if comp_time is None:
            break

    start_samp_iter = 0
    start_training_epoch = 5000
    current_indicator = 0
    for D in range(2, 11):
        net_name = f"results/comp_cost_random_trainable_D={D}.pth"
        comp_time, start_samp_iter, start_training_epoch, current_indicator = (
            get_comp_cost_trainable(net_name=net_name, D=D, start_samp_iter=start_samp_iter,
                                    start_training_epoch=start_training_epoch, current_indicator=current_indicator))
        print(
            f"Trainable Random sampling, D: {D}, Sample Iter: {start_samp_iter}, Training Iter: {start_training_epoch} Time: {comp_time}")
        if comp_time is None:
            break

if __name__ == '__main__':
    # start_sam_num = 100
    # print("Start sampling number: ", start_sam_num)
    # max_sam_num = start_sam_num * 2 ** 16
    # for D in range(2, 7):
    #     net_name = f"results/comp_cost_random_D={D}.pth"
    #     comp_time, sam_num = get_comp_cost_random(net_name, start_sam_num, max_sam_num, D)
    #     print(f"Random sampling, D: {D}, Sample number: {sam_num}, Time: {comp_time}")
    #     start_sam_num = sam_num
    #     if comp_time is None:
    #         break
    #
    # start_samp_iter = 0
    # start_training_epoch = 5000
    # current_indicator = 0
    # for D in range(2, 11):
    #     net_name = f"results/comp_cost_random_trainable_D={D}.pth"
    #     comp_time, start_samp_iter, start_training_epoch, current_indicator = (
    #         get_comp_cost_trainable(net_name=net_name, D=D, start_samp_iter=start_samp_iter,
    #                                 start_training_epoch=start_training_epoch, current_indicator=current_indicator))
    #     print(f"Trainable Random sampling, D: {D}, Sample Iter: {start_samp_iter}, Training Iter: {start_training_epoch} Time: {comp_time}")
    #     if comp_time is None:
    #         break
    run()