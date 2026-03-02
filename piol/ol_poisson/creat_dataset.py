from grf import grf_2D
from fem_solver.fdm_solver import main as get_results
import numpy as np


def create_dataset2(sam_num, l=0.3, mesh_num=50, k_min=0.5, b=0.3):
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, mesh_num), np.linspace(-1, 1, mesh_num))
    node1 = np.array([grid_y.flatten(), grid_x.flatten()]).T
    node1 = node1.tolist()
    node1 = np.array(node1)
    random_field, _ = grf_2D(node1, l, sam_num)
    scaler_func = np.exp(-2 * np.linalg.norm(node1, axis=1) ** 2)
    scaler_func = np.array(scaler_func)
    scaler_func = np.tile(scaler_func, (sam_num, 1))
    random_field = random_field * scaler_func
    random_field_norm = np.linalg.norm(random_field, axis=1) / (mesh_num * mesh_num) ** 0.5 / 2
    random_field = random_field / random_field_norm[:, None]
    results = []
    for i in range(sam_num):
        print("Creating dataset: ", i + 1, "/", sam_num)
        q_BC = random_field[i]
        q_BC = np.array(q_BC).reshape(mesh_num, mesh_num)
        node, U = get_results(q_BC, N=mesh_num, b=b, k_min=k_min)
        results.append(U)
    # Save the results
    results = np.array(results)
    np.savez("dataset2.npz", branch_input=random_field, trunk_input=node, u_solution=results)

create_dataset2(50, k_min=0.5, b=0.3)
