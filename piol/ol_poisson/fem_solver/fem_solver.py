import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import example.ol_heat_transfer.fem_solver.element.quaelement as element
# import example.ol_heat_transfer.fem_solver.element.trielement as trielement


def get_node(x_len, y_len, num_x, num_y):
    # Generate node coordinates correctly
    node = [[j * x_len / num_x, i * y_len / num_y] for i in range(num_y + 1) for j in range(num_x + 1)]
    return node

def get_mesh(num_x, num_y):
    # Generate mesh connectivity correctly
    mesh = []
    for i in range(num_y):
        for j in range(num_x):
            mesh.append([i * (num_x + 1) + j, i * (num_x + 1) + j + 1, (i + 1) * (num_x + 1) + j + 1, (i + 1) * (num_x + 1) + j])
    return mesh

def solver(node, mesh, T_BC, T_nodeid, q_BC, q_nodeid, k_list, element_type='quad4'):
    # if element_type == 'quad4':
    #     element = quaelement
    #     ele_node_num = 4
    # elif element_type == 'tri3':
    #     element = trielement
    #     ele_node_num = 3
    ele_node_num = 4
    # Initialize the sparse matrix directly
    row_indices = []
    col_indices = []
    data_values = []

    # Assembly of global stiffness matrix
    for z in range(len(mesh)):
        ele = mesh[z]
        tk = k_list[z]
        ele_coords = np.array([node[i] for i in ele])
        ele_k = element.get_elek(ele_coords, tk)
        for i in range(ele_node_num):
            for j in range(ele_node_num):
                row_indices.append(ele[i])
                col_indices.append(ele[j])
                data_values.append(ele_k[i, j])

    # Create CSR matrix from the data
    K = csr_matrix((data_values, (row_indices, col_indices)), shape=(len(node), len(node)))

    # Apply temperature boundary conditions
    F = np.zeros(len(node))
    # Apply heat flux boundary conditions
    F[q_nodeid] += q_BC
    average_k = np.mean(np.abs(K.diagonal()))
    beta = 1e7 * average_k
    big_indices = np.array(T_nodeid, dtype=int)
    K[big_indices, big_indices] = beta
    F[big_indices] = beta * np.array(T_BC)

    # Solve the linear system
    T = spsolve(K, F)
    return T

def get_fourside_nodeid(num_x, num_y):
    left_nodeid = [i * (num_x + 1) for i in range(num_y + 1)]
    right_nodeid = [(i + 1) * (num_x + 1) - 1 for i in range(num_y + 1)]
    top_nodeid = [i + (num_x + 1) * num_y for i in range(num_x + 1)]
    bottom_nodeid = [i for i in range(num_x + 1)]
    return left_nodeid, right_nodeid, top_nodeid, bottom_nodeid

def get_eleloc(node, mesh):
    node = np.array(node)
    mesh = np.array(mesh)
    ele_loc = []
    for ele in mesh:
        ele_coords = node[ele]
        ele_center = np.mean(ele_coords, axis=0)
        ele_loc.append(ele_center)
    return np.array(ele_loc)

def get_k_list(node, mesh, k_max=1, k_min=0.5, b=0.3):
    ele_loc = get_eleloc(node, mesh)
    k_list = np.ones(len(mesh)) * k_max
    # For ele_loc within [-b, b]^2, set k = k_min
    for i in range(len(ele_loc)):
        if -b <= ele_loc[i][0] <= b and -b <= ele_loc[i][1] <= b:
            k_list[i] = k_min
    return k_list

def get_results(q_BC=0, num_x=50, num_y=50, b=0.3):
    x_len = 2.0
    y_len = 2.0
    num_x = num_x - 1
    num_y = num_y - 1
    ele_area = x_len * y_len / num_x / num_y
    node = get_node(x_len, y_len, num_x, num_y)
    node = np.array(node)
    # tnode = np.array(node)
    # node = np.zeros_like(tnode)
    # node[:, 0] = tnode[:, 1]
    # node[:, 1] = tnode[:, 0]
    node = node - np.array([1, 1])
    mesh = get_mesh(num_x, num_y)

    # Define boundary conditions
    left_nodeid = [i * (num_x + 1) for i in range(num_y + 1)]
    right_nodeid = [(i + 1) * (num_x + 1) - 1 for i in range(num_y + 1)]
    top_nodeid = [i + (num_x + 1) * num_y for i in range(num_x + 1)]
    bottom_nodeid = [i for i in range(num_x + 1)]
    outline_nodes = bottom_nodeid + right_nodeid + top_nodeid + left_nodeid
    # Delete the repeated nodes in the outline_nodes
    outline_nodes = list(set(outline_nodes))

    T_nodeid = outline_nodes
    T_BC = np.zeros(len(T_nodeid))
    k_list = get_k_list(node, mesh, b=b)

    q_nodeid = [i for i in range(len(node))]
    q_BC = ele_area * q_BC
    # q_mag = 1 / num_x / num_y / x_len / y_len
    # q_BC = np.ones(len(q_nodeid)) * q_mag  # Evenly distribute heat flux
    # q_BC = q_BC * (x_len / num_x) * (y_len / num_y)
    T = solver(node, mesh, T_BC, T_nodeid, q_BC, q_nodeid, k_list=k_list)
    return node, T

if __name__ == '__main__':
    num_x = num_y = 100
    q_BC = np.ones(num_x * num_y)
    node, T = get_results(q_BC, num_x, num_y,0.3)
    # Plotting
    plt.figure()
    plt.tricontourf([n[0] for n in node], [n[1] for n in node], T, levels=100)
    plt.colorbar()
    plt.title("Temperature Distribution")
    plt.xlabel("X")
    plt.ylabel("Y")
    # adjust the aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')
    # plot the contour lines
    # plt.tricontour([n[0] for n in node], [n[1] for n in node], T, colors='k', levels=10)
    plt.show()
