import numpy as np
from tqdm import tqdm

def solver(u0, nx=50, ny=50, nt=5000, dt=1e-3, nu=0.05, recorded_interval=1000):
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)

    u = u0

    # X, Y = numpy.meshgrid(x, y)

    U = []
    U.append(u.copy())

    B1 = u[0, :]
    B2 = u[-1, :]
    B3 = u[:, 0]
    B4 = u[:, -1]

    # Initialize the progress bar
    progress_bar = tqdm(range(nt))

    # Time marching loop
    for step in progress_bar:
        un = u.copy()
        # Backward Difference Scheme for Convection Term
        # Central Difference Scheme for Diffusion Term

        u[1:-1, 1:-1] = (un[1:-1, 1:-1] - dt / dx * un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         dt / dy * un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) +
                         nu * dt / dx ** 2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         nu * dt / dy ** 2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))

        # boundary condition

        u[0, :] = B1
        u[-1, :] = B2
        u[:, 0] = B3
        u[:, -1] = B4
        if (step + 1) % recorded_interval == 0:
            U.append(u.copy())
            if np.any(np.isnan(U)):
                break

    return U

def solver_maccormack(u0, nx=50, ny=50, nt=5000, dt=1e-3, nu=0.05, recorded_interval=1000, is_qbar=False):
    """
    Solve 2D Burger's equation with MacCormack scheme for the
    convection terms and explicit central-difference for the
    diffusion terms.

    Parameters:
    -----------
    u0 : 2D numpy.ndarray
        Initial condition for the velocity field (scalar in this example).
    nx, ny : int
        Number of grid points in x and y directions.
    nt : int
        Number of time steps.
    dt : float
        Time step size.
    nu : float
        Diffusion coefficient (viscosity).
    recorded_interval : int
        Interval for saving (recording) solutions.

    Returns:
    --------
    U : list of 2D numpy.ndarrays
        List of solutions (snapshots) at certain time intervals.
    """
    # Grid spacing
    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)

    # Copy the initial condition
    u = u0.copy()

    # Save the boundary values (assuming they do not change in time)
    B1 = u[0, :].copy()    # top
    B2 = u[-1, :].copy()   # bottom
    B3 = u[:, 0].copy()    # left
    B4 = u[:, -1].copy()   # right

    # Container for recorded solutions
    U = []
    U.append(u.copy())

    # Time stepping
    if is_qbar:
        progress_bar = tqdm(range(nt), desc="Time Steps")
    else:
        progress_bar = range(nt)
    for step in progress_bar:
        un = u.copy()

        #--------------------------
        # 1) Predictor step
        #--------------------------
        # Forward difference for convective derivatives in x, y
        # (equivalent to a forward-in-time, forward-in-space approach)
        # We'll also add the diffusion term explicitly here.

        # Create an array for the predictor
        u_pred = un.copy()

        # Interior updates (predictor)
        u_pred[1:-1, 1:-1] = (
            un[1:-1, 1:-1]
            # Advection in x (forward difference in space)
            - dt/dx * un[1:-1, 1:-1] * (un[1:-1, 2:] - un[1:-1, 1:-1])
            # Advection in y (forward difference in space)
            - dt/dy * un[1:-1, 1:-1] * (un[2:, 1:-1] - un[1:-1, 1:-1])
            # Diffusion in x
            + nu * dt/dx**2 * (un[1:-1, 2:] - 2.0*un[1:-1, 1:-1] + un[1:-1, 0:-2])
            # Diffusion in y
            + nu * dt/dy**2 * (un[2:, 1:-1] - 2.0*un[1:-1, 1:-1] + un[0:-2, 1:-1])
        )

        #--------------------------
        # 2) Corrector step
        #--------------------------
        # Backward difference for convective derivatives (using u_pred)
        # Then average with the predictor (MacCormack).
        u_corr = un.copy()

        # Interior updates (corrector)
        u_corr[1:-1, 1:-1] = (
            0.5 * (
                # average of the old solution ...
                un[1:-1, 1:-1]
                # ... and the predictor
                + u_pred[1:-1, 1:-1]
            )
            - 0.5 * dt/dx * u_pred[1:-1, 1:-1] * (u_pred[1:-1, 1:-1] - u_pred[1:-1, 0:-2])
            - 0.5 * dt/dy * u_pred[1:-1, 1:-1] * (u_pred[1:-1, 1:-1] - u_pred[0:-2, 1:-1])
            + 0.5 * nu * dt/dx**2 * (u_pred[1:-1, 2:] - 2.0*u_pred[1:-1, 1:-1] + u_pred[1:-1, 0:-2])
            + 0.5 * nu * dt/dy**2 * (u_pred[2:, 1:-1] - 2.0*u_pred[1:-1, 1:-1] + u_pred[0:-2, 1:-1])
        )

        # Update the solution with the corrector
        u = u_corr

        # Enforce boundary conditions
        u[0, :]  = B1
        u[-1, :] = B2
        u[:, 0]  = B3
        u[:, -1] = B4

        # Record solution
        if (step + 1) % recorded_interval == 0:
            U.append(u.copy())

    return U

def interpolation(original_field, original_nodes, updated_nodes):
    x = updated_nodes[:, 0]
    y = updated_nodes[:, 1]
    n_x = n_y = int(np.sqrt(original_nodes.shape[0]))
    seg_length_x = 2 / (n_x - 1)
    seg_length_y = 2 / (n_y - 1)
    index_x = (x + 1) // seg_length_x
    index_y = (y + 1) // seg_length_y
    index_x = index_x.astype(np.int64)
    index_y = index_y.astype(np.int64)
    end_idx_x = np.where(index_x == n_x - 1)
    end_idx_y = np.where(index_y == n_y - 1)
    index_x[end_idx_x] = n_x - 2
    index_y[end_idx_y] = n_y - 2
    u1_index = index_x + index_y * n_x
    u2_index = index_x + 1 + index_y * n_x
    u3_index = index_x + 1 + (index_y + 1) * n_x
    u4_index = index_x + (index_y + 1) * n_x
    u1_index = u1_index.astype(np.int64)
    u2_index = u2_index.astype(np.int64)
    u3_index = u3_index.astype(np.int64)
    u4_index = u4_index.astype(np.int64)
    node1 = original_nodes[u1_index]
    u1 = original_field[:, u1_index]
    u2 = original_field[:, u2_index]
    u3 = original_field[:, u3_index]
    u4 = original_field[:, u4_index]
    x_prime = x - node1[:, 0]
    y_prime = y - node1[:, 1]
    # f = a * x' + b * y' + c*x'*y' + d
    a = (u2 - u1) / seg_length_x
    b = (u4 - u1) / seg_length_y
    c = (u1 - u2 - u4 + u3) / (seg_length_x * seg_length_y)
    d = u1
    f = a * x_prime + b * y_prime + c * x_prime * y_prime + d
    return f

def create_dataset(filename, sam_num, l=0.3, nu=0.02):
    nx = 64
    ny = 64
    nt = 10000
    recorded_nt = 100
    tmax = 1
    dt = tmax / nt
    x = np.linspace(-1, 1, nx)  # Coordinate Along X direction
    y = np.linspace(-1, 1, ny) # Coordinate Along Y direction
    X, Y = np.meshgrid(x, y)
    orignal_node = np.stack((X.flatten(), Y.flatten()), axis=1)
    from grf import grf_2D
    u0_list, _ = grf_2D(orignal_node, l, sam_num, std=1, L=None, is_torch=False, device=None)
    u0_list = np.array(u0_list)
    u0_list = u0_list * (1 - X.flatten() ** 2) * (1 - Y.flatten() ** 2) / 2
    results_u = []

    nx2 = ny2 = 128

    for i in range(sam_num):
        print(i)
        tnx2 = nx2
        tny2 = ny2
        grid_X2 = np.linspace(0, 2, tnx2) - 1
        grid_Y2 = np.linspace(0, 2, tny2) - 1
        X2, Y2 = np.meshgrid(grid_X2, grid_Y2)
        updated_nodes = np.stack((X2.flatten(), Y2.flatten()), axis=1)
        u0_2 = interpolation(u0_list[i].reshape(1,-1), orignal_node, updated_nodes)
        u0_2 = u0_2.reshape(nx2, ny2)
        U = solver_maccormack(u0_2.copy(), nx=tnx2, ny=tny2, nt=nt, dt=dt, nu=nu, recorded_interval=int(nt / recorded_nt))
        U = np.array(U)
        scaler = 2
        while np.any(np.isnan(U)):
            print("solver failed, try again with more steps. scaler: ", scaler)
            U = solver_maccormack(u0_2.copy(), nx=tnx2, ny=tny2, nt=int(scaler * nt), dt=dt/scaler, nu=nu, recorded_interval=int(scaler * nt / recorded_nt))
            U = np.array(U)
            if np.any(np.isnan(U)):
                scaler = scaler * 2
                print("solver failed, try again with more steps. scaler: ", scaler)
        U = U[:, ::int(tnx2 / nx), ::int(tny2 / ny)]
        U = U.reshape(-1)
        results_u.append(U)
    results_u = np.array(results_u)
    t_list = [i * tmax / recorded_nt for i in range(recorded_nt + 1)]
    # Trunk net input: concatenate node and t_list
    trunk_input = []
    for t in t_list:
        t_input = np.ones(len(orignal_node)) * t
        trunk_input.append(np.concatenate((orignal_node, t_input[:, None]), axis=1))
    trunk_input = np.stack(trunk_input, axis=0).reshape(-1, 3)
    # Save the dataset to .npz
    np.savez(filename, node=orignal_node, trunk_input=trunk_input, label=results_u, branch_input=np.array(u0_list))
    return trunk_input, results_u

def merge_dataset(filename, dataset1, dataset2):
    dataset1 = np.load(dataset1)
    dataset2 = np.load(dataset2)
    node = np.concatenate((dataset1["node"], dataset2["node"]), axis=0)
    trunk_input = dataset1["trunk_input"]
    label = np.concatenate((dataset1["label"], dataset2["label"]), axis=0)
    branch_input = np.concatenate((dataset1["branch_input"], dataset2["branch_input"]), axis=0)
    np.savez(filename, node=node, trunk_input=trunk_input, label=label, branch_input=branch_input)

if __name__ == '__main__':
    dataset_path = r"datasets"
    import os
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    nu = 0.1
    l = 0.5
    # nu = 0.02
    # l = 0.3
    create_dataset(os.path.join(dataset_path, 'validation_dataset.npz'), 100, l=l, nu=nu)
    create_dataset(os.path.join(dataset_path, 'testing_dataset.npz'), 100, l=l, nu=nu)
    create_dataset(os.path.join(dataset_path, 'training_dataset.npz'), 3000, l=l, nu=nu)