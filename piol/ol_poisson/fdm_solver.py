import numpy as np
import matplotlib.pyplot as plt

import grf

def solve_ADR(xmin, xmax, tmin, tmax, k, v, g, dg, f, u0, Nx, Nt):
    """Solve 1D
    u_t = (k(x) u_x)_x - v(x) u_x + g(u) + f(x, t)
    with zero boundary condition.
    """

    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h ** 2

    D1 = np.eye(Nx, k=1) - np.eye(Nx, k=-1)
    D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)
    D3 = np.eye(Nx - 2)
    k = k(x)
    M = -np.diag(D1 @ k) @ D1 - 4 * np.diag(k) @ D2
    m_bond = 8 * h2 / dt * D3 + M[1:-1, 1:-1]
    v = v(x)
    v_bond = 2 * h * np.diag(v[1:-1]) @ D1[1:-1, 1:-1] + 2 * h * np.diag(
        v[2:] - v[: Nx - 2]
    )
    mv_bond = m_bond + v_bond
    c = 8 * h2 / dt * D3 - M[1:-1, 1:-1] - v_bond
    f = f(x[:, None], t)

    u = np.zeros((Nx, Nt))
    u[:, 0] = u0(x)
    for i in range(Nt - 1):
        gi = g(u[1:-1, i])
        dgi = dg(u[1:-1, i])
        h2dgi = np.diag(4 * h2 * dgi)
        A = mv_bond - h2dgi
        b1 = 8 * h2 * (0.5 * f[1:-1, i] + 0.5 * f[1:-1, i + 1] + gi)
        b2 = (c - h2dgi) @ u[1:-1, i].T
        u[1:-1, i + 1] = np.linalg.solve(A, b1 + b2)
    return x, t, u

#
T = 1
D = 0.01
k = 0.01

Nt = 101
m = 100

def eval_s(sensor_value):
    """Compute s(x, t) over m * Nt points for a `sensor_value` of `u`.
    """
    # T = 1
    # D = 0.01
    # k = 0.01
    #
    # Nt = 101
    # m = 100
    return solve_ADR(
        xmin=0,
        xmax=1,
        tmin=0,
        tmax=T,
        k=lambda x: D * np.ones_like(x),
        v=lambda x: np.zeros_like(x),
        g=lambda u: k * u ** 2,
        dg=lambda u: 2 * k * u,
        f=lambda x, t: np.tile(sensor_value[:, None], (1, len(t))),
        u0=lambda x: np.zeros_like(x),
        Nx=m,
        Nt=Nt,
    )[2]

def create_dataset(ls, sample_num=20):
    X_branch = []
    y = []
    sensor_values = []
    for repeat in range(sample_num):
        sensor_values.append(grf.grf_1D(0, 1, m, ls)[1])
    X_branch.append(sensor_values)
    s_values = list(map(eval_s, sensor_values))
    s_values = np.array(s_values)
    y.append(s_values)
    X_branch = np.concatenate(X_branch, axis=0)
    y = np.concatenate(y, axis=0)
    x = np.linspace(0, 1, m)
    t = np.linspace(0, T, Nt)
    xt = np.array([[a, b] for a in x for b in t])
    filepath = "dataset"
    # Check if the folder exists
    import os
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    filename = filepath + r"\l=" + str(ls) + ".npz"
    np.savez(filename, branch_input=X_branch, trunk_input=xt, y=y)
    print("Dataset created successfully at " + filename)

def create_dataset2(sample_num=200):
    X_branch = []
    y = []
    sensor_values = []
    for repeat in range(sample_num):
        ls = np.random.uniform(0.1, 0.9)
        sensor_values.append(grf.grf_1D(0, 1, m, ls)[1])
    X_branch.append(sensor_values)
    s_values = list(map(eval_s, sensor_values))
    s_values = np.array(s_values)
    y.append(s_values)
    X_branch = np.concatenate(X_branch, axis=0)
    y = np.concatenate(y, axis=0)
    x = np.linspace(0, 1, m)
    t = np.linspace(0, T, Nt)
    xt = np.array([[a, b] for a in x for b in t])
    filepath = "dataset"
    # Check if the folder exists
    import os
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    filename = filepath + r"\datset2" + ".npz"
    np.savez(filename, branch_input=X_branch, trunk_input=xt, y=y)
    print("Dataset created successfully at " + filename)

if __name__ == '__main__':
    # ls_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # for i in range(len(ls_list)):
    #     create_dataset(ls_list[i], sample_num=20)

    # create_dataset2(sample_num=200)
    sensor_value = grf.grf_1D(0, 1, m, 0.3)[1]
    x,t,u =solve_ADR(xmin=0,xmax=1,tmin=0,tmax=T,k=lambda x: D * np.ones_like(x),v=lambda x: np.zeros_like(x), g=lambda u: k * u ** 2,dg=lambda u: 2 * k * u,f=lambda x, t: np.tile(sensor_value[:, None], (1, len(t))),
        u0=lambda x: np.zeros_like(x),Nx=m,Nt=Nt,)
    grid_t, grid_x = np.meshgrid(t, x)
    u = u.reshape(100, 101)
    # Plot via the contour
    plt.contourf(grid_t,grid_x,  u, 100, cmap='coolwarm')
    plt.colorbar()
    plt.show()
    t = grid_t.flatten()
    x = grid_x.flatten()
    trunk_input = np.concatenate([x[:, None], t[:, None]], axis=1)
    print(trunk_input)
