import numpy as np
import matplotlib.pyplot as plt

Ndt = 1000
Ndx = 2.0 / 256
def fdm_burgers(nu):
    # Parameters
    L = 2.0         # Domain length (-1 to 1)
    Ndx = 256         # Number of spatial points
    dx = L / (Ndx - 1)  # Spatial step size
    x = np.linspace(-1, 1, Ndx)

    T = 1.0         # Total time
    dt = T / Ndt
    nt = int(T / dt)  # Number of time steps

    # Initial condition
    u = -np.sin(np.pi * x) #np.zeros(Ndx)
    u_new = np.copy(u)  # Copy of u for time-stepping

    # Array to store the solution at all time steps
    u_solution = np.zeros((Ndt, Ndx))
    u_solution[0, :] = u

    # FDM - upwind scheme
    def compute_derivatives(u):
        u_xx = (u[:-2] - 2 * u[1:-1] + u[2:]) / dx**2

        # Upwind scheme for the convection term
        u_x = np.zeros_like(u[1:-1])
        u_pos = u[1:-1] > 0
        u_neg = u[1:-1] <= 0

        u_x[u_pos] = (u[1:-1][u_pos] - u[:-2][u_pos]) / dx  # Backward difference
        u_x[u_neg] = (u[2:][u_neg] - u[1:-1][u_neg]) / dx   # Forward difference

        return u_xx, u_x

    # Euler forword time-stepping
    for n in range(1, nt):
        u_xx, u_x = compute_derivatives(u)
        u_new[1:-1] = u[1:-1] + dt * (nu * u_xx - u[1:-1] * u_x)

        # Enforce boundary conditions
        u_new[0] = 0
        u_new[-1] = 0

        # Update solution
        u = u_new.copy()
        u_solution[n, :] = u

    t = np.linspace(0, T, nt)
    T_grid, X_grid = np.meshgrid(t, x)
    return T_grid, X_grid, u_solution.T

if __name__ == '__main__':
    nu = 0.01
    T_grid, X_grid, u_solution = fdm_burgers(nu)
    # Visualization with contour plot
    plt.figure(figsize=(10, 6))
    plt.contourf(T_grid, X_grid, u_solution, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title("Burgers' Equation Solution")
    plt.xlabel("Time")
    plt.ylabel("Spatial coordinate")
    plt.show()
