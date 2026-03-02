import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


def create_mesh(N):
    """ Create a uniform 2D mesh on [-1, 1]^2 """
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    return X, Y


def assemble_system(X, Y, k, f):
    N = X.shape[0]
    h = 2 / (N - 1)  # Mesh spacing

    # Create sparse matrix and RHS (right-hand side vector)
    A = sp.lil_matrix((N * N, N * N))
    b = np.zeros(N * N)  # Initialize b

    # Define the source term function
    def source_function(x, y):
        return x ** 2 + y ** 2

    # Fill the matrix A and vector b
    for i in range(N):
        for j in range(N):
            index = i * N + j
            # Applying Dirichlet boundary conditions
            if i == 0 or i == N - 1 or j == 0 or j == N - 1:
                A[index, index] = 1
                b[index] = 0
            else:
                # Interior points
                A[index, index] = -4 * k[i, j] / h ** 2  # Center point
                A[index, index - 1] = k[i, j] / h ** 2  # Left point
                A[index, index + 1] = k[i, j] / h ** 2  # Right point
                A[index, index - N] = k[i, j] / h ** 2  # Bottom point
                A[index, index + N] = k[i, j] / h ** 2  # Top point
                b[index] = f[i,j] #source_function(X[i, j], Y[i, j])  # Update source term at this point

    return sp.csr_matrix(A), b


def main(f, N=50, b=0.3, k_min=0.5):
    X, Y = create_mesh(N)

    # Define coefficient k
    k = np.ones((N, N))
    for i in range(N):
        for j in range(N):
            if -b <= X[i, j] <= b and -b <= Y[i, j] <= b:
                k[i, j] = k_min

    A, b = assemble_system(X, Y, k, f)

    # Solve the linear system
    u = spla.spsolve(A, b)
    # print(min(u), max(u))
    # Reshape solution for plotting
    u = u.reshape((N, N))
    node = np.array([X.flatten(), Y.flatten()]).T
    return node, u.flatten()
    # Plot the solution
    # plt.figure(figsize=(6, 6))
    # plt.contourf(X, Y, u, levels=50, cmap='viridis')
    # plt.colorbar()
    # plt.title('Solution of 2D Poisson Equation with Variable Source')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()


if __name__ == "__main__":
    main(np.ones((50,50)))
