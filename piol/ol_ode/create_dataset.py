import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ode_solver
def t0(x):
    return 1

def t1(x):
    return x

def t2(x):
    return 2*x**2 - 1

def t3(x):
    return 4*x**3 - 3*x

def t4(x):
    return 8*x**4 - 8*x**2 + 1

def t5(x):
    return 16*x**5 - 20*x**3 + 5*x

def t6(x):
    return 32*x**6 - 48*x**4 + 18*x**2 - 1

def t7(x):
    return 64*x**7 - 112*x**5 + 56*x**3 - 7*x

class Chebyshev_poly():
    an = None

    @staticmethod
    def f(x, u):
        results = Chebyshev_poly.an[0] * t0(x)
        results += Chebyshev_poly.an[1] * t1(x)
        results += Chebyshev_poly.an[2] * t2(x)
        results += Chebyshev_poly.an[3] * t3(x)
        results += Chebyshev_poly.an[4] * t4(x)
        results += Chebyshev_poly.an[5] * t5(x)
        results += Chebyshev_poly.an[6] * t6(x)
        results += Chebyshev_poly.an[7] * t7(x)
        results = results * np.exp((-6) * np.linalg.norm(Chebyshev_poly.an - 0.5) ** 2)
        return results

def get_solution(an):
    Chebyshev_poly.an = an
    _, y = ode_solver.rk45_solver(Chebyshev_poly.f, (0, 1), [0], 100)
    return y

def create_dataset1(num=1e4):
    an_list = np.random.uniform(-1, 1, [int(num),8]) # Coefficients for the Chebyshev polynomials
    sol_list = []
    import tqdm
    qb = tqdm.tqdm(range(int(num)))
    for i in qb:
        an = an_list[i]
        sol = get_solution(an)
        sol_list.append(sol)
    # Record an_list and sol_list within dataset1_input.csv and dataset1_output.csv using pandas
    import pandas as pd
    df_input = pd.DataFrame(an_list)
    df_output = pd.DataFrame(sol_list)
    df_input.to_csv('dataset1_input.csv', index=False)
    df_output.to_csv('dataset1_output.csv', index=False)
    return


def sample_d_ball(num_points, dimension, radius=0.5):
    # Generate random Gaussian points
    points = np.random.normal(0, 1, (num_points, dimension))

    # Normalize each point to lie on the unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points_unit = points / norms

    # Scale points to lie within the ball of given radius
    scales = np.random.rand(num_points, 1) ** (1 / dimension)
    points_scaled = points_unit * scales * radius
    return points_scaled

def create_dataset2(num=1e4):
    num = int(num)
    an_list = sample_d_ball(num_points=num, dimension=8, radius=0.5) + 0.5 # Coefficients for the Chebyshev polynomials
    sol_list = []
    import tqdm
    qb = tqdm.tqdm(range(int(num)))
    for i in qb:
        an = an_list[i]
        sol = get_solution(an)
        sol_list.append(sol)
    # Record an_list and sol_list within dataset1_input.csv and dataset1_output.csv using pandas
    import pandas as pd
    df_input = pd.DataFrame(an_list)
    df_output = pd.DataFrame(sol_list)
    df_input.to_csv('dataset2_input.csv', index=False)
    df_output.to_csv('dataset2_output.csv', index=False)
    return

if __name__ == '__main__':
    create_dataset1(num=1e4)
    create_dataset2(num=1e4)
