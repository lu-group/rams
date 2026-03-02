import numpy as np
from scipy.integrate import solve_ivp

def rk45_solver(f, x_span=(0,1), u0=[0], mesh_size=100):
    sol = solve_ivp(f, x_span, u0, method='RK45', t_eval=np.linspace(x_span[0], x_span[1], mesh_size))
    return sol.t, sol.y[0]
