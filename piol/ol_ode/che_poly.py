import numpy as np
import torch

class Chebyshev_poly():
    def __init__(self, x_span, n, device=None, order=8):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.x = torch.linspace(x_span[0], x_span[1], n)
        yn_list = []
        for i in range(order):
            if i == 0:
                yn_list.append(torch.ones(n))
            elif i == 1:
                yn_list.append(self.x)
            else:
                yn_list.append(2 * self.x * yn_list[i-1] - yn_list[i-2])
        self.yn = torch.stack(yn_list, dim=1).transpose(0, 1).to(self.device)

    def get_results(self, xi_tensor):
        # xi_tensor: (batch_size, 8); 8 is the number of Chebyshev polynomials
        return torch.matmul(xi_tensor, self.yn)
