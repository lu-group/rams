import torch

def gradients(U, X, order=1):
    # param U: the function to be differentiated; the shape of U is [N, 1] or [N]
    # param X: the variable of the function U; the shape of X is [N, 1] or [N]
    # param order: the order of the derivative
    # return: the derivative of U with respect to X
    if order == 1:
        return torch.autograd.grad(U, X, grad_outputs=torch.ones_like(U),
                                   create_graph=True,
                                   retain_graph=True,
                                   only_inputs=True,)[0]
    else:
        return gradients(gradients(U, X), X, order=order - 1)

if __name__ == '__main__':
    def f(input):
        return (input[:, 0] + input[:, 1]) ** 2
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32).T
    x.requires_grad = True
    y = f(x)
    y_sum = y.sum()
    print(y)
    # Compute y_x
    print(gradients(y, x))
    # Compute y_xx
    print(gradients(y, x, order=2))