######################################################################################
# Residual-based adaptive refinement (RAR) method
# Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021). DeepXDE: A deep learning
# library for solving differential equations. SIAM review, 63(1), 208-228.
######################################################################################
import torch
def rar_update_sampling(original_points, new_points, loss_new_points, require_grad=True):
    # param original_points: The original sampling points; a lits of tensors xi (0 <= i < len(original_points))
    #       xi: a tensor of shape (n, 1), where n is the number of sampling points
    # param new_points: The new sampling points; a lits of tensors xi
    avg_loss = torch.mean(loss_new_points)
    updated_points = []
    for i in range(len(original_points)):
        updated_points.append([])
    # Only maintain the points with loss greater than the average loss
    for i in range(len(new_points[0])):
        if loss_new_points[i] > avg_loss:
            for j in range(len(original_points)):
                updated_points[j].append(new_points[j][i])

    for i in range(len(original_points)):
        updated_points[i] = torch.tensor(updated_points[i])
        updated_points[i] = updated_points[i].unsqueeze(1)
        updated_points[i].requires_grad = require_grad

    # Concatenate the updated points to the original points
    for i in range(len(original_points)):
        updated_points[i] = torch.cat((updated_points[i], original_points[i]), dim=0)
    return updated_points

if __name__ == '__main__':
    a = torch.tensor([1,2,3])
    a = a.unsqueeze(1)
    print(a)
