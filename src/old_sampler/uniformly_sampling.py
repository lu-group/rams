import torch, math

def uniform_sampling(nodes, sample_num, dim, type, requires_grad=True):
    # param list nodes: The nodes of the geometry (the geometry should be a line/ rectangle/ cuboid)
    # param int sample_num: The number of sampling points
    # param int dim = 0/1/2/3: The dimension of the sampling points
    # param bool require_grad: Whether the sampling points require gradients
    # return: A list contains all the tensors with random values within the specified domain for each dimension.
    # e.g. A = [tensor([[x1], [x2], [x3]... [xn]]), tensor([[y1], [y2], [y3]... [yn]])]
    if dim == 0:
        sampling_point = torch.tensor(nodes, requires_grad=requires_grad)
        return sampling_point
    elif dim == 1:
        return uniform_sampling_1D(range=nodes, sampling_num=sample_num, required_grad=requires_grad)
    elif dim == 2:
        if type == "line":
            return uniform_sampling_2D_line(nodes, sample_num, required_grad=requires_grad)
        elif type == "triangle":
            return uniform_sampling_2D_triangle(nodes, sample_num, required_grad=requires_grad)
        elif type == "rectangle":
            return uniform_sampling_2D_rectangle(nodes, sample_num, required_grad=requires_grad)
        elif type == "polygon":
            return uniform_sampling_2D_polygon(nodes, sample_num, required_grad=requires_grad)
    else:
        raise ValueError("This sampling method is not supported yet!")

def uniform_sampling_1D(range, sampling_num, required_grad):
    """
    Generate uniformly distributed sampling points in a 1D space.

    Parameters:
    - range (list): A list with two elements indicating the start and end of the sampling range.
    - sampling_num (int): The number of sampling points to generate.
    - required_grad (bool): Specifies if the generated tensor should require gradient.

    Returns:
    - torch.Tensor: A 1D tensor of size (1, n) where n is the sampling_num, containing uniformly distributed sampling points.
    """
    # Ensure the input range is a 1x2 list and sampling_num is a positive integer
    if not isinstance(range, list) or len(range) != 2 or not isinstance(sampling_num, int) or sampling_num <= 0:
        raise ValueError("Invalid input parameters.")

    # Generate uniform sampling points
    start, end = range
    sampling_points = torch.linspace(start, end, steps=sampling_num)

    # Adjust the shape to 1xn tensor
    sampling_points = sampling_points.unsqueeze(1)  # Reshape to 1xn

    # Set the requires_grad flag
    sampling_points.requires_grad_(required_grad)

    return sampling_points

def uniform_sampling_2D_line(nodes, sampling_num, required_grad):
    """
    Generate uniformly distributed sampling points on a line segment defined in 2D space.

    Parameters:
    - nodes (list): A 2x2 list; [node1, node2]; used to define the location of the line segment.
                    node1/node2 are the coordinates of the start and end points of the line segment.
    - sampling_num (int): The number of sampling points to generate along the line segment.
    - required_grad (bool): Specifies if the generated tensor should require gradients.

    Returns:
    - x, y (tuple of tensors): Each tensor is of size (n, 1) where n is the sampling_num. They represent
                               the x and y coordinates of the sampling points, respectively.
    """
    if not isinstance(nodes, list) or len(nodes) != 2 or not all(len(node) == 2 for node in nodes):
        raise ValueError("Nodes input must be a 2x2 list.")
    if not isinstance(sampling_num, int) or sampling_num <= 0:
        raise ValueError("sampling_num must be a positive integer.")

    # Extract node coordinates
    node1, node2 = torch.tensor(nodes[0], dtype=torch.float32), torch.tensor(nodes[1], dtype=torch.float32)

    # Generate factors for linear interpolation
    t = torch.linspace(0, 1, steps=sampling_num).unsqueeze(1)

    # Linearly interpolate x and y coordinates
    x = node1[0] + (node2[0] - node1[0]) * t
    y = node1[1] + (node2[1] - node1[1]) * t

    # Set the requires_grad flag
    x.requires_grad_(required_grad)
    y.requires_grad_(required_grad)

    return x, y

def uniform_sampling_2D_triangle(nodes, sampling_num, required_grad):
    """
    Generate evenly distributed sampling points within a 2D triangle,
    aiming for uniformity within the constraints of triangular geometry.

    Parameters:
    - nodes (list): A 3x2 list; [node1, node2, node3]; used to define the triangle corners.
    - sampling_num (int): The target number of sampling points within the triangle.
    - required_grad (bool): Specifies if the generated tensor should require gradients.

    Returns:
    - x, y (tuple of tensors): Tensors of size (n, 1), representing the x and y coordinates of the sampling points.

    Important:
    - This function is not designed for large sampling_num values, as it may be inefficient.
    - The number of the sampling points might be smaller than the target number due to the filtering process.
    """

    def barycentric_coords(p, a, b, c):
        v0 = b - a
        v1 = c - a
        v2 = p - a
        d00 = torch.dot(v0, v0)
        d01 = torch.dot(v0, v1)
        d11 = torch.dot(v1, v1)
        d20 = torch.dot(v2, v0)
        d21 = torch.dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return u, v, w

    def point_in_triangle(p, a, b, c):
        u, v, w = barycentric_coords(p, a, b, c)
        return (u >= 0) & (v >= 0) & (w >= 0)

    nodes = torch.tensor(nodes, dtype=torch.float32, requires_grad=required_grad)
    min_x = torch.min(nodes[:, 0], 0)[0].item()
    max_x = torch.max(nodes[:, 0], 0)[0].item()
    min_y = torch.min(nodes[:, 1], 0)[0].item()
    max_y = torch.max(nodes[:, 1], 0)[0].item()

    # Generate a grid of points
    sqrt_n = int(torch.sqrt(torch.tensor(sampling_num, dtype=torch.float32)))
    x = torch.linspace(min_x, max_x, steps=sqrt_n)
    y = torch.linspace(min_y, max_y, steps=sqrt_n)
    grid_x, grid_y = torch.meshgrid(x, y)
    points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    # Filter points inside the triangle
    mask = torch.tensor([point_in_triangle(p, nodes[0], nodes[1], nodes[2]) for p in points])
    inside_points = points[mask]

    x = inside_points[:, 0].unsqueeze(1)
    y = inside_points[:, 1].unsqueeze(1)

    return x, y

def uniform_sampling_2D_rectangle(nodes, sampling_num, required_grad):
    """
    Generate uniformly distributed sampling points within a 2D rectangle.

    Parameters:
    - nodes (list): A 4x2 list; [node1, node2, node3, node4]; used to define the rectangle corners.
    - sampling_num (int): The number of sampling points to generate within the rectangle.
    - required_grad (bool): Specifies if the generated tensor should require gradients.

    Returns:
    - x, y (tuple of tensors): Each tensor is of size (n, 1) where n is the actual number of points generated.
                               They represent the x and y coordinates of the sampling points, respectively.
    """
    if not isinstance(nodes, list) or len(nodes) != 4 or not all(len(node) == 2 for node in nodes):
        raise ValueError("Nodes input must be a 4x2 list.")
    if not isinstance(sampling_num, int) or sampling_num <= 0:
        raise ValueError("sampling_num must be a positive integer.")

    # Calculate rows and cols
    rows = cols = int(math.sqrt(sampling_num))

    # Adjust if the square of rows/cols is less than sampling_num
    if rows * cols < sampling_num:
        cols += 1
    if rows * cols < sampling_num:
        rows += 1

    # Assuming nodes are axis-aligned, find min and max for x and y
    x_coords = [node[0] for node in nodes]
    y_coords = [node[1] for node in nodes]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Generate uniform points
    x = torch.linspace(min_x, max_x, steps=cols).repeat(rows, 1)
    y = torch.linspace(min_y, max_y, steps=rows).repeat_interleave(cols, dim=0).view(-1, 1)

    # Flatten and take the first sampling_num points if over-generated
    x = x.view(-1, 1)[:]
    y = y[:]

    # Set requires_grad if necessary
    x.requires_grad_(required_grad)
    y.requires_grad_(required_grad)

    return x, y

def uniform_sampling_2D_polygon(nodes, sampling_num, required_grad):
    """
    Attempt to generate uniformly distributed sampling points within a 2D polygon.

    Parameters:
    - nodes (list): A nx2 list containing the coordinates of the polygon's corners.
    - sampling_num (int): The target number of sampling points within the polygon.
    - required_grad (bool): Specifies if the generated tensors should require gradients.

    Returns:
    - Tuple of tensors (x, y): Tensors of size (n, 1), where n approximates the sampling_num. They represent
                               the x and y coordinates of the sampling points, respectively.

    Important:
    - This method aims to distribute points uniformly but the exact number of points may not match sampling_num
      due to the nature of the filtering step.
    - Assumes the polygon is convex and does not contain holes.
    """
    if not isinstance(nodes, list) or not all(len(node) == 2 for node in nodes):
        raise ValueError("Nodes input must be a nx2 list.")
    if not isinstance(sampling_num, int) or sampling_num <= 0:
        raise ValueError("sampling_num must be a positive integer.")

    # Calculate the density of points needed based on the bounding box area and desired sampling_num
    x_coords, y_coords = zip(*nodes)
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    area = (max_x - min_x) * (max_y - min_y)
    density = sampling_num / area

    # Estimate grid size
    grid_width = int((max_x - min_x) * torch.sqrt(torch.tensor(density)))
    grid_height = int((max_y - min_y) * torch.sqrt(torch.tensor(density)))
    grid_width = max(grid_width, 1)
    grid_height = max(grid_height, 1)

    # Generate a grid of points within the bounding box
    linspace_x = torch.linspace(min_x, max_x, steps=grid_width)
    linspace_y = torch.linspace(min_y, max_y, steps=grid_height)
    grid_x, grid_y = torch.meshgrid(linspace_x, linspace_y)
    grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    # Filter points to find those within the polygon
    inside_mask = [is_point_inside_polygon(x, y, nodes) for x, y in grid_points]
    inside_points = grid_points[inside_mask]

    # Separate x and y coordinates
    x, y = inside_points[:, 0].unsqueeze(1), inside_points[:, 1].unsqueeze(1)

    # Set the requires_grad flag
    x.requires_grad_(required_grad)
    y.requires_grad_(required_grad)

    return x, y

def is_point_inside_polygon(x, y, polygon):
    """
    Determine if a point is inside a polygon using the ray casting algorithm.

    Parameters:
    - x, y: Coordinates of the point to test.
    - polygon: List of [x, y] pairs defining the polygon vertices.

    Returns:
    - bool: True if the point is inside the polygon, False otherwise.
    """
    num = len(polygon)
    j = num - 1
    inside = False

    for i in range(num):
        if ((polygon[i][1] >= y) != (polygon[j][1] >= y)) and \
                (x <= (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) / (polygon[j][1] - polygon[i][1]) + polygon[i][0]):
            inside = not inside
        j = i

    return inside


if __name__ == '__main__':
    nodes = [[0,0],[0,1], [1,1], [1,0]]
    sampling_num = 3000
    required_grad = True
    x, y = uniform_sampling_2D_polygon(nodes, sampling_num, required_grad)
    print(x)
    print(y)
    print(x.shape)
    import matplotlib.pyplot as plt
    # Plot the polygon
    plt.plot([node[0] for node in nodes], [node[1] for node in nodes], 'b-')
    plt.plot([nodes[-1][0], nodes[0][0]], [nodes[-1][1], nodes[0][1]], 'b-')
    plt.scatter(x.detach().numpy(), y.detach().numpy())
    plt.show()

