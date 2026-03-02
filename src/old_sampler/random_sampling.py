import torch

def random_sampling(nodes, sample_num, dim, type, requires_grad=True):
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
        return random_sampling_1D(range=nodes, sampling_num=sample_num, required_grad=True)
    elif dim == 2:
        if type == "line":
            return random_sampling_2D_line(nodes, sample_num, required_grad)
        elif type == "triangle":
            return random_sampling_2D_triangle(nodes, sample_num, required_grad)
        elif type == "rectangle":
            return random_sampling_2D_rectangle(nodes, sample_num, required_grad)
        elif type == "polygon":
            return random_sampling_2D_polygon(nodes, sample_num, required_grad)
    else:
        raise ValueError("This sampling method is not supported yet!")

def random_sampling_1D(range, sampling_num, required_grad=True):
    """
    Generate random sampling points in a 1D space.

    Parameters:
    - range (list): A list with two elements indicating the start and end of the sampling range.
    - sampling_num (int): The number of sampling points to generate.
    - required_grad (bool): Specifies if the generated tensor should require gradient.

    Returns:
    - torch.Tensor: A 1D tensor of size (n,) where n is the sampling_num, containing random sampling points.
    """
    # Ensure the input range is a 1x2 list and sampling_num is a positive integer
    if not isinstance(range, list) or len(range) != 2 or not isinstance(sampling_num, int) or sampling_num <= 0:
        raise ValueError("Invalid input parameters.")

    # Generate sampling points
    start, end = range
    sampling_points = torch.rand(sampling_num) * (end - start) + start
    sampling_points, _ = torch.sort(sampling_points)  # Sort the sampling points
    sampling_points = sampling_points.unsqueeze(1)  # Convert to nx1 tensor

    # Set the requires_grad flag
    # Note: Correct use of requires_grad_() for setting the flag in-place
    sampling_points.requires_grad_(required_grad)

    return sampling_points

def random_sampling_2D_line(nodes, sampling_num, required_grad):
    """
    Generate random sampling points on a line segment defined in 2D space.

    Parameters:
    - nodes (list): A 2x2 list containing the coordinates of the two nodes defining the line segment.
                    Each node is specified as [x, y].
    - sampling_num (int): The number of sampling points to generate along the line segment.
    - required_grad (bool): Specifies if the generated tensors should require gradients.

    Returns:
    - Tuple of tensors (x, y): Each tensor is of size (n, 1), where n is the sampling_num. They represent
                               the x and y coordinates of the sampling points, respectively.
    """
    if not isinstance(nodes, list) or len(nodes) != 2 or not all(len(node) == 2 for node in nodes):
        raise ValueError("Nodes input must be a nx2 list.")
    if not isinstance(sampling_num, int) or sampling_num <= 0:
        raise ValueError("sampling_num must be a positive integer.")

    # Extract node coordinates
    node1, node2 = nodes
    x1, y1 = node1
    x2, y2 = node2

    # Generate random sampling points along the line
    t = torch.rand(sampling_num, 1)  # Parameter t to interpolate between node1 and node2
    x = x1 + (x2 - x1) * t
    y = y1 + (y2 - y1) * t

    # Set requires_grad if necessary
    x.requires_grad_(required_grad)
    y.requires_grad_(required_grad)

    return x, y

def random_sampling_2D_triangle(nodes, sampling_num, required_grad):
    """
    Generate random sampling points within a 2D triangle.

    Parameters:
    - nodes (list): A 3x2 list containing the coordinates of the triangle's corners ([node1, node2, node3]).
    - sampling_num (int): The number of sampling points to generate within the triangle.
    - required_grad (bool): Specifies if the generated tensors should require gradients.

    Returns:
    - x, y (tuple of tensors): Each tensor is of size (n, 1), where n is the sampling_num. They represent
                               the x and y coordinates of the sampling points, respectively.
    """
    if not isinstance(nodes, list) or len(nodes) != 3 or not all(len(node) == 2 for node in nodes):
        raise ValueError("Nodes input must be a 3x2 list.")
    if not isinstance(sampling_num, int) or sampling_num <= 0:
        raise ValueError("sampling_num must be a positive integer.")

    # Extract vertex coordinates
    A, B, C = torch.tensor(nodes, dtype=torch.float32)

    # Generate random numbers for barycentric coordinates
    r1 = torch.sqrt(torch.rand(sampling_num, 1))
    r2 = torch.rand(sampling_num, 1)

    # Compute the sampling points
    P = (1 - r1) * A + (r1 * (1 - r2)) * B + (r2 * r1) * C

    # Separate x and y coordinates
    x = P[:, 0:1]
    y = P[:, 1:2]

    # Set requires_grad if necessary
    x.requires_grad_(required_grad)
    y.requires_grad_(required_grad)

    return x, y

def random_sampling_2D_rectangle(nodes, sampling_num, required_grad):
    """
    Generate random sampling points within a 2D axis-aligned rectangle.

    Parameters:
    - nodes (list): A 4x2 list containing the coordinates of the rectangle's corners.
    - sampling_num (int): The number of sampling points to generate within the rectangle.
    - required_grad (bool): Specifies if the generated tensors should require gradients.

    Returns:
    - x, y (tuple of tensors): Each tensor is of size (n, 1), where n is the sampling_num. They represent
                               the x and y coordinates of the sampling points, respectively.
    """
    if not isinstance(nodes, list) or len(nodes) != 4 or not all(len(node) == 2 for node in nodes):
        raise ValueError("Nodes input must be a 4x2 list.")
    if not isinstance(sampling_num, int) or sampling_num <= 0:
        raise ValueError("sampling_num must be a positive integer.")

    # Convert to tensor for easier manipulation
    nodes_tensor = torch.tensor(nodes, dtype=torch.float32)

    # Find min and max for x and y coordinates
    min_x, max_x = torch.min(nodes_tensor[:, 0]), torch.max(nodes_tensor[:, 0])
    min_y, max_y = torch.min(nodes_tensor[:, 1]), torch.max(nodes_tensor[:, 1])

    # Generate random points within the bounds
    x = torch.rand(sampling_num, 1) * (max_x - min_x) + min_x
    y = torch.rand(sampling_num, 1) * (max_y - min_y) + min_y

    # Set requires_grad if necessary
    x.requires_grad_(required_grad)
    y.requires_grad_(required_grad)

    return x, y

def random_sampling_2D_polygon(nodes, sampling_num, required_grad):
    """
    Generate random sampling points within a 2D polygonal surface.

    Parameters:
    - nodes (list): A nx2 list containing the coordinates of the polygon's corners ([node1, node2, ..., noden]).
    - sampling_num (int): The number of sampling points to generate within the polygon.
    - required_grad (bool): Specifies if the generated tensors should require gradients.

    Returns:
    - Tuple of tensors (x, y): Each tensor is of size (n, 1), where n is the actual number of points generated
                               within the polygon. They represent the x and y coordinates of the sampling points,
                               respectively.

    Important:
    - The algorithm used to generate the sampling points is a basic one that may not work for all types of polygons.
    - The algorithm assumes that the polygon is convex and does not contain holes.
    - If the sampling domain is a triangle or rectangular, please use other function for more efficient sampling.
    """
    if not isinstance(nodes, list) or not all(len(node) == 2 for node in nodes):
        raise ValueError("Nodes input must be a nx2 list.")
    if not isinstance(sampling_num, int) or sampling_num <= 0:
        raise ValueError("sampling_num must be a positive integer.")

    # Find the bounding box of the polygon
    x_coords, y_coords = zip(*nodes)
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Initialize tensors to store sampling points
    x = torch.empty((0, 1))
    y = torch.empty((0, 1))

    # Generate points within the bounding box and filter them
    while len(x) < sampling_num:
        remaining_samples = sampling_num - len(x)
        # Generate random points within the bounding box
        random_x = torch.FloatTensor(remaining_samples, 1).uniform_(min_x, max_x)
        random_y = torch.FloatTensor(remaining_samples, 1).uniform_(min_y, max_y)

        # Filter points to find those within the polygon (basic check for convex polygons)
        inside = [i for i, (px, py) in enumerate(zip(random_x, random_y)) if is_point_inside_polygon(px, py, nodes)]
        inside_x = random_x[inside]
        inside_y = random_y[inside]

        # Append valid points
        x = torch.cat((x, inside_x[:remaining_samples]), 0)
        y = torch.cat((y, inside_y[:remaining_samples]), 0)

    # Truncate tensors to the desired number of samples
    x, y = x[:sampling_num], y[:sampling_num]

    # Set requires_grad if necessary
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
        if ((polygon[i][1] > y) != (polygon[j][1] > y)) and \
                (x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) / (polygon[j][1] - polygon[i][1]) + polygon[i][0]):
            inside = not inside
        j = i

    return inside


if __name__ == '__main__':
    nodes = [[0,0],[0,1], [1,1], [1,0]]
    sampling_num = 10
    required_grad = True
    x, y = random_sampling_2D_rectangle(nodes, sampling_num, required_grad)
    print(x)
    print(y)
    print(x/y)
    import matplotlib.pyplot as plt
    # Plot the polygon
    plt.plot([node[0] for node in nodes], [node[1] for node in nodes], 'b-')
    plt.plot([nodes[-1][0], nodes[0][0]], [nodes[-1][1], nodes[0][1]], 'b-')
    plt.scatter(x.detach().numpy(), y.detach().numpy())
    plt.show()

