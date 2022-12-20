import numpy as np

def compute_length(solution, dist_matrix):
    total_length = 0
    starting_node = solution[0]
    from_node = starting_node
    for node in solution[1:]:
        total_length += dist_matrix[from_node, node]
        from_node = node
    total_length += dist_matrix[starting_node, from_node]
    return total_length


def distance_euc(point_i, point_j):
    rounding = 0
    x_i, y_i = point_i[0], point_i[1]
    x_j, y_j = point_j[0], point_j[1]
    distance = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
    return round(distance, rounding)

def distance_matrix(points, dimension):
    matrix = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(i, dimension):
            if (i == j):
                matrix[i, j] = 0
            else:
                matrix[i, j] = distance_euc(points[i], points[j])
    matrix += matrix.T
    return matrix

def extract_problem(path):
    info = {}
    with open(path, "r") as problem:
        lines = problem.read().splitlines()
        info["name"] = " ".join(lines[0].split(" ")[1:])
        info["type"] = " ".join(lines[1].split(" ")[1:])
        info["comment"] = " ".join(lines[2].split(" ")[1:])
        info["dimension"] = int(" ".join(lines[3].split(" ")[1:]))
        info["edge_weight_type"] = " ".join(lines[4].split(" ")[1:])
        info["best_known"] = " ".join(lines[5].split(" ")[1:])
        info["wrapped"] = False
        dimension = int(info["dimension"])    
        points = []
        lines = lines[7:]
        lines.pop()
        for pointline in lines:
            pointline = pointline.split(" ")
            points.append([float(pointline[1]),float(pointline[2])])        
        assert dimension == len(points)
        points = np.array(points, dtype=np.float128)
        info["points"] = points
        info["distance_matrix"] = distance_matrix(points, dimension)
    return info

def tourLength(solution, dist_matrix):
    total_length = 0
    starting_node = solution[0]
    from_node = starting_node
    for node in solution[1:]:
        total_length += dist_matrix[from_node, node]
        from_node = node
    total_length += dist_matrix[starting_node, from_node]
    return total_length

def run(path):
    info = extract_problem(path)
    points = info["points"]
    d_matrix = info["distance_matrix"]
    dist_matrix = np.copy(d_matrix)
    dimension = info["dimension"]
    best_known = info["best_known"]
    node = 0
    tour = [node]
    for _ in range(dimension - 1):
        for new_node in np.argsort(dist_matrix[node]):
            if new_node not in tour:
                tour.append(new_node)
                node = new_node
                break
    # tour.append(starting_node)
    return tourLength(tour, dist_matrix)

def run_path(path):
    info = extract_problem(path)
    points = info["points"]
    d_matrix = info["distance_matrix"]
    dist_matrix = np.copy(d_matrix)
    dimension = info["dimension"]
    best_known = info["best_known"]
    node = 0
    tour = [node]
    for _ in range(dimension - 1):
        for new_node in np.argsort(dist_matrix[node]):
            if new_node not in tour:
                tour.append(new_node)
                node = new_node
                break
    # tour.append(starting_node)
    return tour

