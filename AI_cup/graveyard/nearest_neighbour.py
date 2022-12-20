import numpy as np

def distance_euc(point_i, point_j):
    rounding = 0
    x_i, y_i = point_i[0], point_i[1]
    x_j, y_j = point_j[0], point_j[1]
    distance = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
    return round(distance, rounding)

def solution(points):
    solution = [points[0]]
    length = 0
    a = 0
    for i in range(len(points)):
        shortest_path = np.inf
        shortest_node = []
        for point_looking_at in points:
            if ((distance_euc(points[a], point_looking_at) != 0)  & (distance_euc(points[a], point_looking_at) < shortest_path)):
                shortest_path = distance_euc(points[a], point_looking_at)
                shortest_node = point_looking_at
        solution.append(shortest_node)
        length += shortest_path
        a = points.index(shortest_path)
        
    return length, solution

