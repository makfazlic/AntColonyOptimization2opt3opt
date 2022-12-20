# %% [markdown]
# # AI Cup test
# 
# _Some possible solutions_
# 
# - Brute force
# - DP
# - LKH
# - concorde
# - ACO
# - christofides
# - Simulated Annealing
# 
# Important - changed euclidian distance could result in bugs
# 

# %%
debug = True
SEED = 69

# %% [markdown]
# Module import
# 

# %%
import glob
import random
import numpy as np
from matplotlib import pyplot as plt
random.seed(a=SEED, version=2)

# %% [markdown]
# Load problems and check
# 

# %%
problems = glob.glob('./problems/*.tsp')
print("Files loaded" if np.all([n in ['./problems/fl1577.tsp','./problems/pr439.tsp','./problems/ch130.tsp','./problems/rat783.tsp','./problems/d198.tsp', './problems/kroA100.tsp','./problems/u1060.tsp','./problems/lin318.tsp','./problems/eil76.tsp','./problems/pcb442.tsp'] for n in problems]) else "Missing files")

# %% [markdown]
# Overview of the problem headers
# 

# %%
for problem in problems:
    if(debug):
        break
    with open(problem,"r") as probfile:
        file = probfile.read().splitlines()
        print(file[0])
        print(file[1])
        print(file[2])
        print(file[3])
        print(file[4])
        print(file[5])
        print()

# %% [markdown]
# Euclidian distance
# 

# %%
def distance_euc(point_i, point_j):
    rounding = 0
    x_i, y_i = point_i[0], point_i[1]
    x_j, y_j = point_j[0], point_j[1]
    distance = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
    return round(distance, rounding)

# %% [markdown]
# Implement the plot function and test of euclidian distance
# 

# %%
def plot_euc(point_1, point_2):
    plt.figure(figsize=(5,5))
    distance = distance_euc(point_1, point_2)
    plt.xlim(0, max(point_1[0], point_2[0])+1)
    plt.ylim(0, max(point_1[1], point_2[1])+1)
    plt.grid()
    plt.plot(point_1[0], point_1[1], marker="o", markersize=10, markerfacecolor="blue")
    plt.plot(point_2[0], point_2[1], marker="o", markersize=10, markerfacecolor="blue")
    plt.plot([point_1[0], point_2[0]], [point_1[1], point_2[1]], color="blue")
    title = f"Distance between {point_1} and {point_2} is: ", distance
    plt.title(title)
    plt.show()

for x in range(10):
    if(debug):
        break
    plot_euc([random.randint(0,100), random.randint(0,100)], [random.randint(0,100), random.randint(0,100)])

# %% [markdown]
# Wrap points (To have full tour)
# 

# %%
def wrap_points(points):
    points = np.append(points, [points[0]],axis=0)
    return points

# %% [markdown]
# Total euclidian distance over all points
# 

# %%
def total_euc(points):
    total = 0
    for i in range(len(points)-1):
        total += distance_euc(points[i],points[i+1])
    return total

# %% [markdown]
# Implement the plot function and test of total euclidian distance
# 

# %%
def plot_euc_total(points):
    plt.figure(figsize=(5, 5))
    total = total_euc(points)
    plt.title(("Total distance is: ", total))
    plt.plot([point[0] for point in points] , [point[1] for point in points], 'b-')
    for i in range(len(points)-1): 
        plt.annotate(points[i], (points[i][0], points[i+1][1]))
    plt.show()

for x in range(10):
    if(debug):
        break
    test_points = []
    for y in range(np.random.randint(30)):
        a, b = np.random.randint(100, size=2)
        test_points.append((a, b))
    plot_euc_total(test_points)

# %% [markdown]
# Distance matrix (dimension x dimension)
# 

# %%
def distance_matrix(points, dimension):
    matrix = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(i, dimension):
            if (i == j):
                matrix[i, j] = np.inf
            else:
                matrix[i, j] = distance_euc(points[i], points[j])
        matrix += matrix.T
    return matrix

# %% [markdown]
# Extract problem details from a file
# 

# %%
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

# %% [markdown]
# # Solvers
# 
# Solver API calls for solver as [solvername]\_[py,rust]\_solver(problem_object) and returns -> distance, path
# 

# %% [markdown]
# Random method for finding best path
# 

# %%
def random_py_solver(problem_object):
    dimension = int(problem_object["dimension"])
    path = np.random.choice(np.arange(dimension), size=dimension,
                            replace=False)
    distance = 0
    if problem_object["wrapped"]:
        distance = total_euc(problem_object["points"])
    else:
        distance = total_euc(wrap_points(problem_object["points"]))
    return distance, path

# %% [markdown]
# # Hyperparameter tuning
# 

# %%
from ACO import *

def ACO_py_for_hyperparameters(
    points,
    d_matrix,
    seed,
    n_iter,
    n_ants,
    alpha,
    beta,
    rho,
    Q,
    tau0
):
    ants = AntOpt(
        points,
        d_matrix,
        seed,     
        n_iter,    
        n_ants,     
        alpha,       
        beta,        
        rho,      
        Q,         
        tau0
    )
    #print(ants)
    result = ants.run_ants()
    #print(result)
    #ants.plot_path(result[1])
    return result[1]


problem = extract_problem(problems[2])
distance = problem["distance_matrix"]
points = problem["points"]

ACO_py_for_hyperparameters(points, None, SEED, 500, 30, 2, 3, 0.99, 0.2, 1e-4)


# %%
def get_step(min, max, step_count):
  return (max-min)/step_count


def ACO_py_hyperparameter(steps):

  for problem in problems:
    problem_object = extract_problem(problem)
    problem_name = problem_object["name"]
    problem_points = problem_object["points"]
    problem_dimension = problem_object["dimension"]

    max_ants = problem_dimension/2
    min_ants = 1
    step_ants = get_step(min_ants, max_ants, steps)
    hyper_ants = np.arange(min_ants, max_ants, step_ants)

    max_alpha = 4
    min_alpha = 0.5
    step_alpha = get_step(min_alpha, max_alpha, steps)
    hyper_alpha = np.arange(min_alpha, max_alpha, step_alpha)

    max_beta = 4
    min_beta = 0.5
    step_beta = get_step(min_beta, max_beta, steps)
    hyper_beta = np.arange(min_beta, max_beta, step_beta)

    max_rho = 1
    min_rho = 0.5
    step_rho = get_step(min_rho, max_rho, steps)
    hyper_rho = np.arange(min_rho, max_rho, step_rho)

    max_Q = 1
    min_Q = 0.1
    step_Q = get_step(min_Q, max_Q, steps)
    hyper_Q = np.arange(min_Q, max_Q, step_Q)

    max_tau0 = 1
    min_tau0 = 1e-5
    step_tau0 = get_step(min_tau0, max_tau0, steps)
    hyper_tau0 = np.arange(min_tau0, max_tau0, step_tau0)

    hyperparameters = np.array(np.meshgrid(
      hyper_ants,
      hyper_alpha,
      hyper_beta,
      hyper_rho,
      hyper_Q,
      hyper_tau0
    )).T.reshape(-1, 6)

    print("Ant range ------- ", hyper_ants)
    print("Alpha range ----- ", hyper_alpha)
    print("Beta range ------ ", hyper_beta)
    print("Rho range ------- ", hyper_rho)
    print("Q range --------- ", hyper_Q)
    print("Tau0 range ------ ", hyper_tau0)
    print("HYPERPARAMETERS size", hyperparameters.shape)
    print()


  

    
ACO_py_hyperparameter(5)

# %%
from ACO import *

def ACO_py_solver(problem_object):
    distance = problem_object["distance_matrix"]
    points = problem_object["points"]
    ants = AntOpt(
        points,        # Points of TSP
        d_matrix = None,
        seed=SEED,     # Seed of the model
        n_iter=500,    # Number of iterations
        n_ants=30,     # Number of ants
        alpha=2,       # pheromone importance
        beta=3,        # local importance heuristic
        rho=0.99,      # evaporation factor
        Q=0.2,         # pheromone amplification factor
        tau0=1e-4 
    )
    print(ants)
    best_path = ants.run_ants()
    ants.plot_path(best_path)



ACO_py_solver(extract_problem(problems[2]))



