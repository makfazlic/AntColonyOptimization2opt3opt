def SA_python_solver(problem_info, problem_points):
    prev_cost = total_euc(problem_points)
    T = 100
    cool = 0.9995
    T_init = T
    plot_solution("0", "SA", problem_points)
    for i in range(100000):
        print(i, "Cost =",prev_cost)
        T = T*cool
        r1, r2 = np.random.randint(0, len(problem_points), size=2)
        temp = problem_points[r1]
        problem_points[r1] = problem_points[r2]
        problem_points[r2] = temp

        new_cost = total_euc(problem_points)
        if new_cost < prev_cost:
            prev_cost = new_cost
        else:
            x = np.random.uniform()
            if x <= np.exp((prev_cost-new_cost)/T):
                prev_cost = new_cost
            else:
                temp = problem_points[r1]
                problem_points[r1] = problem_points[r2]
                problem_points[r2] = temp

    plot_solution("0", "SA", problem_points)
    return prev_cost, []