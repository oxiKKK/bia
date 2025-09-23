import numpy as np
import matplotlib.pyplot as plt
from functions import *
from plot import *
import argparse

SAMPLES = 100


################################################
# Algorithms
################################################


# blnd search
def blind_search(func, bounds, samples):
    #
    # generate random points in [bounds[0], bounds[1]]
    #
    lo, hi = bounds
    for _ in range(samples):
        # random point in the given bounds
        x = np.random.uniform(lo, hi)
        y = np.random.uniform(lo, hi)
        yield x, y


def hill_climbing(func, bounds, samples):
    #
    # remember the best point and try to improve it with small random steps
    #
    lo, hi = bounds
    # Start with a random point
    current_point = np.random.uniform(lo, hi, size=2)
    best_value = func(current_point)

    step_size = (hi - lo) * 0.1  # 10% of the range

    for _ in range(samples):
        # create a random step in both axes
        step = np.random.uniform(-step_size, step_size, size=2)

        # calculate the new point
        new_point = current_point + step

        # ensure the new point is within bounds
        if np.all(new_point >= lo) and np.all(new_point <= hi):
            new_value = func(new_point)

            # If the new point is better, move to it
            if new_value < best_value:
                current_point, best_value = new_point, new_value

        yield current_point[0], current_point[1]


def tabu_search(func, bounds, samples):
    #
    # hill climbing, but remembers visited points (tabu list) and doesn't
    # return to them.
    #
    lo, hi = bounds

    # start with random point
    current_point = np.random.uniform(lo, hi, size=2)
    current_value = func(current_point)

    # initialize the tabu list
    tabu_list = [current_point]
    best_value = current_value

    tabu_size = samples // 10  # 10%

    step_size = (hi - lo) * 0.1  # 10% of the range

    for _ in range(samples):
        # random neighbors
        neighborhood = [
            np.clip(
                current_point + np.random.uniform(-step_size, step_size, size=2), lo, hi
            )
            for _ in range(20)
        ]

        # Evaluate neighborhood points and exclude tabu points
        candidates = []
        for point in neighborhood:
            # is this point near enough any tabu point?
            EPSILON = 1e-5
            is_tabu = any(np.allclose(point, tabu, atol=EPSILON) for tabu in tabu_list)
            value = func(point)
            if not is_tabu or value < best_value:
                candidates.append((point, value))

        if not candidates:
            continue

        # select the best point in the neighborhood
        candidates.sort(key=lambda x: x[1])
        next_point, next_value = candidates[0]

        # move only if lower on the minimum
        if next_value < current_value:
            current_point, current_value = next_point, next_value
            if current_value < best_value:
                best_value = current_value

        # update the list of tabu points
        tabu_list.append(current_point.copy())
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        yield current_point[0], current_point[1]


algorithms = [
    (blind_search, "Blind Search"),
    (hill_climbing, "Hill Climbing"),
    (tabu_search, "Tabu Search"),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run optimization algorithms on test functions."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        help="Name of the algorithm to run (e.g., 'Hill Climbing')",
        required=False,
    )
    parser.add_argument(
        "--function",
        type=str,
        help="Name of the function to optimize (e.g., 'Sphere')",
        required=False,
    )
    args = parser.parse_args()

    # Functions for testing (functions.py)
    all_functions = [
        (sphere, [-5, 5], "Sphere"),
        (schwefel, [-500, 500], "Schwefel"),
        (rosenbrock, [-5, 10], "Rosenbrock"),
        (rastrigin, [-5.12, 5.12], "Rastrigin"),
        (griewank, [-10, 10], "Griewank"),
        (levy, [-10, 10], "Levy"),
        (michalewicz, [0, np.pi], "Michalewicz"),
        (zakharov, [-5, 10], "Zakharov"),
        (ackley, [-5, 5], "Ackley"),
    ]

    # Filter functions and algorithms based on arguments
    selected_functions = [
        f for f in all_functions if args.function is None or f[2] == args.function
    ]
    selected_algorithms = [
        a for a in algorithms if args.algorithm is None or a[1] == args.algorithm
    ]

    if not selected_functions:
        print(f"No matching function found for '{args.function}'.")
        exit(1)

    if not selected_algorithms:
        print(f"No matching algorithm found for '{args.algorithm}'.")
        exit(1)

    # For all selected functions, display them in a grid and apply the selected algorithms
    for algo, name in selected_algorithms:
        animate_optimization_grid(
            functions_list=selected_functions,
            algorithm=algo,
            algo_name=name,
            samples=SAMPLES,
        )
