import numpy as np
import matplotlib.pyplot as plt
from functions import *
from plot import *
import math
import argparse

################################################
# Constants
################################################

SAMPLES = 100
ALPHA = 0.95  # cooling rate


################################################
# Heatmap Visualization for Simulated Annealing
################################################


def simulated_annealing_with_heatmap(func, bounds, samples, T_0, T_min, alpha=ALPHA):
    lo, hi = bounds

    # initial solution
    current_point = np.random.uniform(lo, hi, size=2)
    current_value = func(current_point)

    # best solution tracking
    best_point = current_point.copy()
    best_value = current_value

    T = T_0

    # step size for generating neighbors
    step_size = (hi - lo) * 0.1

    # store all points and temperatures for heat map
    points = []
    temperatures = []
    values = []

    #
    # Run till we reach minimum temperature or max iterations
    #
    iteration = 0
    while T > T_min and iteration < samples:
        # create a neighbor
        neighbor = current_point + np.random.normal(0, step_size, size=2)
        neighbor = np.clip(neighbor, lo, hi)
        neighbor_value = func(neighbor)

        # store current state
        points.append(current_point.copy())
        temperatures.append(T)
        values.append(current_value)

        # is the new neighbor better?
        if neighbor_value < current_value:
            current_point = neighbor
            current_value = neighbor_value

            # is it even better than so far?
            if current_value < best_value:
                best_point = current_point.copy()
                best_value = current_value
        else:
            # nope, calculate acceptance probability
            delta = neighbor_value - current_value
            # Prevent division by zero when temperature is very low
            if T <= 1e-10:
                probability = 0.0  # dont accept
            else:
                probability = math.exp(-delta / T)
            r = np.random.uniform(0, 1)

            # accept with a certain probability
            if r < probability:
                current_point = neighbor
                current_value = neighbor_value

        # gradually cool down with a function
        T = T * alpha
        iteration += 1

    return (
        np.array(points),
        np.array(temperatures),
        np.array(values),
        best_point,
        best_value,
    )


def animate_simulated_annealing_combined(
    func, bounds, func_name, T_0, T_min, samples=SAMPLES, alpha=ALPHA
):
    """
    =====================================
    Combined 3D and heatmap visualization
    =====================================
    """
    points, temperatures, values, best_point, best_value = (
        simulated_annealing_with_heatmap(func, bounds, samples, T_0, T_min, alpha)
    )

    # Create the visualization with 3 subplots: 3D surface, heatmap, convergence
    fig = plt.figure(figsize=(15, 5))

    lo, hi = bounds
    grid_samples = 50
    x_grid = np.linspace(lo, hi, grid_samples)
    y_grid = np.linspace(lo, hi, grid_samples)
    X, Y = np.meshgrid(x_grid, y_grid)
    XY = np.stack([X, Y], axis=-1)
    Z = func(XY)

    # Left plot: 3D surface with search trajectory
    ax1 = fig.add_subplot(131, projection="3d")

    # Plot surface
    ax1.plot_surface(
        X,
        Y,
        Z,
        cmap="jet",
        alpha=0.4,
        edgecolor="k",
        linewidth=0.1,
        antialiased=True,
        rstride=2,
        cstride=2,
    )

    # Plot search trajectory
    if len(points) > 0:
        # Calculate Z values for the points
        z_values = func(points)

        # Plot trajectory with temperature-based coloring
        ax1.scatter(
            points[:, 0],
            points[:, 1],
            z_values,
            c=temperatures,
            cmap="hot",
            s=20,
            alpha=0.8,
            edgecolors="black",
            linewidth=0.5,
        )

        # Mark the best point
        best_z = func(best_point)
        ax1.scatter(
            best_point[0],
            best_point[1],
            best_z,
            c="lime",
            s=200,
            marker="*",
            edgecolors="black",
            linewidth=2,
            zorder=10,
        )

        # Draw trajectory line in 3D
        ax1.plot(points[:, 0], points[:, 1], z_values, "white", alpha=0.5, linewidth=1)
    else:
        print("WARNING: No points to plot in 3D trajectory.")

    ax1.set_title(f"{func_name}\nBest: {best_value:.6f}")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_zlabel("f(x1, x2)")
    ax1.view_init(elev=25, azim=-60)

    # Middle plot: Heat map with search trajectory
    ax2 = fig.add_subplot(132)

    # Plot function as heat map
    im = ax2.imshow(Z, extent=[lo, hi, lo, hi], origin="lower", cmap="jet", alpha=0.8)
    plt.colorbar(im, ax=ax2, label="Function Value", shrink=0.8)

    # Plot search trajectory with temperature-based coloring
    if len(points) > 0:
        ax2.scatter(
            points[:, 0],
            points[:, 1],
            c=temperatures,
            cmap="hot",
            s=20,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

        # Mark the best point
        ax2.scatter(
            best_point[0],
            best_point[1],
            c="lime",
            s=200,
            marker="*",
            edgecolors="black",
            linewidth=2,
            zorder=10,
        )

        # Draw trajectory line
        ax2.plot(points[:, 0], points[:, 1], "white", alpha=0.4, linewidth=1)

    ax2.set_title(f"{func_name} - Heat Map")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.grid(True, alpha=0.3)

    # Right plot: Convergence progress
    ax3 = fig.add_subplot(133)

    if len(values) > 0:
        # Calculate running minimum
        running_best = np.minimum.accumulate(values)

        ax3.plot(values, "b-", alpha=0.6, label="Current Value", linewidth=1)
        ax3.plot(running_best, "r-", linewidth=2, label="Best Value")
        ax3.axhline(
            y=best_value,
            color="lime",
            linestyle="--",
            linewidth=2,
            label=f"Final Best: {best_value:.3e}",
        )

        # Add temperature curve on secondary y-axis
        ax3_temp = ax3.twinx()
        ax3_temp.plot(
            temperatures, "orange", alpha=0.7, linewidth=1, label="Temperature"
        )
        ax3_temp.set_ylabel("Temperature", color="orange")
        ax3_temp.tick_params(axis="y", labelcolor="orange")

        ax3.set_xlabel("Nth iteration")
        ax3.set_ylabel("Function value")
        ax3.set_title("2D plot of the progress")
        ax3.legend(loc="upper right")
        ax3_temp.legend(loc="center right")
        ax3.grid(True, alpha=0.3)

    plt.suptitle(f"Simulated Annealing - {func_name}", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run optimization algorithms on test functions."
    )
    parser.add_argument(
        "--function",
        type=str,
        help="Name of the function to optimize (e.g., 'Sphere')",
        required=False,
    )
    args = parser.parse_args()

    selected_functions = [
        f for f in functions if args.function is None or f[2] == args.function
    ]

    if not selected_functions:
        print(f"No matching function found for '{args.function}'.")
        exit(1)

    temperature_for_function = {
        "Sphere": (5e2, 0.1),
        "Schwefel": (2e3, 0.1),
        "Rosenbrock": (2e6, 0.1),
        "Rastrigin": (8e2, 0.1),
        "Griewank": (2e1, 0.1),
        "Levy": (15e2, 0.1),
        "Ackley": (20, -2.0),
        "Zakharov": (5e4, 0.1),
        "Michalewicz": (30, 0.1),
    }

    for func, bounds, func_name in selected_functions:
        T_0, T_MIN = temperature_for_function.get(func_name, (100, 1))
        print(
            f"Running Simulated Annealing on {func_name} with T_0={T_0}, T_min={T_MIN}"
        )
        animate_simulated_annealing_combined(func, bounds, func_name, T_0, T_MIN)
