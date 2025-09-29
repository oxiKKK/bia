import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


# Animated optimization visualization for multiple functions at once
def animate_optimization_grid(functions_list, algorithm, algo_name, samples):
    """
    Animate optimization on multiple functions simultaneously in a grid layout

    Args:
        functions_list: List of tuples (func, bounds, func_name)
        algorithm: Algorithm generator function
        algo_name: Name of the algorithm
        samples: Number of samples to run
    """
    n_functions = len(functions_list)
    # Calculate grid dimensions (try to make it roughly square)
    cols = int(np.ceil(np.sqrt(n_functions)))
    rows = int(np.ceil(n_functions / cols))

    z = 0.0

    # Create figure with subplots
    fig = plt.figure(figsize=(3 * cols, 3 * rows))
    fig.suptitle(algo_name, fontsize=18)

    # Store data for each function
    function_data = []

    for i, (func, bounds, func_name) in enumerate(functions_list):
        lo, hi = bounds
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")

        # Create surface for this function
        grid_samples = 50
        xs_grid = np.linspace(lo, hi, grid_samples)
        ys_grid = np.linspace(lo, hi, grid_samples)
        X, Y = np.meshgrid(xs_grid, ys_grid, indexing="xy")
        XY = np.stack([X, Y, np.full_like(X, z)], axis=-1)
        Z = func(np.asarray(XY))

        # Plot surface
        surf = ax.plot_surface(
            X,
            Y,
            Z,
            cmap="jet",
            edgecolor="k",
            linewidth=0.1,
            antialiased=True,
            alpha=0.4,
            rstride=2,
            cstride=2,
        )

        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.tick_params(axis="z", labelsize=6)

        # Create scatter plots
        scatter = ax.scatter(
            [], [], [], c="red", s=15, alpha=0.8, zorder=10, depthshade=False
        )
        best_scatter = ax.scatter(
            [],
            [],
            [],
            c="lime",
            s=60,
            alpha=1.0,
            zorder=15,
            depthshade=False,
            marker="*",
            edgecolor="black",
            linewidth=1,
        )

        # Store function data
        function_data.append(
            {
                "func": func,
                "bounds": bounds,
                "func_name": func_name,
                "ax": ax,
                "scatter": scatter,
                "best_scatter": best_scatter,
                "xs": [],
                "ys": [],
                "zs": [],
                "best_val": float("inf"),
                "best_point": None,
                "algorithm_generator": algorithm(func, bounds, samples),
            }
        )

    def init():
        for data in function_data:
            data["scatter"]._offsets3d = ([], [], [])
            data["best_scatter"]._offsets3d = ([], [], [])
        return [data["scatter"] for data in function_data] + [
            data["best_scatter"] for data in function_data
        ]

    def update(frame):
        artists = []
        for data in function_data:
            try:
                # Get next point from algorithm
                x, y = next(data["algorithm_generator"])
                pt = np.array([x, y, z])
                val = data["func"](pt)

                data["xs"].append(x)
                data["ys"].append(y)
                data["zs"].append(val)

                # Update best result if current is better
                if val < data["best_val"]:
                    data["best_val"] = val
                    data["best_point"] = (x, y, val)
                    data["best_scatter"]._offsets3d = ([x], [y], [val])

                data["scatter"]._offsets3d = (data["xs"], data["ys"], data["zs"])
                data["ax"].set_title(
                    f"{data['func_name']}, Iteration {frame+1}/{samples}, Best: {data['best_val']:.2f}",
                    fontsize=10,
                )

                artists.extend([data["scatter"], data["best_scatter"]])

            except StopIteration:
                # Algorithm finished for this function
                artists.extend([data["scatter"], data["best_scatter"]])

        return artists

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=samples,
        init_func=init,
        blit=False,
        interval=100,
        repeat=False,
    )

    plt.tight_layout(pad=2.0)
    plt.show()
    return ani


# Animated optimization visualization (universal for all functions and algorithms)
def animate_optimization(
    func, bounds, algorithm, func_name="Function", algo_name="Algorithm", samples=100
):

    z = 0.0

    lo, hi = bounds
    xs, ys, zs = [], [], []

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the function surface (MATLAB style wireframe)
    grid_samples = 50
    xs_grid = np.linspace(lo, hi, grid_samples)
    ys_grid = np.linspace(lo, hi, grid_samples)
    X, Y = np.meshgrid(xs_grid, ys_grid, indexing="xy")
    XY = np.stack([X, Y, np.full_like(X, z)], axis=-1)
    Z = func(np.asarray(XY))

    # Colorful surface plot with mesh lines (like the example image)
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap="jet",
        edgecolor="k",
        linewidth=0.2,
        antialiased=True,
        alpha=0.4,
        rstride=2,
        cstride=2,
    )

    # Set dynamic title and labels
    ax.set_title(f"{func_name} - {algo_name}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x1, x2)")

    # Set axis limits dynamically based on bounds and function values
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    z_min, z_max = np.min(Z), np.max(Z)
    z_range = z_max - z_min
    ax.set_zlim(z_min - 0.1 * z_range, z_max + 0.1 * z_range)

    # Add grid and set view
    ax.grid(True)
    ax.view_init(elev=25, azim=-60)

    # Scatter plot for animated points - small red circles for regular samples
    scatter = ax.scatter(
        [],
        [],
        [],
        c="red",
        s=25,
        alpha=0.8,
        zorder=10,
        depthshade=False,
        edgecolor="white",
        linewidth=0.5,
    )

    # Scatter plot for best point - large green star for best result
    best_scatter = ax.scatter(
        [],
        [],
        [],
        c="lime",
        s=200,
        alpha=1.0,
        zorder=15,
        depthshade=False,
        marker="*",
        edgecolor="black",
        linewidth=3,
    )

    # Track best result
    best_val = float("inf")
    best_point = None

    # Create algorithm generator
    algorithm_generator = algorithm(func, bounds, samples)

    def init():
        scatter._offsets3d = ([], [], [])
        best_scatter._offsets3d = ([], [], [])
        return (scatter, best_scatter)

    def update(frame):
        nonlocal best_val, best_point

        try:
            # Get next point from algorithm
            x, y = next(algorithm_generator)
            pt = np.array([x, y, z])
            val = func(pt)
            xs.append(x)
            ys.append(y)
            zs.append(val)

            # Update best result if current is better
            if val < best_val:
                best_val = val
                best_point = (x, y, val)
                best_scatter._offsets3d = ([x], [y], [val])

            scatter._offsets3d = (xs, ys, zs)
            ax.set_title(
                f"{func_name} - {algo_name} | Sample {frame+1}/{samples} | Best: {best_val:.4f}"
            )

        except StopIteration:
            # Algorithm finished early
            pass

        return (scatter, best_scatter)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=samples,
        init_func=init,
        blit=False,
        interval=80,
        repeat=False,
    )

    plt.show()
    return ani
