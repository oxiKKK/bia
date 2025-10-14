"""
Differential Evolution (DE)

1. Vytváří populaci kandidátských řešení (vektorů)
2. Pro každého kandidáta vytváří zkušební vektor pomocí:
   - Mutace: vytvoření mutanta z náhodných členů populace
   - Křížení: smíchání mutanta s cílovým vektorem
3. Vybere lepší řešení (zkouška vs cíl) pro další generaci

Key concepts:
- NP: Velikost populace (pocet kandidatu)
- F: Faktor mutace (ovlivnuje silu mutace, obvykle 0.5-1.0)
- CR: Míra křížení (pravděpodobnost převzetí genu z mutanta, obvykle 0.5-0.9)
- G_maxim: Maximální počet generací (iterací)
"""

from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import argparse
from functions import functions


# ================================================================
# Configuration
# ================================================================

G_MAXIM = 50  # Maximum generations
NP = 20  # Population size - number of candidate solutions
F = 0.5  # Mutation factor - controls how different mutant is from base
CR = 0.5  # Crossover rate - probability of taking parameter from mutant
DIMENSION = 2  # Problem dimension (2D for visualization)

# Animation settings
ANIMATION_INTERVAL = 1  # milliseconds between frames

# Global reference for animation
ANIM: FuncAnimation | None = None


# ================================================================
# Differential Evolution Core Algorithm
# ================================================================


class Solution:
    def __init__(
        self,
        dimension: int,
        lower_bound: float,
        upper_bound: float,
        fitness_func,
    ):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.fitness_func = fitness_func
        # Initialize with random values within bounds
        self.params = np.random.uniform(lower_bound, upper_bound, dimension)
        self.fitness = fitness_func(self.params)

    @staticmethod
    def deep_copy(
        params: np.ndarray,
        fitness_func,
        lower_bound: float,
        upper_bound: float,
    ) -> "Solution":
        sol = Solution(len(params), lower_bound, upper_bound, fitness_func)
        sol.params = params
        sol.fitness = fitness_func(params)
        return sol


def run_differential_evolution(
    fitness_func,
    dimension: int,
    lower_bound: float,
    upper_bound: float,
    np_size: int = NP,
    f_mutation: float = F,
    cr_crossover: float = CR,
    max_generations: int = G_MAXIM,
):
    # Inicializace N random řešení v populaci
    population: List[Solution] = [
        Solution(dimension, lower_bound, upper_bound, fitness_func)
        for _ in range(np_size)
    ]

    print(f"Vychozi velikost populace: {len(population)}")
    print(f"Vychozi nejlepsi fitness: {min(sol.fitness for sol in population):.4f}")

    # Historie pro animaci/vizualizaci
    history_best: List[np.ndarray] = []
    history_scores: List[float] = []
    history_population: List[List[np.ndarray]] = []

    #
    # Hlavni evolucni smycka - iterace pres generace
    #
    for g in range(max_generations):

        # Deep copy momentalni populace
        new_population = [
            Solution.deep_copy(sol.params, fitness_func, lower_bound, upper_bound)
            for sol in population
        ]

        #
        # Pro kazdeho jedince v populaci:
        #
        for i in range(np_size):
            # Current solution (target vector x_i)
            x_i = population[i]

            # ============================================================
            # MUTACE: Vyber 3 nahodnych jedincu
            # ============================================================
            indices = list(range(np_size))
            indices.remove(i)  # ale rX != i
            r1, r2, r3 = np.random.choice(indices, size=3, replace=False)

            # Mutace vektoru
            v = population[r1].params + f_mutation * (
                population[r2].params - population[r3].params
            )

            # ============================================================
            # CROSSOVER: Mix finalniho vektoru s mutantem
            # ============================================================
            # Finalni vektor u:
            u = x_i.params.copy()

            j_rnd = np.random.randint(0, dimension)  # vynuteni mutace

            # Pro kazdy rozmer rozhodneme, jestli vezmeme z mutanta nebo z finalniho vektoru
            for j in range(dimension):
                # Mutace pokud random < CR nebo pokud je to povinne (j == j_rnd)
                if np.random.uniform() < cr_crossover or j == j_rnd:
                    u[j] = v[j]
                else:
                    u[j] = x_i.params[j]

            # bounds check
            u = np.clip(u, lower_bound, upper_bound)

            # evaluace fitness finalniho vektoru
            f_u = fitness_func(u)

            # uloz lepsi reseni do nove populace
            if f_u <= x_i.fitness:
                new_population[i].params = u
                new_population[i].fitness = f_u

        # Update population for next generation
        population = new_population

        # Track the best solution in current generation
        best_idx = min(range(np_size), key=lambda idx: population[idx].fitness)
        best_solution = population[best_idx]

        history_best.append(best_solution.params.copy())
        history_scores.append(best_solution.fitness)
        history_population.append([sol.params.copy() for sol in population])

        print(
            f"Generace {g+1:3d}: Nejlepsi fitness = {best_solution.fitness:.6f}, "
            f"Pozice = {best_solution.params}"
        )

    print(f"\nUplne nejlepsi fitness: {history_scores[-1]:.6f}")
    print(f"Uplne nejlepsi pozice: {history_best[-1]}")

    return history_best, history_scores, history_population


# ================================================================
# Visualization
# ================================================================


def create_function_mesh(
    func,
    lower_bound: float,
    upper_bound: float,
    resolution: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 2D mesh grid for visualizing the objective function.

    Args:
        func: Function to evaluate (expects vectorized numpy function)
        lower_bound: Minimum coordinate value
        upper_bound: Maximum coordinate value
        resolution: Number of points per dimension

    Returns:
        X, Y: Coordinate grids
        Z: Function values at each grid point
    """
    x = np.linspace(lower_bound, upper_bound, resolution)
    y = np.linspace(lower_bound, upper_bound, resolution)
    X, Y = np.meshgrid(x, y)

    # Create coordinate array for vectorized evaluation
    # Stack X and Y along the last dimension to get shape (resolution, resolution, 2)
    XY = np.stack([X, Y], axis=-1)

    # Evaluate function (functions.py functions are already vectorized)
    Z = func(XY)

    return X, Y, Z


def animate_de_3d(
    fitness_func,
    history_best: List[np.ndarray],
    history_scores: List[float],
    history_population: List[List[np.ndarray]],
    lower_bound: float,
    upper_bound: float,
    func_name: str = "Function",
    interval_ms: int = ANIMATION_INTERVAL,
) -> FuncAnimation:
    """
    Create 3D animation showing the DE optimization process.

    Displays:
    - Left: 3D surface plot with population points evolving over generations
    - Right: Convergence plot showing best fitness over time
    """
    # Create mesh for function surface
    X, Y, Z = create_function_mesh(fitness_func, lower_bound, upper_bound)

    # Setup figure with two subplots
    fig = plt.figure(figsize=(16, 7))
    ax_3d = fig.add_subplot(121, projection="3d")
    ax_convergence = fig.add_subplot(122)

    # Configure 3D plot
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("f(x, y)")
    ax_3d.set_title(f"Differential Evolution - {func_name}")

    # Plot function surface
    ax_3d.plot_surface(
        X, Y, Z, cmap="viridis", alpha=0.6, edgecolor="none", antialiased=True
    )

    # Initialize population scatter plot (red dots for population members)
    pop_scatter = ax_3d.scatter([], [], [], c="red", marker="o", s=50, alpha=0.8)

    # Initialize best solution marker (yellow star)
    best_scatter = ax_3d.scatter(
        [], [], [], c="yellow", marker="*", s=200, edgecolors="black", linewidths=2
    )

    # Add generation text
    gen_text = ax_3d.text2D(
        0.05,
        0.95,
        "",
        transform=ax_3d.transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Configure convergence plot
    ax_convergence.set_xlabel("Generation")
    ax_convergence.set_ylabel("Best Fitness")
    ax_convergence.set_title("Convergence Plot")
    ax_convergence.grid(True, alpha=0.3)
    ax_convergence.set_xlim(0, len(history_scores))

    # Set y-axis limits with some padding
    min_score = min(history_scores)
    max_score = max(history_scores)
    if np.isclose(min_score, max_score):
        ax_convergence.set_ylim(min_score - 1, max_score + 1)
    else:
        ax_convergence.set_ylim(min_score * 0.95, max_score * 1.05)

    (conv_line,) = ax_convergence.plot([], [], "b-", linewidth=2, label="Best Fitness")
    (conv_point,) = ax_convergence.plot([], [], "ro", markersize=8)
    ax_convergence.legend()

    def init():
        """Initialize animation."""
        return update(0)

    def update(frame: int):
        """Update animation for given frame."""
        # Get current generation data
        population = history_population[frame]
        best_params = history_best[frame]

        # Evaluate z-coordinates for population
        pop_array = np.array(population)
        # For functions from functions.py, we need to evaluate them properly
        z_values = np.array([fitness_func(params) for params in population])

        # Update population scatter
        pop_scatter._offsets3d = (pop_array[:, 0], pop_array[:, 1], z_values)

        # Update best solution marker
        best_z = fitness_func(best_params)
        best_scatter._offsets3d = (
            [best_params[0]],
            [best_params[1]],
            [best_z],
        )

        # Update generation text
        gen_text.set_text(
            f"Generation: {frame + 1}/{len(history_best)}\n"
            f"Best Fitness: {history_scores[frame]:.6f}"
        )

        # Update convergence plot
        conv_line.set_data(range(frame + 1), history_scores[: frame + 1])
        conv_point.set_data([frame], [history_scores[frame]])

        return pop_scatter, best_scatter, gen_text, conv_line, conv_point

    anim = FuncAnimation(
        fig,
        update,
        frames=len(history_best),
        init_func=init,
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    plt.tight_layout()
    return anim


# ================================================================
# Main Execution
# ================================================================


def main():
    """Run DE on different test functions."""
    global ANIM

    parser = argparse.ArgumentParser(
        description="Run Differential Evolution on test functions."
    )
    parser.add_argument(
        "--function",
        type=str,
        help="Name of the function to optimize (e.g., 'Sphere', 'Rastrigin')",
        required=False,
    )
    args = parser.parse_args()

    # Filter functions based on argument
    selected_functions = [
        f for f in functions if args.function is None or f[2] == args.function
    ]

    if not selected_functions:
        print(f"No matching function found for '{args.function}'.")
        print("Available functions:", [f[2] for f in functions])
        exit(1)

    # Run DE on each selected function
    for func, bounds, func_name in selected_functions:
        lo, hi = bounds
        print(f"\n{'='*60}")
        print(f"Optimizing: {func_name}")
        print(f"Bounds: [{lo}, {hi}]")
        print(f"Parameters: NP={NP}, F={F}, CR={CR}, G_maxim={G_MAXIM}")
        print(f"{'='*60}\n")

        # Run the algorithm
        history_best, history_scores, history_population = run_differential_evolution(
            fitness_func=func,
            dimension=DIMENSION,
            lower_bound=lo,
            upper_bound=hi,
            np_size=NP,
            f_mutation=F,
            cr_crossover=CR,
            max_generations=G_MAXIM,
        )

        # Create and show animation
        ANIM = animate_de_3d(
            func,
            history_best,
            history_scores,
            history_population,
            lo,
            hi,
            func_name=func_name,
        )

        plt.show()


if __name__ == "__main__":
    main()
