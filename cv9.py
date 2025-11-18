"""
Teaching-Learning Based Optimization (TLBO)

Simuluje proces uceni ve tride, kde zaci se uci od ucitele a navzajem od sebe.
Algoritmus ma dve faze:
1. Teacher Phase: Studenti se uci od nejlepsiho studenta (ucitele)
2. Learner Phase: Studenti se uci navzajem jeden od druheho

Nema zadne specificke parametry (jako F, CR, c1, c2), jen velikost populace a pocet iteraci.
"""

from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import argparse
from functions import functions


# Parametry algoritmu
G_MAX = 50  # Maximalni pocet generaci
POP_SIZE = 30  # Pocet studentu (velikost populace)
DIMENSION = 2  # Dimenze problemu (2D pro vizualizaci)

# Animation settings
ANIMATION_INTERVAL = 1  # milliseconds between frames

# Global reference for animation
ANIM: FuncAnimation | None = None


# ================================================================
# TLBO Core Algorithm
# ================================================================


class Student:
    """Reprezentuje jednoho studenta v populaci."""

    def __init__(
        self,
        dimension: int,
        lower_bound: float,
        upper_bound: float,
        fitness_func,
    ):
        # Nahodna pocatecni pozice (reseni)
        self.position = np.random.uniform(lower_bound, upper_bound, dimension)
        self.fitness = fitness_func(self.position)


def run_tlbo(
    fitness_func,
    dimension: int,
    lower_bound: float,
    upper_bound: float,
    pop_size: int = POP_SIZE,
    max_generations: int = G_MAX,
):
    """
    Teaching-Learning Based Optimization Algorithm

    Args:
        fitness_func: Objective function to minimize
        dimension: Problem dimension
        lower_bound: Lower bound of search space
        upper_bound: Upper bound of search space
        pop_size: Population size (number of students)
        max_generations: Maximum number of generations
    """

    # Inicializace populace studentu
    population = [
        Student(dimension, lower_bound, upper_bound, fitness_func)
        for _ in range(pop_size)
    ]

    print(f"Pocatecni velikost populace: {len(population)}")
    print(f"Pocatecni nejlepsi fitness: {min(s.fitness for s in population):.4f}")

    # Historie pro animaci/vizualizaci
    history_best: List[np.ndarray] = []
    history_scores: List[float] = []
    history_population: List[List[np.ndarray]] = []

    #
    # Hlavni evolucni smycka
    #
    for gen in range(max_generations):

        # ============================================================
        # TEACHER PHASE
        # ============================================================

        # Najdi nejlepsiho studenta (ucitele)
        teacher_idx = min(range(pop_size), key=lambda i: population[i].fitness)
        teacher = population[teacher_idx]

        # Vypocitaj prumer populace (mean)
        mean_position = np.mean([s.position for s in population], axis=0)

        # Pro kazdeho studenta
        for i in range(pop_size):
            student = population[i]

            # Teaching Factor TF - muze byt 1 nebo 2 (nahodne)
            TF = np.random.randint(1, 3)  # 1 nebo 2

            # Vypocitej rozdil mezi ucitelem a prumerem
            # Difference = r * (X_teacher - TF * M)
            r = np.random.uniform(0, 1, dimension)
            difference = r * (teacher.position - TF * mean_position)

            # Nova pozice studenta po uceni od ucitele
            # X_new = X_old + Difference
            new_position = student.position + difference

            # Kontrola hranic
            new_position = np.clip(new_position, lower_bound, upper_bound)

            # Vyhodnot novou pozici
            new_fitness = fitness_func(new_position)

            # Prijmi novou pozici pokud je lepsi
            if new_fitness < student.fitness:
                student.position = new_position
                student.fitness = new_fitness

        # ============================================================
        # LEARNER PHASE
        # ============================================================

        # Pro kazdeho studenta
        for i in range(pop_size):
            student_i = population[i]

            # Vyber nahodne jineho studenta
            j = i
            while j == i:
                j = np.random.randint(0, pop_size)
            student_j = population[j]

            # Podle toho, kdo je lepsi, se student uci
            r = np.random.uniform(0, 1, dimension)

            if student_j.fitness < student_i.fitness:
                # Student j je lepsi, student i se uci od nej
                # X_new,i = X_i + r * (X_j - X_i)
                new_position = student_i.position + r * (
                    student_j.position - student_i.position
                )
            else:
                # Student i je lepsi nebo stejny
                # X_new,i = X_i + r * (X_i - X_j)
                new_position = student_i.position + r * (
                    student_i.position - student_j.position
                )

            # Kontrola hranic
            new_position = np.clip(new_position, lower_bound, upper_bound)

            # Vyhodnot novou pozici
            new_fitness = fitness_func(new_position)

            # Prijmi novou pozici pokud je lepsi
            if new_fitness < student_i.fitness:
                student_i.position = new_position
                student_i.fitness = new_fitness

        # ============================================================
        # Zaznamenej nejlepsiho studenta v teto generaci
        # ============================================================
        best_idx = min(range(pop_size), key=lambda i: population[i].fitness)
        best_student = population[best_idx]

        history_best.append(best_student.position.copy())
        history_scores.append(best_student.fitness)
        history_population.append([s.position.copy() for s in population])

        print(
            f"Generace {gen+1:3d}: Nejlepsi fitness = {best_student.fitness:.6f}, "
            f"Pozice = {best_student.position}"
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
    XY = np.stack([X, Y], axis=-1)

    # Evaluate function (functions.py functions are already vectorized)
    Z = func(XY)

    return X, Y, Z


def animate_tlbo_3d(
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
    Create 3D animation showing the TLBO optimization process.

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
    ax_3d.set_title(f"TLBO - {func_name}")

    # Plot function surface
    ax_3d.plot_surface(
        X, Y, Z, cmap="viridis", alpha=0.6, edgecolor="none", antialiased=True
    )

    # Initialize population scatter plot (green dots for students)
    pop_scatter = ax_3d.scatter([], [], [], c="green", marker="o", s=50, alpha=0.8)

    # Initialize best solution marker (red star for teacher)
    best_scatter = ax_3d.scatter(
        [], [], [], c="red", marker="*", s=200, edgecolors="black", linewidths=2
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
        z_values = np.array([fitness_func(params) for params in population])

        # Update population scatter
        pop_scatter._offsets3d = (pop_array[:, 0], pop_array[:, 1], z_values)

        # Update best solution marker (teacher)
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
    """Run TLBO on different test functions."""
    global ANIM

    parser = argparse.ArgumentParser(
        description="Run Teaching-Learning Based Optimization on test functions."
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

    # Run TLBO on each selected function
    for func, bounds, func_name in selected_functions:
        lo, hi = bounds
        print(f"\n{'='*60}")
        print(f"Optimizing: {func_name}")
        print(f"Bounds: [{lo}, {hi}]")
        print(f"Parameters: POP_SIZE={POP_SIZE}, G_max={G_MAX}")
        print(f"{'='*60}\n")

        # Run the algorithm
        history_best, history_scores, history_population = run_tlbo(
            fitness_func=func,
            dimension=DIMENSION,
            lower_bound=lo,
            upper_bound=hi,
            pop_size=POP_SIZE,
            max_generations=G_MAX,
        )

        # Create and show animation
        ANIM = animate_tlbo_3d(
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
