"""
Particle Swarm Optimization (PSO)

Simuluje hejna castic, ktere hledaji optimum funkce.
Kazda castice ma:
- pozici (reseni)
- rychlost (smer pohybu)
- pamet nejlepsi vlastni pozice (pBest)
Hejno sdili globalne nejlepsi pozici (gBest)
"""

from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import argparse
from functions import functions


# Parametry algoritmu
M_MAX = 50  # Maximalni pocet iteraci
POP_SIZE = 15  # Pocet castic v hejnu
C1 = 2.0  # Cognitive component - tihnuti k vlastnimu nejlepsimu
C2 = 2.0  # Social component - tihnuti ke globalnimu nejlepsimu
DIMENSION = 2  # Dimenze problemu (2D pro vizualizaci)
W_START = 0.9  # Pocatecni vaha inercie (explorace)
W_END = 0.4  # Koncova vaha inercie (exploatace)
ANIMATION_INTERVAL = 1

ANIM: FuncAnimation | None = None


class Particle:
    """Reprezentuje jednu castici v hejnu."""

    def __init__(
        self,
        dimension: int,
        lower_bound: float,
        upper_bound: float,
        v_mini: float,
        v_maxi: float,
        fitness_func,
    ):
        # Nahodna pocatecni pozice v prostoru
        self.position = np.random.uniform(lower_bound, upper_bound, dimension)
        # Nahodna pocatecni rychlost
        self.velocity = np.random.uniform(v_mini, v_maxi, dimension)
        # Nejlepsi dosazena pozice teto castice
        self.p_best = self.position.copy()
        self.p_best_fitness = fitness_func(self.position)
        self.fitness = self.p_best_fitness


def run_pso(
    fitness_func,
    dimension: int,
    lower_bound: float,
    upper_bound: float,
    pop_size: int = POP_SIZE,
    c1: float = C1,
    c2: float = C2,
    max_iterations: int = M_MAX,
    w_start: float = W_START,
    w_end: float = W_END,
):

    # Omezeni rychlosti (20% rozsahu prostoru)
    v_range = (upper_bound - lower_bound) * 0.2
    v_mini, v_maxi = -v_range, v_range

    # Vytvor hejno castic s nahodnymi pozicemi
    swarm = [
        Particle(dimension, lower_bound, upper_bound, v_mini, v_maxi, fitness_func)
        for _ in range(pop_size)
    ]

    # Najdi globalne nejlepsi pozici ze vsech castic (gBest)
    g_best_idx = min(range(pop_size), key=lambda i: swarm[i].p_best_fitness)
    g_best = swarm[g_best_idx].p_best.copy()
    g_best_fitness = swarm[g_best_idx].p_best_fitness

    print(f"Pocatecni nejlepsi fitness: {g_best_fitness:.4f}")

    # Historie pro vizualizaci
    history_best = []
    history_scores = []
    history_population = []

    # Hlavni smycka PSO
    m = 0
    while m < max_iterations:
        # inercie = setrvacnost
        # Linearne snizuj vahu inercie (z explorace do exploatace)
        w = w_start - (w_start - w_end) * (m / max_iterations)

        # Pro kazdou casticu v hejnu
        for particle in swarm:
            # Nahodne vahy pro stochasticke chovani
            r1 = np.random.uniform(0, 1, dimension)
            r2 = np.random.uniform(0, 1, dimension)

            # Vypocet nove rychlosti podle PSO vzorce:
            # v = w*v + c1*r1*(pBest - x) + c2*r2*(gBest - x)
            particle.velocity = (
                w * particle.velocity
                + c1 * r1 * (particle.p_best - particle.position)
                + c2 * r2 * (g_best - particle.position)
            )

            particle.velocity = np.clip(particle.velocity, v_mini, v_maxi)

            # Vypocti novou pozici: x_new = x_old + v (VZDY nahrad starou!)
            particle.position = np.clip(
                particle.position + particle.velocity, lower_bound, upper_bound
            )

            # Vyhodnot fitness v nove pozici
            particle.fitness = fitness_func(particle.position)

            # Pokud je nova pozice lepsi nez osobni nejlepsi (pBest)
            if particle.fitness < particle.p_best_fitness:
                particle.p_best = particle.position.copy()
                particle.p_best_fitness = particle.fitness

                # Pokud je pBest lepsi nez globalni nejlepsi (gBest)
                if particle.p_best_fitness < g_best_fitness:
                    g_best = particle.p_best.copy()
                    g_best_fitness = particle.p_best_fitness

        m += 1
        history_best.append(g_best.copy())
        history_scores.append(g_best_fitness)
        history_population.append([p.position.copy() for p in swarm])

        print(f"Iterace {m:3d}: Fitness = {g_best_fitness:.6f}, Pozice = {g_best}")

    print(f"\nNejlepsi fitness: {g_best_fitness:.6f}")
    print(f"Nejlepsi pozice: {g_best}")

    return history_best, history_scores, history_population


def create_function_mesh(
    func, lower_bound: float, upper_bound: float, resolution: int = 500
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vytvori 2D sit bodu pro vizualizaci funkce."""
    x = np.linspace(lower_bound, upper_bound, resolution)
    y = np.linspace(lower_bound, upper_bound, resolution)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X, Y], axis=-1)
    Z = func(XY)
    return X, Y, Z


def animate_pso_3d(
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
    Vytvori 3D animaci PSO procesu.
    Vlevo: 3D povrch funkce s hejnem castic
    Vpravo: Graf konvergence (nejlepsi fitness v case)
    """
    X, Y, Z = create_function_mesh(fitness_func, lower_bound, upper_bound)

    # Vytvor figure se dvema grafy
    fig = plt.figure(figsize=(16, 7))
    ax_3d = fig.add_subplot(121, projection="3d")
    ax_conv = fig.add_subplot(122)

    # Nastav 3D graf s povrchem funkce
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("f(x, y)")
    ax_3d.set_title(f"PSO - {func_name}")
    ax_3d.plot_surface(X, Y, Z, cmap="viridis", alpha=0.6, edgecolor="none")

    # Modre tecky = castice v hejnu
    swarm_scatter = ax_3d.scatter([], [], [], c="blue", marker="o", s=50, alpha=0.8)
    # Zluta hvezda = globalne nejlepsi pozice
    best_scatter = ax_3d.scatter(
        [], [], [], c="yellow", marker="*", s=200, edgecolors="black", linewidths=2
    )
    iter_text = ax_3d.text2D(
        0.05,
        0.95,
        "",
        transform=ax_3d.transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel("Best Fitness")
    ax_conv.set_title("Convergence")
    ax_conv.grid(True, alpha=0.3)
    ax_conv.set_xlim(0, len(history_scores))

    min_score, max_score = min(history_scores), max(history_scores)
    if np.isclose(min_score, max_score):
        ax_conv.set_ylim(min_score - 1, max_score + 1)
    else:
        ax_conv.set_ylim(min_score * 0.95, max_score * 1.05)

    (conv_line,) = ax_conv.plot([], [], "b-", linewidth=2)
    (conv_point,) = ax_conv.plot([], [], "ro", markersize=8)

    def update(frame: int):
        """Aktualizuj animaci pro dany frame (iteraci)."""
        population = history_population[frame]
        best_params = history_best[frame]

        # Vyhodnot Z souradnice pro vsechny castice
        pop_array = np.array(population)
        z_values = np.array([fitness_func(params) for params in population])

        # Aktualizuj pozice hejna
        swarm_scatter._offsets3d = (pop_array[:, 0], pop_array[:, 1], z_values)

        # Aktualizuj pozici gBest
        best_z = fitness_func(best_params)
        best_scatter._offsets3d = ([best_params[0]], [best_params[1]], [best_z])

        # Aktualizuj text s cislem iterace
        iter_text.set_text(
            f"Iterace: {frame + 1}/{len(history_best)}\n"
            f"Fitness: {history_scores[frame]:.6f}"
        )

        # Aktualizuj konvergencni krivku
        conv_line.set_data(range(frame + 1), history_scores[: frame + 1])
        conv_point.set_data([frame], [history_scores[frame]])

        return swarm_scatter, best_scatter, iter_text, conv_line, conv_point

    anim = FuncAnimation(
        fig,
        update,
        frames=len(history_best),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    plt.tight_layout()
    return anim


def main():
    global ANIM

    parser = argparse.ArgumentParser(description="PSO optimization")
    parser.add_argument(
        "--function", type=str, help="Function to optimize", required=False
    )
    args = parser.parse_args()

    selected_functions = [
        f for f in functions if args.function is None or f[2] == args.function
    ]

    if not selected_functions:
        print(f"Function '{args.function}' not found.")
        print("Available:", [f[2] for f in functions])
        exit(1)

    for func, bounds, func_name in selected_functions:
        lo, hi = bounds
        print(f"\n{'='*60}")
        print(f"Optimizing: {func_name}")
        print(f"Bounds: [{lo}, {hi}]")
        print(f"pop_size={POP_SIZE}, c1={C1}, c2={C2}, M_max={M_MAX}")
        print(f"{'='*60}\n")

        history_best, history_scores, history_population = run_pso(
            func, DIMENSION, lo, hi, POP_SIZE, C1, C2, M_MAX, W_START, W_END
        )

        ANIM = animate_pso_3d(
            func, history_best, history_scores, history_population, lo, hi, func_name
        )
        plt.show()


if __name__ == "__main__":
    main()
