"""
Firefly Algorithm (FA)

Simuluje hejno svetlusek, ktere hledaji optimum funkce.
Kazda svetluska ma:
- pozici (reseni)
- svetelnou intenzitu (urcena fitness funkci)
Svetlusky se pohybuji smerem k svetlejsim svetluskam.
Atraktivita klesa se vzdalenosti.
"""

from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import argparse
from functions import functions


# Parametry algoritmu
MAX_GENERATIONS = 50  # Maximalni pocet generaci
POP_SIZE = 20  # Pocet svetlusek v populaci
BETA_0 = 1.0  # Maximalni atraktivita (na vzdalenosti 0)
GAMMA = 1.0  # Absorpcni koeficient (ovlivnuje rychlost konvergence)
ALPHA = 0.3  # Krok nahodneho pohybu (0 az 1)
DIMENSION = 2  # Dimenze problemu (2D pro vizualizaci)
ANIMATION_INTERVAL = 100

ANIM: FuncAnimation | None = None


class Firefly:
    """Reprezentuje jednu svetlusku v populaci."""

    def __init__(
        self,
        dimension: int,
        lower_bound: float,
        upper_bound: float,
        fitness_func,
    ):
        # Nahodna pocatecni pozice v prostoru
        self.position = np.random.uniform(lower_bound, upper_bound, dimension)
        # Svetelna intenzita je urcena fitness funkci
        self.light_intensity = fitness_func(self.position)
        self.fitness = self.light_intensity


def calculate_attractiveness(beta_0: float, gamma: float, distance: float) -> float:
    """
    Vypocita atraktivitu na zaklade vzdalenosti.
    """
    return beta_0 / (1.0 + gamma * distance**2)


def run_firefly_algorithm(
    fitness_func,
    dimension: int,
    lower_bound: float,
    upper_bound: float,
    pop_size: int = POP_SIZE,
    beta_0: float = BETA_0,
    gamma: float = GAMMA,
    alpha: float = ALPHA,
    max_generations: int = MAX_GENERATIONS,
):
    """
    Hlavni algoritmus
    """
    # Initicalni populace svetlusek s nahodnymi pozicemi
    population = [
        Firefly(dimension, lower_bound, upper_bound, fitness_func)
        for _ in range(pop_size)
    ]

    # nejlepsi svetluskA (s nejnizsi fitness - nejvetsi svetelnou intenzitou)
    best_idx = min(range(pop_size), key=lambda i: population[i].fitness)
    best_fitness = population[best_idx].fitness
    best_position = population[best_idx].position.copy()

    print(f"Pocatecni nejlepsi fitness: {best_fitness:.4f}")

    # Historie
    history_best = []
    history_scores = []
    history_population = []

    #
    # Hlavni smycka Firefly Algorithm
    #
    for t in range(max_generations):
        # Pro kazdou svetlusku i
        for i in range(pop_size):
            firefly_i = population[i]

            # Porovnej s kazdou jinou svetluskou j v teto iteraci
            for j in range(pop_size):
                if i == j:
                    continue  # skipuju current one

                firefly_j = population[j]

                #
                # Pokud je svetluska j svetlejsi (ma nizsi fitness) nez i
                #
                if firefly_j.light_intensity < firefly_i.light_intensity:
                    # Vypocti vzdalenost mezi svetluskami (Euklidovska vzdalenost)
                    distance = np.linalg.norm(
                        firefly_i.position - firefly_j.position)

                    # Vypocti atraktivitu na zaklade vzdalenosti
                    beta = calculate_attractiveness(beta_0, gamma, distance)

                    # Vygeneruj nahodny krok z normalniho rozdeleni
                    epsilon = np.random.normal(0, 1, dimension)

                    #
                    # Pohyb svetlusky i smerem k svetlusky j
                    # x_i = x_i + beta * (x_j - x_i) + alpha * epsilon
                    #
                    firefly_i.position = (
                        firefly_i.position
                        + beta * (firefly_j.position - firefly_i.position)
                        + alpha * epsilon
                    )

                    # Omez pozici na hranice prostoru
                    firefly_i.position = np.clip(
                        firefly_i.position, lower_bound, upper_bound
                    )

                    # Aktualizuj svetelnou intenzitu
                    firefly_i.light_intensity = fitness_func(
                        firefly_i.position)
                    firefly_i.fitness = firefly_i.light_intensity

                    # Aktualizuj globalne nejlepsi pozici
                    if firefly_i.fitness < best_fitness:
                        best_fitness = firefly_i.fitness
                        best_position = firefly_i.position.copy()

        history_best.append(best_position.copy())
        history_scores.append(best_fitness)
        history_population.append([f.position.copy() for f in population])

        print(
            f"Generace {t + 1:3d}: Fitness = {best_fitness:.6f}, Pozice = {
                best_position
            }"
        )

    print(f"\nNejlepsi fitness: {best_fitness:.6f}")
    print(f"Nejlepsi pozice: {best_position}")

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


def animate_firefly_3d(
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
    Vytvori 3D animaci Firefly Algorithm procesu.
    Vlevo: 3D povrch funkce s populaci svetlusek
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
    ax_3d.set_title(f"Firefly Algorithm - {func_name}")
    ax_3d.plot_surface(X, Y, Z, cmap="viridis", alpha=0.6, edgecolor="none")

    # Bile tecky s cervenym okrajem = svetlusky v populaci (lepe videt nez zluta)
    firefly_scatter = ax_3d.scatter(
        [],
        [],
        [],
        c="white",
        marker="o",
        s=60,
        alpha=0.9,
        edgecolors="red",
        linewidths=1.5,
    )
    # Oranzova hvezda = nejlepsi svetluska
    best_scatter = ax_3d.scatter(
        [], [], [], c="orange", marker="*", s=250, edgecolors="black", linewidths=2
    )
    iter_text = ax_3d.text2D(
        0.05,
        0.95,
        "",
        transform=ax_3d.transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Graf konvergence
    ax_conv.set_xlabel("Generation")
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
        """Aktualizuj animaci pro dany frame (generaci)."""
        population = history_population[frame]
        best_params = history_best[frame]

        # Vyhodnot Z souradnice pro vsechny svetlusky
        pop_array = np.array(population)
        z_values = np.array([fitness_func(params) for params in population])

        # Aktualizuj pozice svetlusek
        firefly_scatter._offsets3d = (
            pop_array[:, 0], pop_array[:, 1], z_values)

        # Aktualizuj pozici nejlepsi svetlusky
        best_z = fitness_func(best_params)
        best_scatter._offsets3d = (
            [best_params[0]], [best_params[1]], [best_z])

        # Aktualizuj text s cislem generace
        iter_text.set_text(
            f"Generace: {frame + 1}/{len(history_best)}\n"
            f"Fitness: {history_scores[frame]:.6f}"
        )

        # Aktualizuj konvergencni krivku
        conv_line.set_data(range(frame + 1), history_scores[: frame + 1])
        conv_point.set_data([frame], [history_scores[frame]])

        return firefly_scatter, best_scatter, iter_text, conv_line, conv_point

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

    parser = argparse.ArgumentParser(
        description="Firefly Algorithm optimization")
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
        print(f"\n{'=' * 60}")
        print(f"Optimizing: {func_name}")
        print(f"Bounds: [{lo}, {hi}]")
        print(
            f"pop_size={POP_SIZE}, beta_0={BETA_0}, gamma={GAMMA}, alpha={
                ALPHA
            }, max_gen={MAX_GENERATIONS}"
        )
        print(f"{'=' * 60}\n")

        history_best, history_scores, history_population = run_firefly_algorithm(
            func, DIMENSION, lo, hi, POP_SIZE, BETA_0, GAMMA, ALPHA, MAX_GENERATIONS
        )

        ANIM = animate_firefly_3d(
            func,
            history_best,
            history_scores,
            history_population,
            lo,
            hi,
            func_name,
        )
        plt.show()


if __name__ == "__main__":
    main()
