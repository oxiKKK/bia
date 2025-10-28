"""
Self-Organizing Migrating Algorithm (SOMA) - AllToOne

simuluje hejno organismu, ktere hledaji optimum funkce.
kazdy jedinec (Specimen) migruje smerem k nejlepsimu jedinci (Leader).
behem migrace se pohybuje po cestach s urcitym krokem (Step).
PRT (Perturbation) vector nahodne rozhoduje, ktere dimenze budou aktivni.
"""

from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import argparse
from functions import functions


# parametry algoritmu
M_MAX = 100  # maximalni pocet migraci
POP_SIZE = 20  # pocet jedincu v populaci
PRT = 0.4  # perturbation - pravdepodobnost aktivity dimenze
PATH_LENGTH = 3.0  # delka migracni cesty
STEP = 0.11  # krok po ceste
DIMENSION = 2  # dimenze problemu (2D pro vizualizaci)
ANIMATION_INTERVAL = 1000

ANIM: FuncAnimation | None = None


class Specimen:
    """Reprezentuje jednoho jedince v populaci."""

    def __init__(
        self,
        dimension: int,
        lower_bound: float,
        upper_bound: float,
        fitness_func,
    ):
        # nahodna pocatecni pozice v prostoru
        self.position = np.random.uniform(lower_bound, upper_bound, dimension)
        self.fitness = fitness_func(self.position)


def run_soma(
    fitness_func,
    dimension: int,
    lower_bound: float,
    upper_bound: float,
    pop_size: int = POP_SIZE,
    prt: float = PRT,
    path_length: float = PATH_LENGTH,
    step: float = STEP,
    max_migrations: int = M_MAX,
):
    # vytvor populaci jedincu s nahodnymi pozicemi
    population = [
        Specimen(dimension, lower_bound, upper_bound, fitness_func)
        for _ in range(pop_size)
    ]

    # najdi nejlepsiho jedince (Leader)
    leader_idx = min(range(pop_size), key=lambda i: population[i].fitness)
    best_fitness = population[leader_idx].fitness
    best_position = population[leader_idx].position.copy()

    print(f"Pocatecni nejlepsi fitness: {best_fitness:.4f}")

    # historie pro vizualizaci
    history_best = []
    history_scores = []
    history_population = []
    history_paths = []  # cesty migrace pro vizualizaci

    # hlavni smycka SOMA
    migrations = 0
    while migrations < max_migrations:
        # najdi aktualniho leadera
        leader_idx = min(range(pop_size), key=lambda i: population[i].fitness)
        leader_position = population[leader_idx].position.copy()

        migration_paths = []

        # kazdy jedinec (krome leadera) migruje k leaderovi
        for i in range(pop_size):
            if i == leader_idx:
                continue  # leader se nehybe

            specimen = population[i]
            start_position = specimen.position.copy()
            start_fitness = specimen.fitness

            # vygeneruj PRT vector (nahodne urcuje, ktere dimenze budou aktivni)
            prt_vector = (np.random.random(dimension) < prt).astype(float)

            # sleduj cestu migrace
            path_positions = [start_position.copy()]

            # migrace po ceste k leaderovi
            t = 0.0
            best_position_on_path = start_position.copy()
            best_fitness_on_path = start_fitness

            # iteruj po krocich na ceste
            while t <= path_length:
                # vypocti novou pozici vzhledem k leaderovi
                # t = jedinec se priblizuje leaderovi
                new_position = (
                    start_position + (leader_position - start_position) * t * prt_vector
                )

                new_position = np.clip(new_position, lower_bound, upper_bound)

                # vyhodnot fitness v nove pozici
                new_fitness = fitness_func(new_position)

                path_positions.append(new_position.copy())

                # uloz nejlepsi pozici na ceste
                if new_fitness < best_fitness_on_path:
                    best_fitness_on_path = new_fitness
                    best_position_on_path = new_position.copy()

                t += step

            migration_paths.append(path_positions)

            # po migraci: pokud je nejlepsi pozice na ceste lepsi nez start, pouzij ji
            if best_fitness_on_path < start_fitness:
                specimen.position = best_position_on_path
                specimen.fitness = best_fitness_on_path

                # aktualizuj globalniho leadera
                if specimen.fitness < best_fitness:
                    best_fitness = specimen.fitness
                    best_position = specimen.position.copy()

        migrations += 1
        history_best.append(best_position.copy())
        history_scores.append(best_fitness)
        history_population.append([s.position.copy() for s in population])
        history_paths.append(migration_paths)

        print(
            f"Migrace {migrations:3d}: Fitness = {best_fitness:.6f}, Pozice = {best_position}"
        )

    print(f"\nNejlepsi fitness: {best_fitness:.6f}")
    print(f"Nejlepsi pozice: {best_position}")

    return history_best, history_scores, history_population, history_paths


def create_function_mesh(
    func, lower_bound: float, upper_bound: float, resolution: int = 500
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """vytvori 2D sit bodu pro vizualizaci funkce."""
    x = np.linspace(lower_bound, upper_bound, resolution)
    y = np.linspace(lower_bound, upper_bound, resolution)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X, Y], axis=-1)
    Z = func(XY)
    return X, Y, Z


def animate_soma_3d(
    fitness_func,
    history_best: List[np.ndarray],
    history_scores: List[float],
    history_population: List[List[np.ndarray]],
    history_paths: List[List[List[np.ndarray]]],
    lower_bound: float,
    upper_bound: float,
    func_name: str = "Function",
    interval_ms: int = ANIMATION_INTERVAL,
) -> FuncAnimation:
    """
    vytvori 3D animaci SOMA procesu.
    vlevo: 3D povrch funkce s populaci a migracnimi cestami
    vpravo: graf konvergence (nejlepsi fitness v case)
    """
    X, Y, Z = create_function_mesh(fitness_func, lower_bound, upper_bound)

    # vytvor figure se dvema grafy
    fig = plt.figure(figsize=(16, 7))
    ax_3d = fig.add_subplot(121, projection="3d")
    ax_conv = fig.add_subplot(122)

    # nastav 3D graf s povrchem funkce
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("f(x, y)")
    ax_3d.set_title(f"SOMA AllToOne - {func_name}")
    ax_3d.plot_surface(X, Y, Z, cmap="viridis", alpha=0.6, edgecolor="none")

    # modre tecky = jedinci v populaci
    pop_scatter = ax_3d.scatter([], [], [], c="blue", marker="o", s=50, alpha=0.8)
    # cervena hvezda = leader (nejlepsi jedinec)
    leader_scatter = ax_3d.scatter(
        [], [], [], c="red", marker="*", s=200, edgecolors="black", linewidths=2
    )
    iter_text = ax_3d.text2D(
        0.05,
        0.95,
        "",
        transform=ax_3d.transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # graf konvergence
    ax_conv.set_xlabel("Migration")
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

    # pro kresleni cest migrace
    path_lines = []

    def update(frame: int):
        """aktualizuj animaci pro dany frame (migraci)."""
        nonlocal path_lines

        # smaz predchozi cesty
        for line in path_lines:
            line.remove()
        path_lines = []

        population = history_population[frame]
        best_params = history_best[frame]

        # vyhodnot Z souradnice pro vsechny jedince
        pop_array = np.array(population)
        z_values = np.array([fitness_func(params) for params in population])

        # aktualizuj pozice populace
        pop_scatter._offsets3d = (pop_array[:, 0], pop_array[:, 1], z_values)

        # aktualizuj pozici leadera
        leader_z = fitness_func(best_params)
        leader_scatter._offsets3d = ([best_params[0]], [best_params[1]], [leader_z])

        # aktualizuj text s cislem migrace
        iter_text.set_text(
            f"Migrace: {frame + 1}/{len(history_best)}\n"
            f"Fitness: {history_scores[frame]:.6f}"
        )

        # aktualizuj konvergencni krivku
        conv_line.set_data(range(frame + 1), history_scores[: frame + 1])
        conv_point.set_data([frame], [history_scores[frame]])

        # vykresli migracni cesty (volitelne - muze zpomalovet animaci)
        if frame < len(history_paths):
            for path in history_paths[frame]:
                if len(path) > 1:
                    path_array = np.array(path)
                    path_z = np.array([fitness_func(p) for p in path])
                    line = ax_3d.plot(
                        path_array[:, 0],
                        path_array[:, 1],
                        path_z,
                        "g-",
                        alpha=0.3,
                        linewidth=1,
                    )[0]
                    path_lines.append(line)

        return [
            pop_scatter,
            leader_scatter,
            iter_text,
            conv_line,
            conv_point,
        ] + path_lines

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

    parser = argparse.ArgumentParser(description="SOMA AllToOne optimization")
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
        print(
            f"pop_size={POP_SIZE}, PRT={PRT}, PathLength={PATH_LENGTH}, Step={STEP}, M_max={M_MAX}"
        )
        print(f"{'='*60}\n")

        history_best, history_scores, history_population, history_paths = run_soma(
            func, DIMENSION, lo, hi, POP_SIZE, PRT, PATH_LENGTH, STEP, M_MAX
        )

        ANIM = animate_soma_3d(
            func,
            history_best,
            history_scores,
            history_population,
            history_paths,
            lo,
            hi,
            func_name,
        )
        plt.show()


if __name__ == "__main__":
    main()
