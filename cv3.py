from typing import List, Sequence

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


# ================================================================
# basic configuration
# ================================================================

# pocet mest ktere musime navstivit
CITY_COUNT = 30

# pocet permutaci nahodnych tras
# vyssi hodnota = lepsi vysledek, ale pomalejsi beh
NP = 120

# pocet iteraci (generaci) algoritmu. vyssi hodnota = lepsi vysledek
# ale pomalejsi beh
ITERATIONS = 100

# sleep mezi animacemi v ms
ANIMATION_INTERVAL = 1

# assert CANDIDATE_ROUTES_PER_GENERATION_SIZE > 1

# global reference
ANIM: FuncAnimation | None = None


def generate_cities(num_cities: int) -> np.ndarray:
    """Create random city coordinates in a square grid."""
    rng = np.random.default_rng()
    return rng.uniform(0.0, 200.0, size=(num_cities, 2))


def closed_route(cities: np.ndarray, route: np.ndarray) -> np.ndarray:
    """Return coordinates ordered by route with the first point appended."""
    ordered = cities[route]
    return np.vstack([ordered, ordered[0]])


def route_length(cities: np.ndarray, route: np.ndarray) -> float:
    """Compute the total length of a TSP route returning to the start."""
    ordered = closed_route(cities, route)
    diffs = np.diff(ordered, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


# ================================================================
# Genetic operators
# ================================================================


def ordered_crossover(
    parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Perform ordered crossover producing a valid permutation."""
    n = len(parent_a)
    # nahodne vybereme dva body pro rez
    cut1, cut2 = sorted(rng.integers(0, n, size=2))
    # vytvorime masku pro potomka a zkopirujeme segment z prvniho rodice
    mask = -np.ones(n, dtype=int)
    mask[cut1:cut2] = parent_a[cut1:cut2]

    # zbytek segmentu doplnime hodnotami z druheho rodice, ktere nejsou v masce
    fill_values = [gene for gene in parent_b if gene not in mask]
    fill_iter = iter(fill_values)
    for i in range(n):
        if mask[i] == -1:
            mask[i] = next(fill_iter)

    return mask


def swap_mutation(route: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Swap two random cities in the route."""
    mutated = route.copy()
    i, j = rng.choice(len(route), size=2, replace=False)
    mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


# ================================================================
# core loop
# ================================================================


def visualize_candidate_routes(cities: np.ndarray, candidate_routes: list[np.ndarray]):
    """Visualize the initial candidate routes."""
    _, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(cities[:, 0], cities[:, 1], c="crimson", s=50, zorder=3)
    for idx, (x, y) in enumerate(cities, start=1):
        ax.text(x + 2, y + 2, str(idx), fontsize=9, color="black")

    for route in candidate_routes:
        path = closed_route(cities, route)
        ax.plot(path[:, 0], path[:, 1], "-o", alpha=0.5)

    ax.set_title("Initial Candidate Routes")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    plt.show()


def run_algo(cities: np.ndarray) -> tuple[List[np.ndarray], List[float]]:
    """
    Vice tras najednou = populace.
    Jedna iterace zlepsuje soubor v prumeru = generace.
    Nove trasy z krizeni (crossover) + mutace = napodobi dedicnost.
    """
    rng = np.random.default_rng()
    n_cities = len(cities)
    PROB_MUTATE = 0.5  # z pseudokodu

    # vyber N permutaci jako vychozi populaci
    candidate_routes = [rng.permutation(n_cities) for _ in range(NP)]
    distance = [route_length(cities, route) for route in candidate_routes]

    print(f"Mame {len(candidate_routes)} variant tras.")
    print(candidate_routes[0])

    visualize_candidate_routes(cities, candidate_routes)

    history_routes: List[np.ndarray] = []
    history_scores: List[float] = []

    #
    # hlavni evolucni smycka
    #
    for i in range(ITERATIONS):
        # vem trasu s nejmensi vzdalenosti
        best_idx = int(np.argmin(distance))
        history_routes.append(candidate_routes[best_idx].copy())
        history_scores.append(distance[best_idx])

        new_routes = list(candidate_routes)
        new_distance = list(distance)

        #
        # pro kazdou trasu provedeme offspring a mozna mutaci
        #
        for j in range(NP):
            # vybereme 2 nahodne permutace mest
            parent_A = candidate_routes[j]
            while True:  # Dokud neni jina nez A
                idx_b = int(rng.integers(0, NP))
                if idx_b != j:
                    break
            parent_B = candidate_routes[idx_b]

            # offspring = crossover(A, B)
            offspring = ordered_crossover(parent_A, parent_B, rng)

            # mutate s pravdepodobnosti 0.5
            # jestli vyjde, provedeme swap mezi 2 random mesty
            if rng.random() < PROB_MUTATE:
                offspring = swap_mutation(offspring, rng)

            # je ted vysledna trasa lepsi nez puvodni?
            d = route_length(cities, offspring)
            if d < distance[j]:
                new_routes[j] = offspring
                new_distance[j] = d

        candidate_routes = new_routes
        distance = new_distance
        print(f"{i:02}: Nejlepsi vzdalenost v generaci: {min(distance):.0f} km")

    # zarad do historie nejlepsi jedince
    best_idx = int(np.argmin(distance))
    history_routes.append(candidate_routes[best_idx].copy())
    history_scores.append(distance[best_idx])

    return history_routes, history_scores


# ================================================================
# Visualisation
# ================================================================


def animate_history(
    points: np.ndarray,
    routes: Sequence[np.ndarray],
    scores: Sequence[float],
    interval_ms: int = ANIMATION_INTERVAL,
) -> FuncAnimation:
    """Create an animation showing the algorithm search process."""
    fig, (ax_route, ax_score) = plt.subplots(1, 2, figsize=(12, 6))

    ax_route.set_title("Genetic Algorithm TSP")
    ax_route.set_xlabel("x")
    ax_route.set_ylabel("y")
    ax_route.grid(True, alpha=0.2)

    ax_route.scatter(points[:, 0], points[:, 1], c="crimson", s=50, zorder=3)
    for idx, (x, y) in enumerate(points, start=1):
        ax_route.text(x + 2, y + 2, str(idx), fontsize=9, color="black")

    (route_line,) = ax_route.plot([], [], "-o", color="tab:red", linewidth=2, alpha=0.8)
    best_text = ax_route.text(
        0.02,
        0.95,
        "",
        transform=ax_route.transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    ax_score.set_title("Best Distance")
    ax_score.set_xlabel("Generation")
    ax_score.set_ylabel("Distance")
    ax_score.grid(True, alpha=0.3)
    ax_score.set_xlim(0, max(1, len(scores) - 1))
    ymin = max(0.0, min(scores) * 0.95)
    ymax = max(scores) * 1.05
    if np.isclose(ymin, ymax):
        ymax = ymin + 1.0
    ax_score.set_ylim(ymin, ymax)
    (score_line,) = ax_score.plot([], [], color="tab:blue", linewidth=2)

    def init():
        return update(0)

    def update(frame: int):
        route = routes[frame]
        path = closed_route(points, route)
        route_line.set_data(path[:, 0], path[:, 1])

        score_line.set_data(np.arange(frame + 1), scores[: frame + 1])
        best_text.set_text(f"Generation {frame + 1}\nDistance: {scores[frame]:.2f}")
        return route_line, score_line, best_text

    anim = FuncAnimation(
        fig,
        update,
        frames=len(routes),
        init_func=init,
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    plt.tight_layout()
    return anim


def main() -> None:
    global ANIM
    cities = generate_cities(CITY_COUNT)
    routes, scores = run_algo(cities)
    ANIM = animate_history(cities, routes, scores)  # keep a reference
    plt.show()


if __name__ == "__main__":
    main()
