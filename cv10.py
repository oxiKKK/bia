"""
NSGA-II (Non-dominated Sorting Genetic Algorithm II)

Algoritmus pro multikriterialni optimalizaci.
Hleda mnozinu Pareto-optimalnich reseni (Pareto frontu).
Pouziva:
1. Fast Non-dominated Sorting - razeni do front podle dominance
2. Crowding Distance - udrzovani diverzity v populaci
3. Tournament Selection - vyber rodicu
4. SBX Crossover a Polynomial Mutation - geneticke operatory

Problem (Cone Problem):
    Minimalizovat plochu plaste (S) a celkovou plochu (T) kuzelu,
    pri zachovani minimalniho objemu (V).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple
import random
from dataclasses import dataclass, field

# Parametry problemu
R_MIN, R_MAX = 0.1, 10.0  # polomer podstavy
H_MIN, H_MAX = 0.1, 20.0  # vyska kuzelu
MIN_VOLUME = 200.0  # minimalni pozadovany objem

# Parametry NSGA-II
POP_SIZE = 100  # Velikost populace
MAX_GEN = 50  # Pocet generaci
CROSSOVER_RATE = 0.9  # Pravdepodobnost krizeni
MUTATION_RATE = 0.1  # Pravdepodobnost mutace
ETA_C = 20  # Index distribuce pro krizeni (SBX)
ETA_M = 20  # Index distribuce pro mutaci

ANIMATION_INTERVAL = 100  # Rychlost animace (ms)


@dataclass
class Individual:
    x: np.ndarray = field(default_factory=lambda: np.zeros(2))  # [r, h]
    objectives: np.ndarray = field(default_factory=lambda: np.zeros(2))  # [S, T]
    rank: int = 0
    crowding_distance: float = 0.0
    violation: float = 0.0
    domination_count: int = 0
    dominated_individuals: List["Individual"] = field(default_factory=list)


def evaluate(ind: Individual):
    """Vypocita ucelove funkce a poruseni omezeni."""
    r, h = ind.x
    s_slant = np.sqrt(r**2 + h**2)
    V = (np.pi / 3) * r**2 * h

    ind.objectives[:] = [np.pi * r * s_slant, np.pi * r * (r + s_slant)]
    ind.violation = max(0.0, MIN_VOLUME - V)


def dominates(ind1: Individual, ind2: Individual) -> bool:
    """
    Vraci True, pokud ind1 dominuje ind2 (Constrained Domination).
    Resi i omezujici podminky.
    """
    # 1. Pokud je jeden pripustny a druhy ne -> pripustny vyhrava
    if ind1.violation == 0 and ind2.violation > 0:
        return True
    if ind1.violation > 0 and ind2.violation == 0:
        return False

    # 2. Pokud jsou oba nepripustne -> lepsi je ten s mensim porusenim
    if ind1.violation > 0 and ind2.violation > 0:
        return ind1.violation < ind2.violation

    # 3. Pokud jsou oba pripustne -> klasicka Pareto dominance
    # ind1 dominuje ind2, pokud je ve vsech objektivech lepsi nebo stejny (<=)
    # a alespon v jednom ostre lepsi (<)
    better_or_equal = np.all(ind1.objectives <= ind2.objectives)
    strictly_better = np.any(ind1.objectives < ind2.objectives)

    return better_or_equal and strictly_better


def fast_non_dominated_sort(population: List[Individual]) -> List[List[Individual]]:
    """
    Rozdeli populaci do Pareto front (F1, F2, ...).
    F1 obsahuje nedominovana reseni.
    """
    fronts = [[]]

    for p in population:
        p.domination_count = 0
        p.dominated_individuals = []

        for q in population:
            if dominates(p, q):
                p.dominated_individuals.append(q)  # p dominuje q
            elif dominates(q, p):
                p.domination_count += 1  # q dominuje p

        if p.domination_count == 0:
            p.rank = 0
            fronts[0].append(p)

    # Postupne vytvareni dalsich front
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in p.dominated_individuals:
                q.domination_count -= 1
                if q.domination_count == 0:
                    q.rank = i + 1
                    next_front.append(q)
        i += 1
        if len(next_front) > 0:
            fronts.append(next_front)
        else:
            break  # Zadna dalsi fronta

    return fronts


def crowding_distance_assignment(front: List[Individual]):
    """
    Vypocita crowding distance pro jedince ve fronte.
    Vetsi vzdalenost znamena lepsi diverzitu (jedinec je v ridsi oblasti).
    """
    if len(front) == 0:
        return

    l = len(front)
    for ind in front:
        ind.crowding_distance = 0.0

    num_objectives = len(front[0].objectives)

    for m in range(num_objectives):
        # Seradit podle m-teho objektivu
        front.sort(key=lambda x: x.objectives[m])

        # Krajni body maji nekonecnou vzdalenost (chceme je zachovat)
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")

        obj_min = front[0].objectives[m]
        obj_max = front[-1].objectives[m]

        if obj_max == obj_min:
            continue

        # Vypocet vzdalenosti pro vnitrni body
        for i in range(1, l - 1):
            front[i].crowding_distance += (
                front[i + 1].objectives[m] - front[i - 1].objectives[m]
            ) / (obj_max - obj_min)


def tournament_selection(population: List[Individual]) -> Individual:
    """
    Vybere rodice pomoci binarniho turnaje.
    Pouziva Crowded Comparison Operator (Rank, Crowding Distance).
    """
    i1 = random.randint(0, len(population) - 1)
    i2 = random.randint(0, len(population) - 1)

    ind1 = population[i1]
    ind2 = population[i2]

    # 1. Lepsi Rank (nizsi je lepsi)
    if ind1.rank < ind2.rank:
        return ind1
    elif ind1.rank > ind2.rank:
        return ind2
    else:
        # 2. Stejny Rank -> Vetsi Crowding Distance (lepsi diverzita)
        if ind1.crowding_distance > ind2.crowding_distance:
            return ind1
        else:
            return ind2


def sbx_crossover(p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
    """
    Simulated Binary Crossover (SBX).
    Simuluje chovani jednobodoveho krizeni z binarniho GA, ale pro realna cisla.
    Vytvari potomky, kteri maji podobne rozlozeni jako rodice.
    """
    c1 = Individual()
    c2 = Individual()

    # simple real-valued crossover: convex combination of parents
    if random.random() <= CROSSOVER_RATE:
        alpha = np.random.uniform(0.0, 1.0, size=2)
        c1.x = alpha * p1.x + (1 - alpha) * p2.x
        c2.x = (1 - alpha) * p1.x + alpha * p2.x
    else:
        c1.x = p1.x.copy()
        c2.x = p2.x.copy()

    # clip to bounds
    c1.x[0] = np.clip(c1.x[0], R_MIN, R_MAX)
    c1.x[1] = np.clip(c1.x[1], H_MIN, H_MAX)
    c2.x[0] = np.clip(c2.x[0], R_MIN, R_MAX)
    c2.x[1] = np.clip(c2.x[1], H_MIN, H_MAX)

    return c1, c2


def polynomial_mutation(ind: Individual):
    """
    Polynomial Mutation.
    Provadi malou zmenu hodnoty promenne s urcitou pravdepodobnosti.
    Pouziva polynomiální rozdeleni pravdepodobnosti pro urceni velikosti zmeny.
    """
    # simple mutation: small random shift in each variable
    for i in range(2):
        if random.random() <= MUTATION_RATE:
            if i == 0:
                lb, ub = R_MIN, R_MAX
            else:
                lb, ub = H_MIN, H_MAX
            span = ub - lb
            shift = np.random.uniform(-0.1, 0.1) * span
            ind.x[i] = np.clip(ind.x[i] + shift, lb, ub)


def run_nsga2():
    # Inicializace populace
    population = []
    for _ in range(POP_SIZE):
        ind = Individual()
        ind.x[0] = np.random.uniform(R_MIN, R_MAX)
        ind.x[1] = np.random.uniform(H_MIN, H_MAX)
        evaluate(ind)  # vypoctu cilove funkce a violation
        population.append(ind)

    history_fronts = []

    for gen in range(MAX_GEN):
        # iteruju, dokud nemam tolik potomku, kolik je velikost populace
        offspring = []
        while len(offspring) < POP_SIZE:
            # 1. vybiram rodice turjanovou selekci
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            # 2. krizim SBX
            c1, c2 = sbx_crossover(p1, p2)
            # 3. mutuju polynomiální mutací
            polynomial_mutation(c1)
            polynomial_mutation(c2)
            # 4. nove vznikle potomky ohodnotim a vypoctu violation
            evaluate(c1)
            evaluate(c2)
            offspring.append(c1)
            offspring.append(c2)

        # 5. spojim rodice a potomky
        combined_pop = population + offspring

        # 6. provedu trideni a ziskam pareto fronty
        #    zjistim kdo dominuje komu
        fronts = fast_non_dominated_sort(combined_pop)

        new_pop = []
        for front in fronts:
            # 7. kazde fronte spocitak crowding distance
            crowding_distance_assignment(front)

            # 8. pokud se cela fronta vejde, pridam ji celou
            if len(new_pop) + len(front) <= POP_SIZE:
                new_pop.extend(front)
            else:
                # 9. pokud se cela fronta nevejde, vybereme ty s nejvetsi crowding distance
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                new_pop.extend(front[: POP_SIZE - len(new_pop)])
                break

        population = new_pop

        # Ulozeni prvni fronty pro vizualizaci
        first_front = [ind for ind in population if ind.rank == 0]
        history_fronts.append(first_front)

        print(f"Generace {gen+1}: Velikost Pareto fronty = {len(first_front)}")

    return history_fronts


def animate_nsga2(history_fronts):
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.set_xlabel("Plocha plaste (S)")
    ax.set_ylabel("Celkova plocha (T)")
    ax.set_title("NSGA-II: Optimalizace kuzele")
    ax.grid(True, alpha=0.3)

    scatter = ax.scatter([], [], c="blue", label="Pareto fronta")
    gen_text = ax.text(0.05, 0.95, "", transform=ax.transAxes)

    # Nastaveni limitu grafu
    all_s = []
    all_t = []
    for front in history_fronts:
        for ind in front:
            all_s.append(ind.objectives[0])
            all_t.append(ind.objectives[1])

    if all_s:
        ax.set_xlim(min(all_s) * 0.9, max(all_s) * 1.1)
        ax.set_ylim(min(all_t) * 0.9, max(all_t) * 1.1)

    def update(frame):
        front = history_fronts[frame]
        s_vals = [ind.objectives[0] for ind in front]
        t_vals = [ind.objectives[1] for ind in front]

        scatter.set_offsets(np.c_[s_vals, t_vals])
        gen_text.set_text(f"Generace: {frame+1}")
        return scatter, gen_text

    anim = FuncAnimation(
        fig, update, frames=len(history_fronts), interval=ANIMATION_INTERVAL, blit=True
    )

    plt.legend()
    plt.tight_layout()
    return anim


def main():
    print("Spoustim NSGA-II pro Cone Problem...")
    history = run_nsga2()

    # Vykresleni finalniho stavu
    final_front = history[-1]
    final_front.sort(key=lambda x: x.objectives[0])

    print("\nFinalni Pareto fronta (vybrane reseni):")
    print(f"{'r':>10} {'h':>10} {'S':>10} {'T':>10} {'V':>10}")
    print("-" * 55)

    for ind in final_front[:10]:  # Vypis prvnich 10
        r, h = ind.x
        V = (np.pi / 3) * r**2 * h
        print(
            f"{r:10.4f} {h:10.4f} {ind.objectives[0]:10.4f} {ind.objectives[1]:10.4f} {V:10.4f}"
        )

    anim = animate_nsga2(history)
    plt.show()


if __name__ == "__main__":
    main()
