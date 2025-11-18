"""
Spousti experimenty pro TLBO a ostatni optimalizacni algoritmy.

Parametry:
- D = 30 (dimenze)
- NP = 30 (velikost populace)
- Max_OFE = 3000 (maximalni pocet vyhodnoceni objektove funkce)
- 30 experimentu pro kazdou kombinaci algoritmu a funkce

Vyuziva jiz existujici implementace z cv4.py (DE), cv5.py (PSO),
cv6.py (SOMA), cv8.py (FA) a cv7.py (TLBO).
"""

import numpy as np
from typing import Callable, Tuple
import pandas as pd
from functions import functions
import sys
from tqdm import tqdm
import io
import contextlib

# import existujicich algoritmu
from cv4 import run_differential_evolution
from cv5 import run_pso
from cv6 import run_soma
from cv8 import run_firefly_algorithm
from cv9 import run_tlbo

# parametry podle zadani
DIMENSION = 30
POPULATION_SIZE = 30
MAX_OFE = 3000  # maximalni pocet vyhodnoceni objektove funkce
NUM_EXPERIMENTS = 30


def de_wrapper(
    fitness_func: Callable,
    dimension: int,
    lower_bound: float,
    upper_bound: float,
    pop_size: int,
    max_ofe: int,
) -> Tuple[float, np.ndarray]:
    """Wrapper pro DE algoritmus."""
    max_generations = max_ofe // pop_size
    history_best, history_scores, _ = run_differential_evolution(
        fitness_func=fitness_func,
        dimension=dimension,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        np_size=pop_size,
        f_mutation=0.5,
        cr_crossover=0.5,
        max_generations=max_generations,
    )
    return history_scores[-1], history_best[-1]


def pso_wrapper(
    fitness_func: Callable,
    dimension: int,
    lower_bound: float,
    upper_bound: float,
    pop_size: int,
    max_ofe: int,
) -> Tuple[float, np.ndarray]:
    """Wrapper pro PSO algoritmus."""
    max_iterations = max_ofe // pop_size
    history_best, history_scores, _ = run_pso(
        fitness_func=fitness_func,
        dimension=dimension,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        pop_size=pop_size,
        c1=2.0,
        c2=2.0,
        max_iterations=max_iterations,
        w_start=0.9,
        w_end=0.4,
    )
    return history_scores[-1], history_best[-1]


def soma_wrapper(
    fitness_func: Callable,
    dimension: int,
    lower_bound: float,
    upper_bound: float,
    pop_size: int,
    max_ofe: int,
) -> Tuple[float, np.ndarray]:
    """Wrapper pro SOMA algoritmus."""
    # soma ma specifickou strukturu - pocet migraci
    # kazda migrace vyhodnoti priblizne pop_size * (path_length/step) reseni
    path_length = 3.0
    step = 0.11
    evaluations_per_migration = int(pop_size * (path_length / step))
    max_migrations = max(1, max_ofe // evaluations_per_migration)

    history_best, history_scores, _, _ = run_soma(
        fitness_func=fitness_func,
        dimension=dimension,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        pop_size=pop_size,
        prt=0.4,
        path_length=path_length,
        step=step,
        max_migrations=max_migrations,
    )
    return history_scores[-1], history_best[-1]


def fa_wrapper(
    fitness_func: Callable,
    dimension: int,
    lower_bound: float,
    upper_bound: float,
    pop_size: int,
    max_ofe: int,
) -> Tuple[float, np.ndarray]:
    """Wrapper pro FA algoritmus."""
    # fa ma O(n^2) evaluaci na generaci
    evaluations_per_gen = pop_size * pop_size
    max_generations = max(1, max_ofe // evaluations_per_gen)

    history_best, history_scores, _ = run_firefly_algorithm(
        fitness_func=fitness_func,
        dimension=dimension,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        pop_size=pop_size,
        beta_0=1.0,
        gamma=1.0,
        alpha=0.3,
        max_generations=max_generations,
    )
    return history_scores[-1], history_best[-1]


def tlbo_wrapper(
    fitness_func: Callable,
    dimension: int,
    lower_bound: float,
    upper_bound: float,
    pop_size: int,
    max_ofe: int,
) -> Tuple[float, np.ndarray]:
    """Wrapper pro TLBO algoritmus."""
    # tlbo ma 2*pop_size evaluaci na generaci (teacher + learner phase)
    max_generations = max(1, max_ofe // (2 * pop_size))

    history_best, history_scores, _ = run_tlbo(
        fitness_func=fitness_func,
        dimension=dimension,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        pop_size=pop_size,
        max_generations=max_generations,
    )
    return history_scores[-1], history_best[-1]


# ================================================================
# hlavni experiment
# ================================================================
def run_experiments():
    """spusti vsechny experimenty a ulozi vysledky do excel souboru"""

    algorithms = {
        "DE": de_wrapper,
        "PSO": pso_wrapper,
        "SOMA": soma_wrapper,
        "FA": fa_wrapper,
        "TLBO": tlbo_wrapper,
    }

    # slovnik pro ulozeni vysledku: {func_name: DataFrame}
    results_by_function = {}

    print(f"Parametry experimentu:")
    print(f"  Dimenze: {DIMENSION}")
    print(f"  Velikost populace: {POPULATION_SIZE}")
    print(f"  Max OFE: {MAX_OFE}")
    print(f"  Pocet experimentu: {NUM_EXPERIMENTS}")
    print(f"  Algoritmy: {list(algorithms.keys())}")
    print(f"  Funkce: {[f[2] for f in functions]}")
    print()

    # pro kazdou testovaci funkci
    for func, bounds, func_name in functions:
        print(f"\n{'='*60}")
        print(f"Funkce: {func_name}")
        print(f"Hranice: [{bounds[0]}, {bounds[1]}]")
        print(f"{'='*60}")

        lower_bound, upper_bound = bounds

        # slovnik pro ulozeni vysledku teto funkce
        function_results = {algo_name: [] for algo_name in algorithms.keys()}

        # pro kazdy algoritmus
        for algo_name, algo_func in algorithms.items():
            print(f"\n  {algo_name}:")

            # spust NUM_EXPERIMENTS experimentu
            for exp in tqdm(
                range(NUM_EXPERIMENTS), desc=f"    Experimenty", file=sys.stdout
            ):
                np.random.seed(exp)

                best_fitness, _ = algo_func(
                    fitness_func=func,
                    dimension=DIMENSION,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    pop_size=POPULATION_SIZE,
                    max_ofe=MAX_OFE,
                )

                function_results[algo_name].append(best_fitness)

            # vypocet statistik
            mean = np.mean(function_results[algo_name])
            std = np.std(function_results[algo_name])
            print(f"    Mean: {mean:.6f}, Std: {std:.6f}")

        # vytvoreni dataframe pro tuto funkci
        df_data = {}
        for algo_name in algorithms.keys():
            df_data[algo_name] = function_results[algo_name]

        # pridani radku s prumerem a std
        df = pd.DataFrame(df_data)
        df.index = [f"Experiment {i+1}" for i in range(NUM_EXPERIMENTS)]

        # pridani statistickych radku
        mean_row = df.mean().to_frame().T
        mean_row.index = ["Mean"]
        std_row = df.std().to_frame().T
        std_row.index = ["Std. dev"]

        df_final = pd.concat([df, mean_row, std_row])

        results_by_function[func_name] = df_final

    # ulozeni do excel souboru
    output_file = "tlbo_results.xlsx"
    print(f"\n{'='*60}")
    print(f"Ukladani vysledku do: {output_file}")
    print(f"{'='*60}")

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for func_name, df in results_by_function.items():
            # zkraceni nazvu pro sheet (max 31 znaku)
            sheet_name = func_name[:31]
            df.to_excel(writer, sheet_name=sheet_name)
            print(f"  Ulozeno: {sheet_name}")

    print(f"\nHotovo! Vysledky ulozeny do: {output_file}")


if __name__ == "__main__":
    run_experiments()
