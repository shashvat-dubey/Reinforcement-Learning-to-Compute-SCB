from .graph import (
    evaluate,
    structural_diversity,
    representation_diversity,
    random_chromosome,
)

from .selection import select_parent
from .crossover import crossover
from .mutation import mutate
from .local_search import local_search

def run_ga(problem, fitness_key, generations=50, pop_size=10, verbose=True):

    elite_k = max(2, pop_size // 20)
    # -------------------------------------------------------
    # INITIAL POPULATION
    # -------------------------------------------------------

    population = []

    for i in range(pop_size):

        chrom = random_chromosome(problem, i, pop_size)
        population.append(evaluate(problem, chrom))

    # Generation where local search begins
    local_search_start = int(0.7 * generations)

    # -------------------------------------------------------
    # EVOLUTION LOOP
    # -------------------------------------------------------

    for gen in range(generations):

        population.sort(
            key=lambda x: x[fitness_key],
            reverse=True
        )

        best = population[0]
        worst = population[-1]

        avg_fit = (
            sum(ind[fitness_key] for ind in population)
            / len(population)
        )

        struct_div = structural_diversity(population, fitness_key)
        repr_div = representation_diversity(population)

        if verbose:
            print("\n-------------------------------------------")
            print(f"Generation {gen}")

            print(f"Structural Diversity     : {struct_div}")
            print(f"Representation Diversity : {repr_div}")

            print("\nBEST")
            print(
                f"Fitness : {best[fitness_key]:.4f}"
                f" | Cut : {best['cut']}"
                f" | Sep : {best['sep']}"
            )

            print("\nBest Chromosome:")
            print(sorted(best["chrom"]))

            print("\nWORST")
            print(
                f"Fitness : {worst[fitness_key]:.4f}"
                f" | Cut : {worst['cut']}"
                f" | Sep : {worst['sep']}"
            )

            print(f"\nAverage Fitness : {avg_fit:.4f}")

            if fitness_key == "fitness1":
                print(
                    "All Sessions Separated :",
                    best["sep"] == len(problem.sessions)
                )

            print("-------------------------------------------")

        # ---------------------------------------------------
        # CREATE NEXT GENERATION
        # ---------------------------------------------------

        new_population = population[:elite_k]

        while len(new_population) < pop_size:

            parent1 = select_parent(population, fitness_key)
            parent2 = select_parent(population, fitness_key)

            child = crossover(parent1, parent2, fitness_key)

            child = mutate(problem, child)

            child_eval = evaluate(problem, child)

            # ---------------------------------------------------
            # Apply Local Search only in the final 30%
            # and only on promising offspring
            # ---------------------------------------------------

            if (
                gen >= local_search_start
                and
                child_eval[fitness_key] >= min(
                    parent1[fitness_key],
                    parent2[fitness_key]
                )
            ):

                child = local_search(problem, child, fitness_key)
                child_eval = evaluate(problem, child)

            new_population.append(child_eval)

        population = new_population

    # -------------------------------------------------------
    # FINAL SOLUTION
    # -------------------------------------------------------

    population.sort(
        key=lambda x: x[fitness_key],
        reverse=True
    )

    best = population[0]
    return best