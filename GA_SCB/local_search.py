import random
from .graph import evaluate


def local_search(problem,chrom, fitness_key):

    current = set(chrom)
    current_eval = evaluate(problem, current)

    budget = min(20, max(5, problem.E//20))

    improved = True

    while improved and budget > 0:

        improved = False

        # -------------------------
        # ADDITION
        # -------------------------

        remaining = list(set(problem.edges) - current)
        random.shuffle(remaining)

        for edge in remaining[:budget]:

            candidate = set(current)
            candidate.add(edge)

            cand = evaluate(problem, candidate)

            if cand[fitness_key] > current_eval[fitness_key]:

                current = candidate
                current_eval = cand
                improved = True
                budget -= 1
                break

        # -------------------------
        # REMOVAL
        # -------------------------

        if budget <= 0:
            break

        cuts = list(current)
        random.shuffle(cuts)

        for edge in cuts[:budget]:

            candidate = set(current)
            candidate.remove(edge)

            cand = evaluate(problem, candidate)

            if cand[fitness_key] > current_eval[fitness_key]:

                current = candidate
                current_eval = cand
                improved = True
                budget -= 1
                break

        # -------------------------
        # SWAP
        # -------------------------

        if budget <= 0:
            break

        cuts = list(current)
        remaining = list(set(problem.edges) - current)

        random.shuffle(cuts)
        random.shuffle(remaining)

        for remove_edge in cuts[:budget]:

            for add_edge in remaining[:budget]:

                candidate = set(current)

                candidate.remove(remove_edge)
                candidate.add(add_edge)

                cand = evaluate(problem, candidate)

                if cand[fitness_key] > current_eval[fitness_key]:

                    current = candidate
                    current_eval = cand
                    improved = True
                    budget -= 1
                    break

            if improved:
                break

    return current