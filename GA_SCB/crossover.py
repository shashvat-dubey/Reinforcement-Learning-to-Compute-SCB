import random

# -------------------------------------------------------
# ADAPTIVE FITNESS-BIASED CONSENSUS SET CROSSOVER
# -------------------------------------------------------

def crossover(parent1, parent2, fitness_key):
    """
    Adaptive Fitness-Biased Consensus Set Crossover

    1. Preserve all common cut edges.
    2. Maintain average chromosome size.
    3. Bias inheritance towards fitter parent.
    4. Automatically becomes more conservative
       as parents become similar.
    """

    # Ensure parent1 is always the fitter parent
    if parent2[fitness_key] > parent1[fitness_key]:
        parent1, parent2 = parent2, parent1

    p1 = parent1["chrom"]
    p2 = parent2["chrom"]

    # --------------------------------------------------
    # Common and Unique edges
    # --------------------------------------------------

    common = p1 & p2

    unique1 = list(p1 - common)
    unique2 = list(p2 - common)

    random.shuffle(unique1)
    random.shuffle(unique2)

    # --------------------------------------------------
    # Target chromosome size
    # --------------------------------------------------

    target_size = round((len(p1) + len(p2)) / 2)

    # If common edges already exceed target,
    # randomly trim them.
    if len(common) >= target_size:
        return set(random.sample(list(common), target_size))

    child = set(common)

    # --------------------------------------------------
    # Adaptive inheritance bias
    # --------------------------------------------------

    union = p1 | p2

    if len(union) == 0:
        return child

    agreement = len(common) / len(union)

    # Bias varies from 0.70 → 0.95
    bias = min(0.95, 0.70 + 0.25 * agreement)

    # --------------------------------------------------
    # Fill remaining positions
    # --------------------------------------------------

    while len(child) < target_size:

        choose_parent1 = (
            random.random() < bias and len(unique1) > 0
        ) or len(unique2) == 0

        if choose_parent1 and unique1:

            child.add(unique1.pop())

        elif unique2:

            child.add(unique2.pop())

        else:
            break

    return child