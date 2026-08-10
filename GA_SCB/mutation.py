import random

# -------------------------------------------------------
# ADAPTIVE SET MUTATION
# -------------------------------------------------------

def mutate(problem, chrom):
    """
    Mutation Operators

    70% : Swap one cut edge with one non-cut edge.
    15% : Add one new cut edge.
    15% : Remove one cut edge.
    """

    child = set(chrom)

    r = random.random()

    # -------------------------------------
    # SWAP MUTATION
    # -------------------------------------

    if r < 0.70:

        if len(child) > 0:

            remove_edge = random.choice(list(child))
            child.remove(remove_edge)

        available = list(set(problem.edges) - child)

        if available:
            add_edge = random.choice(available)
            child.add(add_edge)

    # -------------------------------------
    # ADD MUTATION
    # -------------------------------------

    elif r < 0.85:

        available = list(set(problem.edges) - child)

        if available:

            add_edge = random.choice(available)
            child.add(add_edge)

    # -------------------------------------
    # REMOVE MUTATION
    # -------------------------------------

    else:

        if len(child) > 1:

            remove_edge = random.choice(list(child))
            child.remove(remove_edge)

    return child