
# -------------------------------------------------------
# PRINT SOLUTION
# -------------------------------------------------------

def print_solution(problem, solution):

    print("\n==========================================")
    print("FINAL SOLUTION")
    print("==========================================")

    print(f"Cut Edges           : {solution['cut']}")
    print(f"Separated Sessions  : {solution['sep']}")
    print(f"FULL Fitness        : {solution['fitness1']:.4f}")
    print(f"SCB Fitness         : {solution['fitness2']:.4f}")

    if solution["sep"] > 0:
        print(f"SCB Ratio (Cut/Sep) : {solution['cut']/solution['sep']:.4f}")

    # ----------------------------------------
    # Components
    # ----------------------------------------

    partitions = {}

    for idx, comp in enumerate(solution["component"]):

        node = problem.idx_to_node[idx]

        if comp not in partitions:
            partitions[comp] = []

        partitions[comp].append(node)

    print("\nPartitions:")

    for comp in sorted(partitions):
        print(f"P{comp+1}: {sorted(partitions[comp])}")

    # ----------------------------------------
    # Chromosome
    # ----------------------------------------

    print("\nChromosome (Cut Set):")

    if len(solution["chrom"]) == 0:
        print("None")

    else:
        for edge in sorted(solution["chrom"]):
            print(edge)

    # ----------------------------------------
    # Cut Edges
    # ----------------------------------------

    print("\nCut Edges:")

    if len(solution["cut_edges"]) == 0:
        print("None")

    else:
        for u, v in sorted(solution["cut_edges"]):
            print(f"{u} - {v}")

    # ----------------------------------------
    # Separated Sessions
    # ----------------------------------------

    print("\nSeparated Sessions:")

    if len(solution["sessions_sep"]) == 0:
        print("None")

    else:
        for s, t in solution["sessions_sep"]:
            print(f"{s} - {t}")

    print("==========================================")