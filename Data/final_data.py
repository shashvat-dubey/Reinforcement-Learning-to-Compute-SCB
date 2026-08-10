"""
Runs the GA on every generated graph and creates
the final labelled dataset.

Author: SCB-RL
"""

import pickle

from GA_SCB.solver import SCBGeneticSolver

INPUT_DATASET = "dataset.pkl"

OUTPUT_DATASET = "labelled_dataset2.pkl"

LOG_FILE = "ga_dataset_log.txt"

GA_RUNS = 10
GA_GENERATIONS = 150
GA_POPULATION = 50

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

log_file = open(
    LOG_FILE,
    "w",
    encoding="utf-8"
)


def log(msg=""):

    print(msg)

    log_file.write(str(msg) + "\n")

    log_file.flush()


# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

with open(
    INPUT_DATASET,
    "rb"
) as f:

    dataset = pickle.load(f)


# ---------------------------------------------------------
# Create Solver
# ---------------------------------------------------------

solver = SCBGeneticSolver(

    generations=GA_GENERATIONS,

    pop_size=GA_POPULATION

)


# ---------------------------------------------------------
# Label Dataset
# ---------------------------------------------------------

for graph in dataset:

    graph_id = graph["graph_id"]

    log("=" * 70)

    log(f"GRAPH {graph_id}")

    best_result = None
    best_solution = None
    best_scb = float("inf")
    best_run = -1

    # -----------------------------------------------------
    # Multiple Independent GA Runs
    # -----------------------------------------------------

    for run in range(GA_RUNS):

        result = solver.solve(

            graph["nodes"],

            graph["edges"],

            graph["sessions"],

            verbose=False

        )

        solution = result["solution"]

        # SCB = cut / sep
        scb = (

            solution["cut"] / solution["sep"]

            if solution["sep"] > 0

            else float("inf")

        )

        if scb < best_scb:

            best_scb = scb

            best_solution = solution

            best_result = result

            best_run = run + 1

    # -----------------------------------------------------
    # Store Best Solution
    # -----------------------------------------------------

    graph["ga_objective"] = best_result["objective"]

    graph["ga_fitness1"] = best_solution["fitness1"]

    graph["ga_fitness2"] = best_solution["fitness2"]

    graph["ga_cut"] = best_solution["cut"]

    graph["ga_sep"] = best_solution["sep"]

    graph["ga_cut_edges"] = best_solution["cut_edges"]

    graph["ga_components"] = best_solution["component"]

    graph["ga_sessions_sep"] = best_solution["sessions_sep"]

    graph["ga_scb"] = best_scb

    # -----------------------------------------------------
    # Debug
    # -----------------------------------------------------

    log(f"Nodes: {len(graph['nodes'])}")
    log(f"Edges: {len(graph['edges'])}")
    log(f"Sessions: {len(graph['sessions'])}")

    # -----------------------------------------------------
    # Logging
    # -----------------------------------------------------

    log(

        f"Best Run={best_run}/{GA_RUNS} | "

        f"SCB={graph['ga_scb']:.6f} | "

        f"Cut={graph['ga_cut']} | "

        f"Separated={graph['ga_sep']}"

    )


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

with open(
    OUTPUT_DATASET,
    "wb"
) as f:

    pickle.dump(dataset, f)


log()

log("=" * 70)

log("DONE")

log("=" * 70)

log(f"Saved to : {OUTPUT_DATASET}")

log_file.close()