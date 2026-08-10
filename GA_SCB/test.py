from GA_SCB.graph import SCBProblem
from GA_SCB.genetic import run_ga
from GA_SCB.solution import print_solution
from GA_SCB.solver import SCBGeneticSolver

# -----------------------
# Your graph
# -----------------------

nodes = [
    "v1", "v2", "v3", "v4",
    "v5", "v6", "v7"
]

sessions = [
    ("v1", "v4"),
    ("v2", "v5"),
    ("v3", "v6")
]

edges = [
    ("v4", "v3"),

    ("v5", "v7"),

    ("v3", "v7"),

    ("v6", "v7"),
    ("v6", "v1"),

    ("v7", "v1"),
    ("v7", "v4"),
    ("v7", "v2"),

    ("v4", "v2"),

    ("v1", "v2")
]

# ----------------------------------------
# Create Solver
# ----------------------------------------

solver = SCBGeneticSolver(
    generations=50,
    pop_size=100
)

# ----------------------------------------
# Solve
# ----------------------------------------

result = solver.solve(
    nodes,
    edges,
    sessions,
    verbose=False      # False if you don't want final printing
)

# ----------------------------------------
# Access Results
# ----------------------------------------

# print("\nBest Objective:", result["objective"])
print("Final SCB:", result["solution"]["fitness2"])