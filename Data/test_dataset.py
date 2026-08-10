from .generator import generate_graph
from GA_SCB.solver import SCBGeneticSolver

LOG_FILE = "ga_test_log.txt"

log_file = open(LOG_FILE, "w", encoding="utf-8")


def log(msg=""):
    print(msg)
    log_file.write(str(msg) + "\n")
    log_file.flush()


# ---------------------------------------------------
# Generate one graph
# ---------------------------------------------------

nodes, edges, sessions = generate_graph()

log("=" * 70)
log("GENERATED GRAPH")
log("=" * 70)

log(f"Nodes    : {len(nodes)}")
log(f"Edges    : {len(edges)}")
log(f"Sessions : {len(sessions)}")

log("\nNode List")
for n in nodes:
    log(n)

log("\nSession Pairs")
for s in sessions:
    log(s)

log("\nEdges")
for e in edges:
    log(e)

# ---------------------------------------------------
# Create Solver
# ---------------------------------------------------

solver = SCBGeneticSolver(
    generations=50,
    pop_size=100
)

# ---------------------------------------------------
# Solve
# ---------------------------------------------------

log("\n")
log("=" * 70)
log("RUNNING GA")
log("=" * 70)

result = solver.solve(
    nodes,
    edges,
    sessions,
    verbose=False
)

# ---------------------------------------------------
# Results
# ---------------------------------------------------

solution = result["solution"]

log("\n")
log("=" * 70)
log("RESULT")
log("=" * 70)

log(f"Fitness1          : {solution['fitness1']}")
log(f"Fitness2 (SCB)    : {solution['fitness2']}")
log(f"Cut Size          : {solution['cut']}")
log(f"Separated Sessions: {solution['sep']}")
log(f"SCB value        : {solution['sep'] / solution['cut'] if solution['cut'] > 0 else float('inf')}")

log("\nCut Edges")
for e in solution["cut_edges"]:
    log(e)

log("\nSeparated Sessions")
for s in solution["sessions_sep"]:
    log(s)

log("\nComponents")
log(solution["component"])

log("\nDone.")

log_file.close()