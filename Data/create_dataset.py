"""
create_dataset.py

Generates a dataset of random SCB graphs.

Author: SCB-RL
"""

import pickle

from Data.generator import generate_graph


# ==========================================================
# CONFIGURATION
# ==========================================================

NUM_GRAPHS = 1000

OUTPUT_FILE = "dataset.pkl"

SUMMARY_FILE = "dataset_summary.txt"


# ==========================================================
# DATASET GENERATION
# ==========================================================

dataset = []

summary = []

print()
print("=" * 60)
print("Generating Dataset")
print("=" * 60)
print()

for graph_id in range(NUM_GRAPHS):

    nodes, edges, sessions = generate_graph(
        seed=graph_id
    )

    dataset.append(

        {

            "graph_id": graph_id,

            "nodes": nodes,

            "edges": edges,

            "sessions": sessions

        }

    )

    summary.append(

        f"Graph {graph_id:4d} | "
        f"Nodes: {len(nodes):2d} | "
        f"Edges: {len(edges):3d} | "
        f"Sessions: {len(sessions):2d}"

    )

    if (graph_id + 1) % 100 == 0:

        print(
            f"{graph_id + 1}/{NUM_GRAPHS} graphs generated..."
        )


# ==========================================================
# SAVE DATASET
# ==========================================================

with open(
    OUTPUT_FILE,
    "wb"
) as f:

    pickle.dump(dataset, f)


# ==========================================================
# SAVE SUMMARY
# ==========================================================

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write("\n".join(summary))


# ==========================================================
# STATISTICS
# ==========================================================

avg_nodes = sum(
    len(g["nodes"])
    for g in dataset
) / NUM_GRAPHS

avg_edges = sum(
    len(g["edges"])
    for g in dataset
) / NUM_GRAPHS

avg_sessions = sum(
    len(g["sessions"])
    for g in dataset
) / NUM_GRAPHS


print()
print("=" * 60)
print("Dataset Generated Successfully")
print("=" * 60)

print(f"Graphs       : {NUM_GRAPHS}")
print(f"Avg Nodes    : {avg_nodes:.2f}")
print(f"Avg Edges    : {avg_edges:.2f}")
print(f"Avg Sessions : {avg_sessions:.2f}")

print()
print(f"Saved to     : {OUTPUT_FILE}")
print(f"Summary      : {SUMMARY_FILE}")