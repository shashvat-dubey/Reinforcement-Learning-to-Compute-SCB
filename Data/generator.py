"""
generator.py

Random graph generator for SCB-RL.

Produces random connected graphs together with
random source-destination session pairs.

Author: SCB-RL
"""

import random


# ==========================================================
# Node Generation
# ==========================================================

def generate_nodes(
    min_nodes=5,
    max_nodes=20
):
    """
    Generates graph nodes.

    Returns
    -------
    list[str]
    """

    num_nodes = random.randint(
        min_nodes,
        max_nodes
    )

    return [

        f"v{i+1}"

        for i in range(num_nodes)

    ]


# ==========================================================
# Edge Generation
# ==========================================================

def generate_edges(nodes):
    """
    Generates a connected undirected graph.

    Returns
    -------
    list[tuple]
    """

    edges = set()

    # --------------------------------------------------
    # Random Spanning Tree
    # --------------------------------------------------

    connected = [nodes[0]]

    remaining = nodes[1:]

    while remaining:

        u = random.choice(connected)

        v = random.choice(remaining)

        edges.add(

            tuple(sorted((u, v)))

        )

        connected.append(v)

        remaining.remove(v)

    # --------------------------------------------------
    # Random Extra Edges
    # --------------------------------------------------

    num_nodes = len(nodes)

    tree_edges = num_nodes - 1

    max_edges = num_nodes * (num_nodes - 1) // 2

    max_extra_edges = max_edges - tree_edges

    min_extra_edges = max(
        1,
        num_nodes // 2
    )

    extra_edges = random.randint(

        min_extra_edges,

        max_extra_edges

    )

    target_edges = tree_edges + extra_edges

    while len(edges) < target_edges:

        u, v = random.sample(nodes, 2)

        edge = tuple(sorted((u, v)))

        edges.add(edge)

    return sorted(edges)


# ==========================================================
# Session Generation
# ==========================================================

def generate_sessions(nodes):
    """
    Generates random unicast sessions.

    Returns
    -------
    list[tuple]
    """

    max_sessions = min(

        len(nodes),

        10

    )

    num_sessions = random.randint(

        2,

        max_sessions

    )

    sessions = []

    used = set()

    while len(sessions) < num_sessions:

        src, dst = random.sample(

            nodes,

            2

        )

        if (src, dst) in used:

            continue

        used.add(

            (src, dst)

        )

        sessions.append(

            (src, dst)

        )

    return sessions


# ==========================================================
# Graph Generation
# ==========================================================

def generate_graph(seed=None):
    """
    Generates one random graph.

    Returns
    -------
    nodes
    edges
    sessions
    """

    if seed is not None:

        random.seed(seed)

    nodes = generate_nodes()

    edges = generate_edges(nodes)

    sessions = generate_sessions(nodes)

    return (

        nodes,

        edges,

        sessions

    )


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    nodes, edges, sessions = generate_graph()

    print()

    print("Nodes")

    print(nodes)

    print()

    print("Sessions")

    print(sessions)

    print()

    print("Edges")

    print(edges)

    print()

    print(

        len(nodes),

        len(edges),

        len(sessions)

    )