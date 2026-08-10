import random
from collections import deque
from dataclasses import dataclass


# ==========================================================
# SCB PROBLEM
# ==========================================================

@dataclass
class SCBProblem:
    """
    Stores one graph instance for the Sparsest Cut Bound problem.
    """

    nodes: list
    edges: list
    sessions: list

    def __post_init__(self):

        # Basic information
        self.N = len(self.nodes)
        self.E = len(self.edges)

        # Node mappings
        self.node_to_idx = {
            node: idx
            for idx, node in enumerate(self.nodes)
        }

        self.idx_to_node = {
            idx: node
            for node, idx in self.node_to_idx.items()
        }

        # Sessions converted to integer indices
        self.sessions_idx = [

            (
                self.node_to_idx[s],
                self.node_to_idx[t]
            )

            for s, t in self.sessions
        ]

        self.edge_to_idx = {
            edge: idx
            for idx, edge in enumerate(self.edges)
        }

        self.idx_to_edge = {
            idx: edge
            for idx, edge in enumerate(self.edges)
        }


# ==========================================================
# RANDOM CHROMOSOME
# ==========================================================

def random_chromosome(problem, pop_index=None, pop_size=None):
    """
    Stratified initialization.

    Returns a chromosome represented as a set of cut edges.
    """

    if pop_index is None or pop_size is None:

        k = random.randint(
            max(1, problem.E // 10),
            max(2, problem.E // 2)
        )

        return set(random.sample(problem.edges, k))

    bucket = (5 * pop_index) // pop_size

    if bucket == 0:

        low, high = 1, max(2, problem.E // 5)

    elif bucket == 1:

        low = max(2, problem.E // 5)
        high = max(3, 2 * problem.E // 5)

    elif bucket == 2:

        low = max(3, 2 * problem.E // 5)
        high = max(4, 3 * problem.E // 5)

    elif bucket == 3:

        low = max(4, 3 * problem.E // 5)
        high = max(5, 4 * problem.E // 5)

    else:

        low = max(5, 4 * problem.E // 5)
        high = problem.E

    k = random.randint(low, high)

    return set(random.sample(problem.edges, k))


# ==========================================================
# REMAINING GRAPH
# ==========================================================

def remaining_graph(problem, chrom):
    """
    Builds the graph after removing the cut edges.
    """

    adj = [[] for _ in range(problem.N)]

    for u_name, v_name in problem.edges:

        if (u_name, v_name) in chrom:
            continue

        u = problem.node_to_idx[u_name]
        v = problem.node_to_idx[v_name]

        adj[u].append(v)
        adj[v].append(u)

    return adj


# ==========================================================
# CONNECTED COMPONENTS
# ==========================================================

def connected_components(adj):
    """
    Computes connected components using BFS.
    """

    component = [-1] * len(adj)

    comp = 0

    for start in range(len(adj)):

        if component[start] != -1:
            continue

        q = deque([start])

        component[start] = comp

        while q:

            u = q.popleft()

            for v in adj[u]:

                if component[v] == -1:

                    component[v] = comp
                    q.append(v)

        comp += 1

    return component


# ==========================================================
# EVALUATE
# ==========================================================

def evaluate(problem, chrom):
    """
    Evaluates one chromosome.
    """

    cut = len(chrom)

    adj = remaining_graph(problem, chrom)

    component = connected_components(adj)

    sep = 0

    sessions_sep = []

    for s, t in problem.sessions_idx:

        if component[s] != component[t]:

            sep += 1

            sessions_sep.append(

                (
                    problem.idx_to_node[s],
                    problem.idx_to_node[t]
                )

            )

    # Full-cut fitness
    fitness1 = sep * (problem.E + 1) - cut

    # Sparsest-cut fitness
    fitness2 = sep / cut if cut > 0 and sep > 0 else 0

    return {

        "chrom": chrom,

        "cut": cut,

        "sep": sep,

        "fitness1": fitness1,

        "fitness2": fitness2,

        "component": component,

        "cut_edges": sorted(chrom),

        "sessions_sep": sessions_sep

    }


# ==========================================================
# DIVERSITY
# ==========================================================

def structural_diversity(pop, fitness_key):

    return len(

        set(

            ind[fitness_key]

            for ind in pop

        )

    )


def representation_diversity(pop):

    return len(

        set(

            frozenset(ind["chrom"])

            for ind in pop

        )

    )