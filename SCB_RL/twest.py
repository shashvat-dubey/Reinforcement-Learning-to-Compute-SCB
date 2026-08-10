# test_builder.py

from GA_SCB.graph import SCBProblem
from SCB_RL.environment import SCBEnvironment
from SCB_RL.gnn import GraphBuilder
from SCB_RL.gnn import SCBGraphEncoder
from SCB_RL.policy import HierarchicalSCBPolicy

nodes = [
    "v1","v2","v3",
    "v4","v5","v6","v7"
]

sessions = [
    ("v1","v4"),
    ("v2","v5"),
    ("v3","v6")
]

edges = [
    ("v4","v3"),
    ("v5","v7"),
    ("v3","v7"),
    ("v6","v7"),
    ("v6","v1"),
    ("v7","v1"),
    ("v7","v4"),
    ("v7","v2"),
    ("v4","v2"),
    ("v1","v2")
]

problem = SCBProblem(
    nodes,
    edges,
    sessions
)

env = SCBEnvironment(problem)

state = env.reset()

builder = GraphBuilder()

data = builder.build(state)

# print(data)

# print(data.x.shape)
# print(data.edge_index.shape)
# print(data.edge_attr.shape)

# encoder = SCBGraphEncoder()

# node_embeddings = encoder(state)

# print(node_embeddings)

# print(node_embeddings.shape)

encoder = SCBGraphEncoder()

encoding = encoder(state)

policy = HierarchicalSCBPolicy()

output = policy(encoding)


for _ in range(10):

    action, log_prob, entropy = policy.sample_action(encoding)

    print(action)