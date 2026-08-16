"""
Pure PyTorch Graph Neural Network Encoder
for the Sparsest Cut Bound RL Agent.

This version intentionally avoids:
    - torch_geometric.nn
    - SAGEConv
    - pyg-lib
    - torch-scatter

The goal is to provide a clean PyTorch-only
baseline for diagnosing the intermittent
native PyG/PyTorch crashes.
"""

import torch
import torch.nn as nn

from .state import SCBState


# ==========================================================
# Graph Builder
# ==========================================================

class GraphBuilder:
    """
    Converts an SCBState into plain PyTorch tensors.

    No PyTorch-Geometric objects are created.
    """

    def __init__(self):
        pass

    def build(self, state: SCBState):

        x = self._node_features(state)

        edge_index = self._edge_index(state)

        edge_attr = self._edge_features(state)

        return {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr
        }

    # ------------------------------------------------------

    def _node_features(self, state: SCBState):

        problem = state.problem

        # --------------------------------------------------
        # Degree
        # --------------------------------------------------

        degree = [0] * problem.N

        for u, v in problem.edges:

            u = problem.node_to_idx[u]
            v = problem.node_to_idx[v]

            degree[u] += 1
            degree[v] += 1

        max_degree = max(degree)

        # Prevent division by zero
        if max_degree == 0:
            max_degree = 1

        # --------------------------------------------------
        # Session endpoint
        # --------------------------------------------------

        endpoint = [0] * problem.N

        for s, t in problem.sessions_idx:

            endpoint[s] = 1
            endpoint[t] = 1

        # --------------------------------------------------
        # Feature matrix
        # --------------------------------------------------

        features = []

        for i in range(problem.N):

            features.append(
                [
                    degree[i] / max_degree,
                    endpoint[i]
                ]
            )

        return torch.tensor(
            features,
            dtype=torch.float32
        )

    # ------------------------------------------------------

    def _edge_features(self, state: SCBState):

        problem = state.problem

        features = []

        cut = state.cut

        for edge in problem.edges:

            is_cut = float(edge in cut)

            # Forward
            features.append([is_cut])

            # Reverse
            features.append([is_cut])

        return torch.tensor(
            features,
            dtype=torch.float32
        )

    # ------------------------------------------------------

    def _edge_index(self, state: SCBState):

        problem = state.problem

        src = []
        dst = []

        for u, v in problem.edges:

            u = problem.node_to_idx[u]
            v = problem.node_to_idx[v]

            # u -> v
            src.append(u)
            dst.append(v)

            # v -> u
            src.append(v)
            dst.append(u)

        return torch.tensor(
            [src, dst],
            dtype=torch.long
        )


# ==========================================================
# Pure PyTorch GraphSAGE Layer
# ==========================================================

class TorchSAGELayer(nn.Module):
    """
    Pure PyTorch implementation of a GraphSAGE-style layer.

    For every node:

        h'_v =
            W_self h_v
            +
            W_neigh mean(h_u)

    where u are the neighbors of v.
    """

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.self_linear = nn.Linear(
            in_channels,
            out_channels
        )

        self.neighbor_linear = nn.Linear(
            in_channels,
            out_channels
        )

    def forward(
        self,
        x,
        edge_index
    ):

        num_nodes = x.size(0)

        src = edge_index[0]
        dst = edge_index[1]

        # --------------------------------------------------
        # Gather source node features
        # --------------------------------------------------

        messages = x[src]

        # --------------------------------------------------
        # Sum messages into destination nodes
        # --------------------------------------------------

        neighbor_sum = torch.zeros(
            num_nodes,
            x.size(1),
            dtype=x.dtype,
            device=x.device
        )

        neighbor_sum.index_add_(
            0,
            dst,
            messages
        )

        # --------------------------------------------------
        # Degree of every destination node
        # --------------------------------------------------

        degree = torch.zeros(
            num_nodes,
            dtype=x.dtype,
            device=x.device
        )

        ones = torch.ones(
            dst.size(0),
            dtype=x.dtype,
            device=x.device
        )

        degree.index_add_(
            0,
            dst,
            ones
        )

        # --------------------------------------------------
        # Mean aggregation
        # --------------------------------------------------

        degree = degree.clamp_min(1.0)

        neighbor_mean = (
            neighbor_sum
            /
            degree.unsqueeze(1)
        )

        # --------------------------------------------------
        # GraphSAGE-style update
        # --------------------------------------------------

        out = (
            self.self_linear(x)
            +
            self.neighbor_linear(neighbor_mean)
        )

        return out


# ==========================================================
# Graph Encoder
# ==========================================================

class SCBGraphEncoder(nn.Module):

    """
    Pure PyTorch GraphSAGE-style encoder.

    Interface intentionally matches the original
    SCBGraphEncoder.
    """

    def __init__(
        self,
        hidden_dim=64,
        num_layers=2
    ):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.builder = GraphBuilder()

        # --------------------------------------------------
        # GraphSAGE Layers
        # --------------------------------------------------

        self.convs = nn.ModuleList()

        in_channels = 2

        for _ in range(num_layers):

            self.convs.append(
                TorchSAGELayer(
                    in_channels,
                    hidden_dim
                )
            )

            in_channels = hidden_dim

    # ======================================================
    # Forward
    # ======================================================

    def forward(
        self,
        state: SCBState
    ):

        # --------------------------------------------------
        # Build graph
        # --------------------------------------------------

        data = self.builder.build(state)

        x = data["x"]

        edge_index = data["edge_index"]

        # --------------------------------------------------
        # Message passing
        # --------------------------------------------------

        for conv in self.convs:

            x = conv(
                x,
                edge_index
            )

            x = torch.relu(x)

        # --------------------------------------------------
        # Edge embeddings
        # --------------------------------------------------

        edge_embeddings = self._build_edge_embeddings(
            x,
            state
        )

        # --------------------------------------------------
        # Graph embedding
        # --------------------------------------------------

        graph_embedding = x.mean(
            dim=0
        )

        return {
            "node_embeddings": x,
            "edge_embeddings": edge_embeddings,
            "graph_embedding": graph_embedding
        }

    # ======================================================
    # Edge Embeddings
    # ======================================================

    def _build_edge_embeddings(
        self,
        node_embeddings,
        state
    ):

        edge_embeddings = []

        cut = state.cut

        for edge in state.problem.edges:

            u, v = edge

            u = state.problem.node_to_idx[u]
            v = state.problem.node_to_idx[v]

            hu = node_embeddings[u]

            hv = node_embeddings[v]

            is_cut = torch.tensor(
                [
                    float(edge in cut)
                ],
                dtype=node_embeddings.dtype,
                device=node_embeddings.device
            )

            edge_embedding = torch.cat(
                [
                    hu,
                    hv,
                    is_cut
                ]
            )

            edge_embeddings.append(
                edge_embedding
            )

        return torch.stack(
            edge_embeddings
        )