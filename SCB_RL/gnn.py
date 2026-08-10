"""
gnn.py

Graph Neural Network Encoder for the
Sparsest Cut Bound RL Agent.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.nn import (
    SAGEConv,
    global_mean_pool
)

from .state import SCBState


# ==========================================================
# Graph Builder
# ==========================================================

class GraphBuilder:
    """
    Converts an SCBState into a
    PyTorch-Geometric Data object.

    This class is responsible ONLY for
    feature construction.

    No neural network computations occur here.
    """

    def __init__(self):
        pass


    def build(self, state: SCBState):

        """
        Main entry point.

        SCBState
            ↓
        PyG Data
        """

        x = self._node_features(state)

        edge_index = self._edge_index(state)

        edge_attr = self._edge_features(state)

        return Data(

            x=x,

            edge_index=edge_index,

            edge_attr=edge_attr

        )


    # ------------------------------------------------------

    def _node_features(
        self,
        state: SCBState
    ):
        """
        Build node feature matrix.

        Features
        --------
        Feature 0 : Normalized Degree

        Feature 1 : Session Endpoint Flag

        Returns
        -------
        Tensor
            Shape = [num_nodes, 2]
        """

        problem = state.problem

        # -----------------------------
        # Compute node degrees
        # -----------------------------

        degree = [0] * problem.N

        for u, v in problem.edges:

            u = problem.node_to_idx[u]
            v = problem.node_to_idx[v]

            degree[u] += 1
            degree[v] += 1

        max_degree = max(degree)

        # -----------------------------
        # Session endpoint flag
        # -----------------------------

        endpoint = [0] * problem.N

        for s, t in problem.sessions_idx:

            endpoint[s] = 1
            endpoint[t] = 1

        # -----------------------------
        # Build feature matrix
        # -----------------------------

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

            dtype=torch.float

        )

    # ------------------------------------------------------

    def _edge_features(
        self,
        state: SCBState
    ):
        """
        Build edge feature matrix.

        Feature 0
        ---------
        is_cut

        Returns
        -------
        Tensor
            Shape = [2E, 1]
        """

        problem = state.problem

        features = []

        cut = state.cut

        for edge in problem.edges:

            is_cut = float(edge in cut)

            # Forward edge
            features.append([is_cut])

            # Reverse edge
            features.append([is_cut])

        return torch.tensor(
            features,
            dtype=torch.float
        )
    # ------------------------------------------------------

    def _edge_index(
        self,
        state: SCBState
    ):
        """
        Build PyTorch Geometric edge_index.

        Every undirected edge is stored twice.

        Returns
        -------
        Tensor

            Shape = [2, 2E]
        """

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
# Graph Encoder
# ==========================================================

class SCBGraphEncoder(nn.Module):

    """
    GraphSAGE Encoder.

    Converts a graph into

        Node embeddings

        Edge embeddings

        Graph embedding
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

        # Input:
        # degree + endpoint flag = 2 features

        in_channels = 2

        for _ in range(num_layers):

            self.convs.append(

                SAGEConv(

                    in_channels,

                    hidden_dim

                )

            )

            in_channels = hidden_dim

    def forward(
        self,
        state: SCBState
    ):

        # ---------------------------------------------
        # Build PyG Graph
        # ---------------------------------------------

        data = self.builder.build(state)

        x = data.x

        edge_index = data.edge_index

        # ---------------------------------------------
        # GraphSAGE
        # ---------------------------------------------

        for conv in self.convs:

            x = conv(

                x,

                edge_index

            )

            x = torch.relu(x)

        # ---------------------------------------------
        # Stage 1 Output
        # ---------------------------------------------
        edge_embeddings = self._build_edge_embeddings(
            x,
            state
        )

        graph_embedding = x.mean(dim=0)

        return {

            "node_embeddings": x,

            "edge_embeddings": edge_embeddings,

            "graph_embedding": graph_embedding

        }
    
    def _build_edge_embeddings(
        self,
        node_embeddings,
        state
    ):
        """
        Construct an embedding for every undirected edge.
        """

        edge_embeddings = []

        cut = state.cut

        for edge in state.problem.edges:

            u, v = edge

            u = state.problem.node_to_idx[u]
            v = state.problem.node_to_idx[v]

            hu = node_embeddings[u]
            hv = node_embeddings[v]

            is_cut = torch.tensor(

                [float(edge in cut)],

                device=node_embeddings.device

            )

            edge_embedding = torch.cat(

                [

                    hu,

                    hv,

                    is_cut

                ]

            )

            edge_embeddings.append(edge_embedding)

        return torch.stack(edge_embeddings)