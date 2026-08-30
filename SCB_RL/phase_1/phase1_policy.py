import torch
import torch.nn as nn
from torch.distributions import Categorical


from SCB_RL.actions import Action, ActionType
from SCB_RL.gnn import SCBGraphEncoder

class HierarchicalSCBPolicy(nn.Module):
    """
    Hierarchical Actor Network.

    High-Level Policy
        Chooses the operation:
            ADD
            REMOVE
            SWAP
            STOP

    Low-Level Policies
        Execute the chosen operation by
        selecting the appropriate edge(s).
    """

    def __init__(
        self,
        graph_dim=64,
        edge_dim=129,
        hidden_dim=64
    ):

        super().__init__()

        self.graph_dim = graph_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        # =====================================================
        # Shared Edge Encoder
        # =====================================================

        self.edge_encoder = nn.Sequential(

            nn.Linear(edge_dim, hidden_dim),

            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),

            nn.ReLU()
            )
        # =====================================================
        # Operator Head
        # =====================================================

        self.operator_head = nn.Sequential(

            nn.Linear(graph_dim, hidden_dim),

            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),

            nn.ReLU(),

            nn.Linear(hidden_dim, 4)

        )

        # =====================================================
        # Edge Heads
        # =====================================================

        self.add_head = self._make_edge_head()

        self.remove_head = self._make_edge_head()

        self.swap_remove_head = self._make_edge_head()

        self.swap_add_head = self._make_edge_head()


    def _make_edge_head(self):
        """
        Creates one edge scoring head.

        Input:
            Shared Edge Feature (hidden_dim)

        Output:
            One logit per edge.
        """

        return nn.Sequential(

            nn.Linear(self.hidden_dim, self.hidden_dim),

            nn.ReLU(),

            nn.Linear(self.hidden_dim, 1)

        )

    
    def forward(
        self,
        encoding
    ):
        """
        Forward pass of the policy.

        Parameters
        ----------
        encoding : dict

            Output of SCBGraphEncoder

        Returns
        -------
        dict

            operator_logits

            add_logits

            remove_logits

            swap_remove_logits

            swap_add_logits
        """

        # --------------------------------------------------
        # Encoder Outputs
        # --------------------------------------------------

        graph_embedding = encoding["graph_embedding"]

        edge_embeddings = encoding["edge_embeddings"]

        # --------------------------------------------------
        # Operator
        # --------------------------------------------------

        operator_logits = self.operator_head(

            graph_embedding

        )

        # --------------------------------------------------
        # Shared Edge Representation
        # --------------------------------------------------

        edge_features = self.edge_encoder(

            edge_embeddings

        )

        # --------------------------------------------------
        # Edge Heads
        # --------------------------------------------------

        add_logits = self.add_head(

            edge_features

        ).squeeze(-1)

        remove_logits = self.remove_head(

            edge_features

        ).squeeze(-1)

        swap_remove_logits = self.swap_remove_head(

            edge_features

        ).squeeze(-1)

        swap_add_logits = self.swap_add_head(

            edge_features

        ).squeeze(-1)

        return {

            "operator_logits": operator_logits,

            "add_logits": add_logits,

            "remove_logits": remove_logits,

            "swap_remove_logits": swap_remove_logits,

            "swap_add_logits": swap_add_logits

        }

    
    def sample_action(
        self,
        encoding,
        state
    ):
        """
        Samples one VALID action from the policy.

        Returns
        -------
        Action
        Tensor : total log probability
        Tensor : total entropy
        """

        # ==================================================
        # Forward Pass
        # ==================================================

        output = self.forward(encoding)

        cut = state.cut
        num_edges = output["add_logits"].shape[0]

        # ==================================================
        # Operator Mask
        # ==================================================

        operator_logits = output["operator_logits"].clone()

        MIN_STEPS_BEFORE_STOP = 5

        # Empty cut
        if len(cut) == 0:

            operator_logits[ActionType.REMOVE.value] = -1e9
            operator_logits[ActionType.SWAP.value] = -1e9

        # Full cut
        elif len(cut) == num_edges:

            operator_logits[ActionType.ADD.value] = -1e9
            operator_logits[ActionType.SWAP.value] = -1e9

        # Don't stop immediately
        if state.step < MIN_STEPS_BEFORE_STOP:

            operator_logits[ActionType.STOP.value] = -1e9

        # self._check_logits(
        #         "operator_logits",
        #         operator_logits
        #     )

        operator_dist = Categorical(
            logits=operator_logits
        )

        operator = operator_dist.sample()

        # print(
        #     "SELECTED OPERATOR:",
        #     ActionType(operator.item()),
        #     flush=True
        # )

        log_prob = operator_dist.log_prob(operator)

        entropy = operator_dist.entropy()

        # ==================================================
        # ADD
        # ==================================================

        if operator.item() == ActionType.ADD.value:

            add_logits = output["add_logits"].clone()

            for edge in cut:

                add_logits[edge] = -1e9

            # self._check_logits(
            #     "add_logits",
            #     add_logits
            # )

            # print(
            #     "ADD logits:",
            #     add_logits.detach(),
            #     flush=True
            # )

            edge_dist = Categorical(
                logits=add_logits
            )

            edge = edge_dist.sample()

            log_prob += edge_dist.log_prob(edge)
            entropy += edge_dist.entropy()

            return (
                Action(
                    ActionType.ADD,
                    edge=edge.item()
                ),
                log_prob,
                entropy
            )

        # ==================================================
        # REMOVE
        # ==================================================

        elif operator.item() == ActionType.REMOVE.value:

            remove_logits = torch.full_like(
                output["remove_logits"],
                -1e9
            )

            for edge in cut:

                remove_logits[edge] = output["remove_logits"][edge]

            # self._check_logits(
            #     "remove_logits",
            #     remove_logits
            # )

            # print(
            #     "REMOVE logits:",
            #     remove_logits.detach(),
            #     flush=True
            # )

            edge_dist = Categorical(
                logits=remove_logits
            )

            edge = edge_dist.sample()

            log_prob += edge_dist.log_prob(edge)
            entropy += edge_dist.entropy()

            return (
                Action(
                    ActionType.REMOVE,
                    edge=edge.item()
                ),
                log_prob,
                entropy
            )

        # ==================================================
        # SWAP
        # ==================================================

        elif operator.item() == ActionType.SWAP.value:

            # ---------- Remove ----------

            remove_logits = torch.full_like(
                output["swap_remove_logits"],
                -1e9
            )
            # print("CUT =", cut)
            # print("CUT TYPE =", type(cut))

            # if len(cut) > 0:
            #     first = next(iter(cut))
            #     print("FIRST ELEMENT =", first)
            #     print("FIRST ELEMENT TYPE =", type(first))


            for edge in cut:

                remove_logits[edge] = output["swap_remove_logits"][edge]

            # self._check_logits(
            #         "remove_logits",
            #         remove_logits
            #     )

            # print(
            #         "SWAP REMOVE logits:",
            #         remove_logits.detach(),
            #         flush=True
            #     )

            remove_dist = Categorical(
                logits=remove_logits
            )

            remove_edge = remove_dist.sample()

            # ---------- Add ----------

            add_logits = output["swap_add_logits"].clone()

            # Can't add edges already in cut
            for edge in cut:

                add_logits[edge] = -1e9

            # Can't swap with itself
            add_logits[remove_edge.item()] = -1e9

            # self._check_logits(
            #     "add_logits",
            #     add_logits
            # )

            # print(
            #     "SWAP ADD logits:",
            #     add_logits.detach(),
            #     flush=True
            # )

            add_dist = Categorical(
                logits=add_logits
            )

            add_edge = add_dist.sample()

            log_prob += (
                remove_dist.log_prob(remove_edge)
                + add_dist.log_prob(add_edge)
            )

            entropy += (
                remove_dist.entropy()
                + add_dist.entropy()
            )

            return (
                Action(
                    ActionType.SWAP,
                    remove_edge=remove_edge.item(),
                    add_edge=add_edge.item()
                ),
                log_prob,
                entropy
            )

        # ==================================================
        # STOP
        # ==================================================

        else:

            return (
                Action(
                    ActionType.STOP
                ),
                log_prob,
                entropy
            )

    def evaluate_actions(
        self,
        encoding,
        state,
        action
    ):
        """
        Evaluates a previously selected action.

        Used by PPO during training.

        Returns
        -------
        log_prob : Tensor

        entropy : Tensor
        """

        output = self.forward(encoding)

        cut = state.cut

        num_edges = output["add_logits"].shape[0]

        # ==================================================
        # Operator Mask
        # ==================================================

        operator_logits = output["operator_logits"].clone()

        MIN_STEPS_BEFORE_STOP = 5

        if len(cut) == 0:

            operator_logits[ActionType.REMOVE.value] = -1e9
            operator_logits[ActionType.SWAP.value] = -1e9

        elif len(cut) == num_edges:

            operator_logits[ActionType.ADD.value] = -1e9
            operator_logits[ActionType.SWAP.value] = -1e9

        if state.step < MIN_STEPS_BEFORE_STOP:

            operator_logits[ActionType.STOP.value] = -1e9

        # self._check_logits(
        #         "operator_logits",
        #         operator_logits
        #     )

        operator_dist = Categorical(
            logits=operator_logits
        )

        operator = torch.tensor(
            action.action_type.value,
            device=operator_logits.device
        )

        log_prob = operator_dist.log_prob(operator)

        entropy = operator_dist.entropy()

        # ==================================================
        # ADD
        # ==================================================

        if action.action_type == ActionType.ADD:

            add_logits = output["add_logits"].clone()

            for edge in cut:
                add_logits[edge] = -1e9

            # self._check_logits(
            #     "add_logits",
            #     add_logits
            # )

            edge_dist = Categorical(
                logits=add_logits
            )

            edge = torch.tensor(
                action.edge,
                device=add_logits.device
            )

            log_prob += edge_dist.log_prob(edge)

            entropy += edge_dist.entropy()

        # ==================================================
        # REMOVE
        # ==================================================

        elif action.action_type == ActionType.REMOVE:

            remove_logits = torch.full_like(
                output["remove_logits"],
                -1e9
            )

            for edge in cut:
                remove_logits[edge] = output["remove_logits"][edge]

            # self._check_logits(
            #     "remove_logits",
            #     remove_logits
            # )

            edge_dist = Categorical(
                logits=remove_logits
            )

            edge = torch.tensor(
                action.edge,
                device=remove_logits.device
            )

            log_prob += edge_dist.log_prob(edge)

            entropy += edge_dist.entropy()

        # ==================================================
        # SWAP
        # ==================================================

        elif action.action_type == ActionType.SWAP:

            remove_logits = torch.full_like(
                output["swap_remove_logits"],
                -1e9
            )

            for edge in cut:
                remove_logits[edge] = output["swap_remove_logits"][edge]

            # self._check_logits(
            #     "remove_logits",
            #     remove_logits
            # )

            remove_dist = Categorical(
                logits=remove_logits
            )

            remove_edge = torch.tensor(
                action.remove_edge,
                device=remove_logits.device
            )

            log_prob += remove_dist.log_prob(remove_edge)

            entropy += remove_dist.entropy()

            add_logits = output["swap_add_logits"].clone()

            for edge in cut:
                add_logits[edge] = -1e9

            add_logits[action.remove_edge] = -1e9

            # self._check_logits(
            #         "add_logits",
            #         add_logits
            #     )

            add_dist = Categorical(
                logits=add_logits
            )

            add_edge = torch.tensor(
                action.add_edge,
                device=add_logits.device
            )

            log_prob += add_dist.log_prob(add_edge)

            entropy += add_dist.entropy()

        # ==================================================
        # STOP
        # ==================================================

        return log_prob, entropy
    
    def _check_logits(self, name, logits):

        if torch.isnan(logits).any():

            raise RuntimeError(
                f"{name}: NaN LOGITS\n"
                f"{logits}"
            )

        if torch.isposinf(logits).any():

            raise RuntimeError(
                f"{name}: +INF LOGITS\n"
                f"{logits}"
            )

        finite = torch.isfinite(logits)

        if not finite.any():

            raise RuntimeError(
                f"{name}: ALL LOGITS ARE -INF\n"
                f"shape={tuple(logits.shape)}\n"
                f"logits={logits}"
            )