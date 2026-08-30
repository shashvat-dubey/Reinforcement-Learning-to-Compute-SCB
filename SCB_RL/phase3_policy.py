import torch
import torch.nn as nn
from torch.distributions import Categorical

from SCB_RL.actions import Action, ActionType


class HierarchicalSCBPolicy(nn.Module):
    """
    Phase-3 Hierarchical Actor Network.

    High-Level Policy
        ADD
        REMOVE
        SWAP
        STOP

    Low-Level Policies
        ADD    -> select an edge
        REMOVE -> select an edge from the cut
        SWAP   -> select one edge to remove and one edge to add
        STOP   -> terminate the episode

    Phase-3 design:
        Controlled STOP exploration.

    The STOP exploration is injected into the operator distribution
    BEFORE sampling, so the sampled action and its log-probability
    remain consistent for PPO.

    Phase-3 objective:
        Starting from the final Phase-2 cut, learn to choose
        ADD / REMOVE / SWAP operations that reduce SCB, then STOP.

    This policy is trained from scratch for Phase 3.  It does not
    inherit Phase-1/Phase-2 policy weights unless the trainer
    explicitly chooses to load them.

    stop_exploration = 0.0
        Normal policy.

    stop_exploration = 0.10
        10% exploration mass is assigned to STOP while preserving
        the remaining 90% of the original policy distribution.

    IMPORTANT:
        STOP is still forbidden before MIN_STEPS_BEFORE_STOP.
        In that case no exploration mass is injected.
    """

    def __init__(
        self,
        graph_dim=64,
        edge_dim=129,
        hidden_dim=64,
        stop_exploration=0.01,
        min_steps_before_stop=5,
    ):
        super().__init__()

        self.graph_dim = graph_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        # -----------------------------------------------------
        # Phase-2 STOP exploration configuration
        # -----------------------------------------------------

        self.stop_exploration = float(stop_exploration)
        self.min_steps_before_stop = int(
            min_steps_before_stop
        )

        if not 0.0 <= self.stop_exploration < 1.0:
            raise ValueError(
                "stop_exploration must be in [0, 1)."
            )

        # =====================================================
        # Shared Edge Encoder
        # =====================================================

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # =====================================================
        # Operator Head
        # =====================================================

        self.operator_head = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

        # =====================================================
        # Edge Heads
        # =====================================================

        self.add_head = self._make_edge_head()
        self.remove_head = self._make_edge_head()
        self.swap_remove_head = self._make_edge_head()
        self.swap_add_head = self._make_edge_head()

    # =========================================================
    # Configuration
    # =========================================================

    def set_stop_exploration(self, probability):
        """
        Update the Phase-3 STOP exploration probability.

        This can be called by the Phase-2 trainer as exploration
        decays over training.
        """

        probability = float(probability)

        if not 0.0 <= probability < 1.0:
            raise ValueError(
                "stop exploration must be in [0, 1)."
            )

        self.stop_exploration = probability

    # =========================================================
    # Edge Head
    # =========================================================

    def _make_edge_head(self):
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    # =========================================================
    # Forward
    # =========================================================

    def forward(self, encoding):
        """
        Parameters
        ----------
        encoding : dict
            Output of SCBGraphEncoder.

        Returns
        -------
        dict
            operator_logits
            add_logits
            remove_logits
            swap_remove_logits
            swap_add_logits
        """

        graph_embedding = encoding["graph_embedding"]
        edge_embeddings = encoding["edge_embeddings"]

        # -----------------------------------------------------
        # Operator
        # -----------------------------------------------------

        operator_logits = self.operator_head(
            graph_embedding
        )

        # -----------------------------------------------------
        # Shared Edge Representation
        # -----------------------------------------------------

        edge_features = self.edge_encoder(
            edge_embeddings
        )

        # -----------------------------------------------------
        # Edge Heads
        # -----------------------------------------------------

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
            "swap_add_logits": swap_add_logits,
        }

    # =========================================================
    # Operator Distribution
    # =========================================================

    def _get_operator_distribution(
        self,
        operator_logits,
        state,
    ):
        """
        Build the operator distribution used by BOTH:

            sample_action()
            evaluate_actions()

        This is important for PPO because the log-probability
        must correspond to the same distribution that produced
        the action.

        STOP exploration is applied only when STOP is valid.
        """

        logits = operator_logits.clone()

        cut = state.cut
        num_edges = operator_logits.shape[-1]

        # -----------------------------------------------------
        # Validity masks
        # -----------------------------------------------------

        if len(cut) == 0:

            logits[ActionType.REMOVE.value] = -1e9
            logits[ActionType.SWAP.value] = -1e9

        elif len(cut) == self._num_edges_from_state(
            state,
            default=None,
        ):
            # This branch is normally handled more reliably by
            # the caller, because operator logits have size 4,
            # not number of graph edges.
            pass

        # -----------------------------------------------------
        # Minimum steps before STOP
        # -----------------------------------------------------

        stop_allowed = (
            state.step >= self.min_steps_before_stop
        )

        if not stop_allowed:
            logits[ActionType.STOP.value] = -1e9

        # -----------------------------------------------------
        # Build normal masked distribution
        # -----------------------------------------------------

        probs = torch.softmax(
            logits,
            dim=-1,
        )

        # -----------------------------------------------------
        # Controlled STOP exploration
        # -----------------------------------------------------

        if (
            stop_allowed
            and self.stop_exploration > 0.0
        ):
            epsilon = self.stop_exploration

            # Keep the original masked policy distribution,
            # then reserve epsilon mass specifically for STOP.
            probs = probs * (1.0 - epsilon)

            probs[
                ActionType.STOP.value
            ] += epsilon

        # -----------------------------------------------------
        # Numerical normalization
        # -----------------------------------------------------

        probs = probs / probs.sum(
            dim=-1,
            keepdim=True,
        )

        return Categorical(
            probs=probs
        )

    @staticmethod
    def _num_edges_from_state(
        state,
        default=None,
    ):
        """
        Kept only as a defensive helper.

        Full-cut validity is handled in sample_action() and
        evaluate_actions(), where the number of edges is known.
        """

        return default

    def _operator_distribution(
        self,
        operator_logits,
        state,
        num_edges,
    ):
        """
        Actual shared operator distribution.

        Kept separate so both sampling and PPO evaluation use
        exactly the same masking + STOP exploration logic.
        """

        logits = operator_logits.clone()

        cut = state.cut

        # -----------------------------------------------------
        # Empty cut
        # -----------------------------------------------------

        if len(cut) == 0:

            logits[
                ActionType.REMOVE.value
            ] = -1e9

            logits[
                ActionType.SWAP.value
            ] = -1e9

        # -----------------------------------------------------
        # Full cut
        # -----------------------------------------------------

        elif len(cut) == num_edges:

            logits[
                ActionType.ADD.value
            ] = -1e9

            logits[
                ActionType.SWAP.value
            ] = -1e9

        # -----------------------------------------------------
        # STOP validity
        # -----------------------------------------------------

        stop_allowed = (
            state.step >= self.min_steps_before_stop
        )

        if not stop_allowed:

            logits[
                ActionType.STOP.value
            ] = -1e9

        # -----------------------------------------------------
        # Normal masked probabilities
        # -----------------------------------------------------

        probs = torch.softmax(
            logits,
            dim=-1,
        )

        # -----------------------------------------------------
        # STOP exploration
        # -----------------------------------------------------

        if (
            stop_allowed
            and self.stop_exploration > 0.0
        ):

            epsilon = self.stop_exploration

            probs = probs * (
                1.0 - epsilon
            )

            probs[
                ActionType.STOP.value
            ] += epsilon

        # -----------------------------------------------------
        # Normalize
        # -----------------------------------------------------

        probs = probs / probs.sum(
            dim=-1,
            keepdim=True,
        )

        return Categorical(
            probs=probs
        )

    # =========================================================
    # Sample Action
    # =========================================================

    # =========================================================
    # Phase-3 configuration helpers
    # =========================================================

    def set_stop_exploration(self, probability):
        """
        Set the externally injected STOP exploration mass.

        Kept explicit so the Phase-3 trainer can control exploration
        without modifying the PPO sampling/log-probability logic.
        """

        probability = float(probability)

        if not 0.0 <= probability < 1.0:
            raise ValueError(
                "stop exploration must be in [0, 1)."
            )

        self.stop_exploration = probability

    def get_action_probabilities(self, encoding, state=None):
        """
        Return the operator probabilities after Phase-3 STOP
        exploration is applied.

        Primarily intended for diagnostics/logging.
        """

        outputs = self.forward(encoding)

        logits = outputs["operator_logits"]

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        if (
            state is not None
            and getattr(state, "step", 0) <
            self.min_steps_before_stop
        ):
            probabilities = probabilities.clone()
            probabilities[..., ActionType.STOP] = 0.0

            total = probabilities.sum(
                dim=-1,
                keepdim=True
            )

            probabilities = probabilities / (
                total + 1e-8
            )

        return probabilities



    def sample_action(
        self,
        encoding,
        state,
    ):
        """
        Samples one valid action.

        Returns
        -------
        Action
        Tensor
            Total log probability.
        Tensor
            Total entropy.
        """

        output = self.forward(
            encoding
        )

        cut = state.cut

        num_edges = output[
            "add_logits"
        ].shape[0]

        # -----------------------------------------------------
        # Operator
        # -----------------------------------------------------

        operator_dist = self._operator_distribution(
            output["operator_logits"],
            state,
            num_edges,
        )

        operator = operator_dist.sample()

        log_prob = operator_dist.log_prob(
            operator
        )

        entropy = operator_dist.entropy()

        # =====================================================
        # ADD
        # =====================================================

        if (
            operator.item()
            == ActionType.ADD.value
        ):

            add_logits = output[
                "add_logits"
            ].clone()

            for edge in cut:
                add_logits[edge] = -1e9

            edge_dist = Categorical(
                logits=add_logits
            )

            edge = edge_dist.sample()

            log_prob += edge_dist.log_prob(
                edge
            )

            entropy += edge_dist.entropy()

            return (
                Action(
                    ActionType.ADD,
                    edge=edge.item(),
                ),
                log_prob,
                entropy,
            )

        # =====================================================
        # REMOVE
        # =====================================================

        elif (
            operator.item()
            == ActionType.REMOVE.value
        ):

            remove_logits = torch.full_like(
                output["remove_logits"],
                -1e9,
            )

            for edge in cut:
                remove_logits[edge] = (
                    output["remove_logits"][edge]
                )

            edge_dist = Categorical(
                logits=remove_logits
            )

            edge = edge_dist.sample()

            log_prob += edge_dist.log_prob(
                edge
            )

            entropy += edge_dist.entropy()

            return (
                Action(
                    ActionType.REMOVE,
                    edge=edge.item(),
                ),
                log_prob,
                entropy,
            )

        # =====================================================
        # SWAP
        # =====================================================

        elif (
            operator.item()
            == ActionType.SWAP.value
        ):

            # -------------------------------------------------
            # Remove
            # -------------------------------------------------

            remove_logits = torch.full_like(
                output["swap_remove_logits"],
                -1e9,
            )

            for edge in cut:
                remove_logits[edge] = (
                    output["swap_remove_logits"][edge]
                )

            remove_dist = Categorical(
                logits=remove_logits
            )

            remove_edge = remove_dist.sample()

            # -------------------------------------------------
            # Add
            # -------------------------------------------------

            add_logits = output[
                "swap_add_logits"
            ].clone()

            for edge in cut:
                add_logits[edge] = -1e9

            add_logits[
                remove_edge.item()
            ] = -1e9

            add_dist = Categorical(
                logits=add_logits
            )

            add_edge = add_dist.sample()

            log_prob += (
                remove_dist.log_prob(
                    remove_edge
                )
                + add_dist.log_prob(
                    add_edge
                )
            )

            entropy += (
                remove_dist.entropy()
                + add_dist.entropy()
            )

            return (
                Action(
                    ActionType.SWAP,
                    remove_edge=remove_edge.item(),
                    add_edge=add_edge.item(),
                ),
                log_prob,
                entropy,
            )

        # =====================================================
        # STOP
        # =====================================================

        return (
            Action(
                ActionType.STOP
            ),
            log_prob,
            entropy,
        )

    # =========================================================
    # Evaluate Action
    # =========================================================

    def evaluate_actions(
        self,
        encoding,
        state,
        action,
    ):
        """
        Evaluate a previously selected action.

        IMPORTANT:
            Uses the exact same operator distribution logic
            as sample_action(), including Phase-2 STOP
            exploration.

        Returns
        -------
        log_prob : Tensor
        entropy : Tensor
        """

        output = self.forward(
            encoding
        )

        cut = state.cut

        num_edges = output[
            "add_logits"
        ].shape[0]

        # -----------------------------------------------------
        # Operator
        # -----------------------------------------------------

        operator_dist = self._operator_distribution(
            output["operator_logits"],
            state,
            num_edges,
        )

        operator = torch.tensor(
            action.action_type.value,
            device=output[
                "operator_logits"
            ].device,
        )

        log_prob = operator_dist.log_prob(
            operator
        )

        entropy = operator_dist.entropy()

        # =====================================================
        # ADD
        # =====================================================

        if (
            action.action_type
            == ActionType.ADD
        ):

            add_logits = output[
                "add_logits"
            ].clone()

            for edge in cut:
                add_logits[edge] = -1e9

            edge_dist = Categorical(
                logits=add_logits
            )

            edge = torch.tensor(
                action.edge,
                device=add_logits.device,
            )

            log_prob += edge_dist.log_prob(
                edge
            )

            entropy += edge_dist.entropy()

        # =====================================================
        # REMOVE
        # =====================================================

        elif (
            action.action_type
            == ActionType.REMOVE
        ):

            remove_logits = torch.full_like(
                output["remove_logits"],
                -1e9,
            )

            for edge in cut:
                remove_logits[edge] = (
                    output["remove_logits"][edge]
                )

            edge_dist = Categorical(
                logits=remove_logits
            )

            edge = torch.tensor(
                action.edge,
                device=remove_logits.device,
            )

            log_prob += edge_dist.log_prob(
                edge
            )

            entropy += edge_dist.entropy()

        # =====================================================
        # SWAP
        # =====================================================

        elif (
            action.action_type
            == ActionType.SWAP
        ):

            remove_logits = torch.full_like(
                output["swap_remove_logits"],
                -1e9,
            )

            for edge in cut:
                remove_logits[edge] = (
                    output["swap_remove_logits"][edge]
                )

            remove_dist = Categorical(
                logits=remove_logits
            )

            remove_edge = torch.tensor(
                action.remove_edge,
                device=remove_logits.device,
            )

            log_prob += remove_dist.log_prob(
                remove_edge
            )

            entropy += remove_dist.entropy()

            add_logits = output[
                "swap_add_logits"
            ].clone()

            for edge in cut:
                add_logits[edge] = -1e9

            add_logits[
                action.remove_edge
            ] = -1e9

            add_dist = Categorical(
                logits=add_logits
            )

            add_edge = torch.tensor(
                action.add_edge,
                device=add_logits.device,
            )

            log_prob += add_dist.log_prob(
                add_edge
            )

            entropy += add_dist.entropy()

        # =====================================================
        # STOP
        # =====================================================

        return log_prob, entropy

    # =========================================================
    # Logit Diagnostics
    # =========================================================

    def _check_logits(
        self,
        name,
        logits,
    ):

        if torch.isnan(
            logits
        ).any():

            raise RuntimeError(
                f"{name}: NaN LOGITS\n"
                f"{logits}"
            )

        if torch.isposinf(
            logits
        ).any():

            raise RuntimeError(
                f"{name}: +INF LOGITS\n"
                f"{logits}"
            )

        finite = torch.isfinite(
            logits
        )

        if not finite.any():

            raise RuntimeError(
                f"{name}: ALL LOGITS ARE -INF\n"
                f"shape={tuple(logits.shape)}\n"
                f"logits={logits}"
            )
