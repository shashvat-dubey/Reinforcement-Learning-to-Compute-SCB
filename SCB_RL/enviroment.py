"""
SCB RL Environment
V2-C: GA-independent reward environment.

The GA solution is retained ONLY for evaluation/logging.

The RL reward itself never uses:
    - teacher_scb
    - teacher_cut
    - teacher_sep
    - teacher_cut_edges
    - teacher_components

Normal actions are rewarded using:
    1. Session separation progress
    2. SCB improvement
    3. New-best SCB discovery
    4. Graph-size-scaled step penalty

STOP / max-step termination is rewarded using the
best SCB discovered by the agent during the episode.
"""

from GA_SCB.graph import SCBProblem
from GA_SCB.graph import evaluate

from .reward import (
    compute_reward,
    compute_reward_components,
)

from .actions import ActionType
from .state import SCBState


# ==========================================================
# V2-C Hyperparameters
# ==========================================================

# Bonus whenever the agent discovers a genuinely new
# best valid SCB during the episode.
BEST_SCB_BONUS = 0.10

# Weight of terminal solution quality.
TERMINAL_WEIGHT = 1.00

# Invalid action penalty.
INVALID_ACTION_PENALTY = -0.10

# Prevent numerical problems.
EPSILON = 1e-8


class SCBEnvironment:

    def __init__(self, graph, max_steps=100):

        # --------------------------------------------------
        # Original dataset entry
        # --------------------------------------------------

        self.graph = graph

        # --------------------------------------------------
        # Build SCB problem
        # --------------------------------------------------

        self.problem = SCBProblem(
            graph["nodes"],
            graph["edges"],
            graph["sessions"]
        )

        # --------------------------------------------------
        # Teacher / GA information
        #
        # IMPORTANT:
        # These are retained for evaluation and logging.
        # They NEVER participate in RL reward calculation.
        # --------------------------------------------------

        self.teacher_scb = graph["ga_scb"]
        self.teacher_cut = graph["ga_cut"]
        self.teacher_sep = graph["ga_sep"]
        self.teacher_cut_edges = graph["ga_cut_edges"]
        self.teacher_components = graph["ga_components"]

        # --------------------------------------------------
        # Environment configuration
        # --------------------------------------------------

        self.max_steps = max_steps

        # --------------------------------------------------
        # Runtime state
        # --------------------------------------------------

        self.current_state = None
        self.best_state = None

        self.episode_step = 0
        self.done = False

        # --------------------------------------------------
        # SCB tracking
        # --------------------------------------------------

        # Last valid SCB encountered.
        #
        # None is important because the empty cut has SCB=0,
        # which is NOT a meaningful SCB solution.
        self.previous_valid_scb = None

        # Best valid SCB found during this episode.
        self.best_scb = None

        # Step at which best SCB was found.
        self.best_step = None

        # --------------------------------------------------
        # Action dispatch
        # --------------------------------------------------

        self._ACTION_MAP = {

            ActionType.ADD:
                self._apply_add,

            ActionType.REMOVE:
                self._apply_remove,

            ActionType.SWAP:
                self._apply_swap,

            ActionType.STOP:
                self._apply_stop,
        }

    # ==========================================================
    # Private Helpers
    # ==========================================================

    def _build_state(self, cut):

        # --------------------------------------------------
        # Normalize cut -> integer edge indices
        #
        # The environment may currently contain either:
        #
        #   0
        #   17
        #   42
        #
        # OR:
        #
        #   ("v1", "v10")
        #
        # --------------------------------------------------

        edge_to_index = {
            tuple(edge): i
            for i, edge in enumerate(self.problem.edges)
        }

        cut_indices = set()

        for edge in cut:

            # ----------------------------------------------
            # Already an edge index
            # ----------------------------------------------

            if isinstance(edge, int):

                cut_indices.add(edge)

                continue

            # ----------------------------------------------
            # Edge tuple
            # ----------------------------------------------

            edge_tuple = tuple(edge)

            if edge_tuple in edge_to_index:

                cut_indices.add(
                    edge_to_index[edge_tuple]
                )

                continue

            # ----------------------------------------------
            # Undirected reverse orientation
            # ----------------------------------------------

            reverse_edge = (
                edge_tuple[1],
                edge_tuple[0]
            )

            if reverse_edge in edge_to_index:

                cut_indices.add(
                    edge_to_index[reverse_edge]
                )

                continue

            # ----------------------------------------------
            # Unknown edge
            # ----------------------------------------------

            raise ValueError(
                f"Unknown cut edge: {edge}"
            )

        # --------------------------------------------------
        # GA evaluation
        # --------------------------------------------------

        evaluation = evaluate(
            self.problem,
            cut_indices
        )

        # --------------------------------------------------
        # Convert GA cut_edges back into actual edges
        #
        # SCBState expects edge tuples.
        # --------------------------------------------------

        evaluation["cut_edges"] = [
            self.problem.edges[i]
            for i in evaluation["cut_edges"]
        ]

        # --------------------------------------------------
        # Build state
        # --------------------------------------------------

        return SCBState(
            problem=self.problem,
            evaluation=evaluation,
            step=self.episode_step
        )

    # ==========================================================
    # Best-State Tracking
    # ==========================================================

    def _update_best_state(self):

        current = self.current_state
        best = self.best_state

        # --------------------------------------------------
        # Current state is not a valid SCB solution.
        # --------------------------------------------------

        if current.separated_count <= 0:
            return False

        if current.cut_size <= 0:
            return False

        if current.scb <= EPSILON:
            return False

        # --------------------------------------------------
        # First valid solution
        # --------------------------------------------------

        if (
            best is None
            or best.separated_count <= 0
            or best.cut_size <= 0
            or best.scb <= EPSILON
        ):

            self.best_state = current.copy()

            self.best_scb = current.scb
            self.best_step = self.episode_step

            return True

        # --------------------------------------------------
        # New best SCB
        # --------------------------------------------------

        if current.scb < best.scb:

            self.best_state = current.copy()

            self.best_scb = current.scb
            self.best_step = self.episode_step

            return True

        return False

    # ==========================================================
    # Information
    # ==========================================================

    def _get_info(self):

        best_scb = None

        if self.best_state is not None:

            if (
                self.best_state.separated_count > 0
                and self.best_state.cut_size > 0
                and self.best_state.scb > EPSILON
            ):
                best_scb = self.best_state.scb

        return {

            # --------------------------------------------------
            # GA information
            #
            # Evaluation ONLY.
            # --------------------------------------------------

            "teacher_scb":
                self.teacher_scb,

            # --------------------------------------------------
            # Current RL state
            # --------------------------------------------------

            "current_scb":
                self.current_state.scb,

            "current_cut":
                self.current_state.cut_size,

            "current_sep":
                self.current_state.separated_count,

            # --------------------------------------------------
            # Best RL state
            # --------------------------------------------------

            "best_scb":
                best_scb,

            "best_step":
                self.best_step,

            "cut_size":
                self.current_state.cut_size,

            "sessions_separated":
                self.current_state.separated_count,

            # --------------------------------------------------
            # Reward diagnostics
            # --------------------------------------------------

            "reward_components":
                getattr(
                    self,
                    "_last_reward_components",
                    {
                        "separation": 0.0,
                        "scb": 0.0,
                        "best": 0.0,
                        "time": 0.0,
                        "terminal": 0.0,
                        "total": 0.0,
                    }
                ),

            "step":
                self.episode_step
        }

    # ==========================================================
    # Reset
    # ==========================================================

    def reset(self):

        self.done = False

        self.episode_step = 0

        # --------------------------------------------------
        # Reset SCB tracking
        # --------------------------------------------------

        self.previous_valid_scb = None

        self.best_scb = None

        self.best_step = None

        # --------------------------------------------------
        # Reset reward diagnostics
        # --------------------------------------------------

        self._last_reward_components = {

            "separation": 0.0,

            "scb": 0.0,

            "best": 0.0,

            "time": 0.0,

            "terminal": 0.0,

            "total": 0.0,
        }

        # --------------------------------------------------
        # Empty initial cut
        # --------------------------------------------------

        empty_cut = set()

        self.current_state = self._build_state(
            empty_cut
        )

        # --------------------------------------------------
        # IMPORTANT:
        #
        # Empty cut has SCB=0, but it is NOT a valid
        # solution, so it must NOT become best_scb.
        # --------------------------------------------------

        self.best_state = self.current_state.copy()

        return self.current_state.copy()

    # ==========================================================
    # Action Validation
    # ==========================================================

    def _is_valid_action(self, action):

        action.validate()

        cut = self.current_state.cut

        # --------------------------------------------------
        # ADD
        # --------------------------------------------------

        if action.is_add():

            return action.edge not in cut

        # --------------------------------------------------
        # REMOVE
        # --------------------------------------------------

        if action.is_remove():

            return action.edge in cut

        # --------------------------------------------------
        # SWAP
        # --------------------------------------------------

        if action.is_swap():

            if action.remove_edge not in cut:

                return False

            if action.add_edge in cut:

                return False

            return True

        # --------------------------------------------------
        # STOP
        # --------------------------------------------------

        return True

    # ==========================================================
    # Action Application
    # ==========================================================

    def _apply_add(self, action):

        cut = self.current_state.cut.copy()

        cut.add(action.edge)

        return cut

    # ----------------------------------------------------------

    def _apply_remove(self, action):

        cut = self.current_state.cut.copy()

        cut.remove(action.edge)

        return cut

    # ----------------------------------------------------------

    def _apply_swap(self, action):

        cut = self.current_state.cut.copy()

        cut.remove(action.remove_edge)

        cut.add(action.add_edge)

        return cut

    # ----------------------------------------------------------

    def _apply_stop(self, action):

        self.done = True

        return self.current_state.cut.copy()

    # ==========================================================
    # Step
    # ==========================================================

    def step(self, action):

        if self.done:

            raise RuntimeError(
                "Episode already finished. Call reset()."
            )

        # ======================================================
        # INVALID ACTION
        # ======================================================

        if not self._is_valid_action(action):

            self.episode_step += 1

            # --------------------------------------------------
            # Max-step termination
            # --------------------------------------------------

            if self.episode_step >= self.max_steps:

                self.done = True

            self._last_reward_components = (
                compute_reward_components(
                    self.current_state,
                    self.current_state,
                    action,
                    invalid=True,
                )
            )

            # Make sure the component dictionary has the
            # V2-C best field.
            self._last_reward_components.setdefault(
                "best",
                0.0
            )

            reward = (
                self._last_reward_components["total"]
            )

            # --------------------------------------------------
            # If invalid action caused max-step termination,
            # evaluate the best solution found so far.
            # --------------------------------------------------

            if self.done:

                terminal_reward = (
                    self._compute_terminal_reward()
                )

                self._last_reward_components[
                    "terminal"
                ] = terminal_reward

                self._last_reward_components[
                    "total"
                ] += terminal_reward

                reward = (
                    self._last_reward_components["total"]
                )

            state = self.current_state.copy()

            state.step = self.episode_step

            return (
                state,
                reward,
                self.done,
                self._get_info()
            )

        # ======================================================
        # STOP
        # ======================================================

        if action.is_stop():

            self.episode_step += 1

            self.done = True

            state = self.current_state.copy()

            state.step = self.episode_step

            # --------------------------------------------------
            # Terminal reward uses BEST RL solution.
            #
            # NO GA.
            # --------------------------------------------------

            terminal_reward = (
                self._compute_terminal_reward()
            )

            self._last_reward_components = {

                "separation":
                    0.0,

                "scb":
                    0.0,

                "best":
                    0.0,

                "time":
                    0.0,

                "terminal":
                    terminal_reward,

                "total":
                    terminal_reward,
            }

            return (
                state,
                terminal_reward,
                True,
                self._get_info()
            )

        # ======================================================
        # NORMAL ACTION
        # ======================================================

        new_cut = self._ACTION_MAP[
            action.action_type
        ](action)

        old_state = self.current_state.copy()

        self.episode_step += 1

        self.current_state = self._build_state(
            new_cut
        )

        # --------------------------------------------------
        # Calculate normal reward
        # --------------------------------------------------

        self._last_reward_components = (
            compute_reward_components(
                old_state,
                self.current_state,
                action,
            )
        )

        self._last_reward_components.setdefault(
            "best",
            0.0
        )

        reward = (
            self._last_reward_components["total"]
        )

        # ==================================================
        # BEST-SOLUTION TRACKING
        # ==================================================

        old_best_scb = self.best_scb

        found_new_best = (
            self._update_best_state()
        )

        # --------------------------------------------------
        # New-best bonus
        # --------------------------------------------------

        best_bonus = 0.0

        if found_new_best:

            best_bonus = BEST_SCB_BONUS

            self._last_reward_components[
                "best"
            ] = best_bonus

            self._last_reward_components[
                "total"
            ] += best_bonus

            reward += best_bonus

        # --------------------------------------------------
        # Update previous valid SCB
        #
        # Only valid SCB values participate in SCB
        # progression tracking.
        # --------------------------------------------------

        if (
            self.current_state.separated_count > 0
            and self.current_state.cut_size > 0
            and self.current_state.scb > EPSILON
        ):

            self.previous_valid_scb = (
                self.current_state.scb
            )

        # ==================================================
        # MAX STEP
        # ==================================================

        if self.episode_step >= self.max_steps:

            self.done = True

        # ==================================================
        # TERMINAL REWARD
        # ==================================================

        if self.done:

            terminal_reward = (
                self._compute_terminal_reward()
            )

            self._last_reward_components[
                "terminal"
            ] = terminal_reward

            self._last_reward_components[
                "total"
            ] += terminal_reward

            reward = (
                self._last_reward_components["total"]
            )

        # ==================================================
        # RETURN STATE
        # ==================================================

        state = self.current_state.copy()

        state.step = self.episode_step

        return (
            state,
            reward,
            self.done,
            self._get_info()
        )

    # ==========================================================
    # Terminal Reward
    # ==========================================================

    def _compute_terminal_reward(self):
        """
        GA-INDEPENDENT terminal reward.

        The reward depends ONLY on the best valid SCB
        discovered by the RL agent.

        Lower SCB -> higher terminal reward.

        GA/teacher information is intentionally NOT used.
        """

        # --------------------------------------------------
        # No valid solution discovered
        # --------------------------------------------------

        if (
            self.best_state is None
            or self.best_state.separated_count <= 0
            or self.best_state.cut_size <= 0
            or self.best_state.scb <= EPSILON
        ):

            return -1.0

        # --------------------------------------------------
        # Best SCB discovered by RL
        # --------------------------------------------------

        best_scb = self.best_state.scb

        # --------------------------------------------------
        # Convert SCB into bounded quality.
        #
        # SCB -> 0
        #       quality -> 1
        #
        # SCB -> infinity
        #       quality -> 0
        #
        # This uses ONLY the SCB objective itself.
        # --------------------------------------------------

        quality = (
            1.0
            / (1.0 + best_scb)
        )

        terminal_reward = (
            TERMINAL_WEIGHT
            * quality
        )

        # --------------------------------------------------
        # Small efficiency penalty.
        #
        # Larger graphs naturally receive larger budgets.
        # --------------------------------------------------

        step_budget = max(
            0.25 * len(self.problem.edges)
            + 2.0 * len(self.problem.sessions),
            1.0
        )

        efficiency_penalty = (
            0.05
            * self.episode_step
            / step_budget
        )

        efficiency_penalty = min(
            efficiency_penalty,
            0.25
        )

        terminal_reward -= efficiency_penalty

        # --------------------------------------------------
        # Bound terminal reward
        # --------------------------------------------------

        terminal_reward = max(
            -1.0,
            min(terminal_reward, 1.0)
        )

        return terminal_reward

    # ==========================================================
    # Render
    # ==========================================================

    def render(self):

        print(self.current_state)