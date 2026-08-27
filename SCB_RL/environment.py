"""
SCB RL Environment
V2-D

The environment exposes the SCB problem to the RL agent.

GA / teacher information
------------------------

The GA solution is retained ONLY for:

    - evaluation
    - logging
    - post-training comparison

The GA solution NEVER contributes to the PPO reward.

RL reward
---------

Normal transitions use:

    1. SCB-aware separation progress
    2. SCB improvement
    3. New-best SCB bonus
    4. Mild regression penalty
    5. Graph-size-scaled time penalty

STOP uses:

    valid solution
        -> reward based on SCB quality

    invalid solution
        -> strong penalty

Temporary restructuring is allowed:

    good SCB
        ↓
    invalid / inf
        ↓
    better SCB

This is intentional.
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
# Numerical stability
# ==========================================================

EPSILON = 1e-8


# ==========================================================
# Environment
# ==========================================================

class SCBEnvironment:

    def __init__(
        self,
        graph,
        max_steps=100
    ):

        # --------------------------------------------------
        # Original graph
        # --------------------------------------------------

        self.graph = graph


        # --------------------------------------------------
        # SCB problem
        # --------------------------------------------------

        self.problem = SCBProblem(

            graph["nodes"],

            graph["edges"],

            graph["sessions"]

        )


        # --------------------------------------------------
        # GA / teacher information
        #
        # IMPORTANT:
        # These values are NEVER used by reward.py.
        #
        # They remain available for evaluation/logging.
        # --------------------------------------------------

        self.teacher_scb = (
            graph["ga_scb"]
        )

        self.teacher_cut = (
            graph["ga_cut"]
        )

        self.teacher_sep = (
            graph["ga_sep"]
        )

        self.teacher_cut_edges = (
            graph["ga_cut_edges"]
        )

        self.teacher_components = (
            graph["ga_components"]
        )


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
        # SCB bookkeeping
        # --------------------------------------------------

        # Last finite SCB encountered.
        self.previous_valid_scb = None

        # Best finite SCB discovered this episode.
        self.best_scb = None

        # Step where best SCB was found.
        self.best_step = None


        # --------------------------------------------------
        # Reward diagnostics
        # --------------------------------------------------

        self._last_reward_components = {

            "separation": 0.0,

            "scb": 0.0,

            "best_bonus": 0.0,

            "stop": 0.0,

            "time": 0.0,

            "total": 0.0,

        }


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


    # ======================================================
    # Private Helpers
    # ======================================================

    def _build_state(
        self,
        cut
    ):
        """
        Convert RL cut representation into the tuple
        representation expected by the GA evaluator.

        RL representation:
            integer edge indices

        GA representation:
            edge tuples
        """

        # --------------------------------------------------
        # Normalize the RL cut to integer indices.
        #
        # This keeps the internal SCBState representation
        # consistent with the action/policy code.
        # --------------------------------------------------

        normalized_cut = set()

        for edge in cut:

            if isinstance(
                edge,
                int
            ):

                normalized_cut.add(
                    edge
                )

            elif isinstance(
                edge,
                tuple
            ):

                # Edge tuple -> integer index.
                normalized_cut.add(
                    self.problem.edge_to_idx[
                        edge
                    ]
                )

            else:

                raise TypeError(
                    "Unsupported cut edge "
                    f"type: {type(edge)}"
                )


        # --------------------------------------------------
        # Convert indices -> actual graph edge tuples.
        # --------------------------------------------------

        ga_cut = {

            self.problem.edges[i]

            for i in normalized_cut

        }


        # --------------------------------------------------
        # Evaluate SCB.
        #
        # NOTE:
        # This is the SCB evaluator, NOT a GA search.
        # --------------------------------------------------

        evaluation = evaluate(

            self.problem,

            ga_cut

        )


        # --------------------------------------------------
        # Build state.
        # --------------------------------------------------

        return SCBState(

            problem=self.problem,

            evaluation=evaluation,

            step=self.episode_step

        )


    # ======================================================
    # Best Solution Tracking
    # ======================================================

    def _update_best_state(self):
        """
        Update the best finite SCB found so far.

        Returns
        -------
        bool
            True if the current state is a new best.
        """

        current = self.current_state


        # --------------------------------------------------
        # Current state must be a meaningful solution.
        # --------------------------------------------------

        if (
            current.separated_count <= 0
            or current.cut_size <= 0
            or current.scb <= EPSILON
            or current.scb == float("inf")
        ):

            return False


        # --------------------------------------------------
        # First finite solution.
        # --------------------------------------------------

        if self.best_scb is None:

            self.best_scb = (
                current.scb
            )

            self.best_step = (
                self.episode_step
            )

            self.best_state = (
                current.copy()
            )

            return True


        # --------------------------------------------------
        # Improved solution.
        # --------------------------------------------------

        if current.scb < self.best_scb:

            self.best_scb = (
                current.scb
            )

            self.best_step = (
                self.episode_step
            )

            self.best_state = (
                current.copy()
            )

            return True


        return False


    # ======================================================
    # Info
    # ======================================================

    def _get_info(self):
        """
        Return diagnostics used by tests and logging.
        """

        if self.best_state is None:

            best_scb = None

        else:

            best_scb = self.best_scb


        return {

            # ------------------------------------------------
            # Teacher / GA
            # ------------------------------------------------

            "teacher_scb":
                self.teacher_scb,


            # ------------------------------------------------
            # Current RL solution
            # ------------------------------------------------

            "current_scb":
                self.current_state.scb,

            "current_cut":
                self.current_state.cut_size,

            "current_sep":
                self.current_state.separated_count,


            # ------------------------------------------------
            # Best RL solution
            # ------------------------------------------------

            "best_scb":
                best_scb,

            "best_step":
                self.best_step,


            # ------------------------------------------------
            # Cut information
            # ------------------------------------------------

            "cut_size":
                self.current_state.cut_size,

            "sessions_separated":
                self.current_state.separated_count,


            # ------------------------------------------------
            # Reward
            # ------------------------------------------------

            "reward_components":
                self._last_reward_components,


            # ------------------------------------------------
            # Episode
            # ------------------------------------------------

            "step":
                self.episode_step,

            "done":
                self.done,

        }


    # ======================================================
    # Reset
    # ======================================================

    def reset(self):
        """
        Reset the environment to an empty cut.
        """

        self.done = False

        self.episode_step = 0


        # --------------------------------------------------
        # Reset SCB bookkeeping.
        # --------------------------------------------------

        self.previous_valid_scb = None

        self.best_scb = None

        self.best_step = None

        self.best_state = None


        # --------------------------------------------------
        # Reset reward diagnostics.
        # --------------------------------------------------

        self._last_reward_components = {

            "separation": 0.0,

            "scb": 0.0,

            "best_bonus": 0.0,

            "stop": 0.0,

            "time": 0.0,

            "total": 0.0,

        }


        # --------------------------------------------------
        # Empty cut.
        # --------------------------------------------------

        empty_cut = set()


        self.current_state = (
            self._build_state(
                empty_cut
            )
        )


        return (
            self.current_state.copy()
        )


    # ======================================================
    # Action Validation
    # ======================================================

    def _is_valid_action(
        self,
        action
    ):
        """
        Validate an action against the current cut.
        """

        action.validate()


        cut = (
            self.current_state.cut
        )


        # --------------------------------------------------
        # ADD
        # --------------------------------------------------

        if action.is_add():

            return (
                action.edge not in cut
            )


        # --------------------------------------------------
        # REMOVE
        # --------------------------------------------------

        if action.is_remove():

            return (
                action.edge in cut
            )


        # --------------------------------------------------
        # SWAP
        # --------------------------------------------------

        if action.is_swap():

            if (
                action.remove_edge
                not in cut
            ):

                return False


            if (
                action.add_edge
                in cut
            ):

                return False


            return True


        # --------------------------------------------------
        # STOP
        # --------------------------------------------------

        if action.is_stop():

            return True


        return False


    # ======================================================
    # Action Application
    # ======================================================

    def _apply_add(
        self,
        action
    ):

        cut = (
            self.current_state.cut.copy()
        )

        cut.add(
            action.edge
        )

        return cut


    def _apply_remove(
        self,
        action
    ):

        cut = (
            self.current_state.cut.copy()
        )

        cut.remove(
            action.edge
        )

        return cut


    def _apply_swap(
        self,
        action
    ):

        cut = (
            self.current_state.cut.copy()
        )

        cut.remove(
            action.remove_edge
        )

        cut.add(
            action.add_edge
        )

        return cut


    def _apply_stop(
        self,
        action
    ):

        return (
            self.current_state.cut.copy()
        )


    # ======================================================
    # STEP
    # ======================================================

    def step(
        self,
        action
    ):

        if self.done:

            raise RuntimeError(
                "Episode already finished. "
                "Call reset()."
            )


        # ==================================================
        # INVALID ACTION
        # ==================================================

        if not self._is_valid_action(
            action
        ):

            self.episode_step += 1


            # ------------------------------------------------
            # Invalid action reward.
            # ------------------------------------------------

            self._last_reward_components = (
                compute_reward_components(

                    old_state=
                        self.current_state,

                    new_state=
                        self.current_state,

                    action=
                        action,

                    invalid=True,

                    best_scb=
                        self.best_scb,

                    is_stop=False,

                    new_best=False,

                )
            )


            reward = (
                self._last_reward_components[
                    "total"
                ]
            )


            # ------------------------------------------------
            # Max-step termination.
            #
            # NO GA reward.
            # ------------------------------------------------

            if (
                self.episode_step
                >= self.max_steps
            ):

                self.done = True


            state = (
                self.current_state.copy()
            )

            state.step = (
                self.episode_step
            )


            return (

                state,

                reward,

                self.done,

                self._get_info()

            )


        # ==================================================
        # STOP
        # ==================================================

        if action.is_stop():

            self.episode_step += 1

            self.done = True


            state = (
                self.current_state.copy()
            )

            state.step = (
                self.episode_step
            )


            # ------------------------------------------------
            # STOP reward.
            #
            # IMPORTANT:
            # No GA comparison here.
            # ------------------------------------------------

            self._last_reward_components = (
                compute_reward_components(

                    old_state=
                        self.current_state,

                    new_state=
                        self.current_state,

                    action=
                        action,

                    invalid=False,

                    best_scb=
                        self.best_scb,

                    is_stop=True,

                    new_best=False,

                )
            )


            reward = (
                self._last_reward_components[
                    "total"
                ]
            )


            return (

                state,

                reward,

                True,

                self._get_info()

            )


        # ==================================================
        # NORMAL ACTION
        # ==================================================

        old_state = (
            self.current_state.copy()
        )


        # --------------------------------------------------
        # Apply action.
        # --------------------------------------------------

        new_cut = (
            self._ACTION_MAP[
                action.action_type
            ](action)
        )


        # --------------------------------------------------
        # Advance episode.
        # --------------------------------------------------

        self.episode_step += 1


        # --------------------------------------------------
        # Evaluate new state.
        # --------------------------------------------------

        self.current_state = (
            self._build_state(
                new_cut
            )
        )


        # --------------------------------------------------
        # Save best SCB BEFORE updating it.
        #
        # This is the correct reference for determining
        # whether this transition discovered a new best.
        # --------------------------------------------------

        old_best_scb = (
            self.best_scb
        )


        # --------------------------------------------------
        # Determine whether this transition is a new best.
        #
        # We calculate this before updating bookkeeping,
        # then pass the result explicitly to reward.py.
        # --------------------------------------------------

        current = (
            self.current_state
        )


        found_new_best = False


        if (
            current.separated_count > 0
            and current.cut_size > 0
            and current.scb > EPSILON
            and current.scb != float("inf")
        ):

            if (
                old_best_scb is None
                or current.scb < old_best_scb
            ):

                found_new_best = True


        # --------------------------------------------------
        # Calculate transition reward.
        #
        # IMPORTANT:
        # best_scb is the BEST VALUE BEFORE this action.
        # --------------------------------------------------

        self._last_reward_components = (
            compute_reward_components(

                old_state=
                    old_state,

                new_state=
                    self.current_state,

                action=
                    action,

                invalid=False,

                best_scb=
                    old_best_scb,

                is_stop=False,

                new_best=
                    found_new_best,

            )
        )


        reward = (
            self._last_reward_components[
                "total"
            ]
        )


        # --------------------------------------------------
        # NOW update best-state bookkeeping.
        # --------------------------------------------------

        self._update_best_state()


        # --------------------------------------------------
        # Update previous valid SCB.
        # --------------------------------------------------

        if (
            self.current_state.separated_count > 0
            and self.current_state.cut_size > 0
            and self.current_state.scb > EPSILON
            and self.current_state.scb != float("inf")
        ):

            self.previous_valid_scb = (
                self.current_state.scb
            )


        # ==================================================
        # MAX STEP
        # ==================================================

        if (
            self.episode_step
            >= self.max_steps
        ):

            self.done = True


        # --------------------------------------------------
        # IMPORTANT:
        #
        # No GA terminal reward is added.
        #
        # GA remains evaluation-only.
        # --------------------------------------------------


        # ==================================================
        # RETURN
        # ==================================================

        state = (
            self.current_state.copy()
        )

        state.step = (
            self.episode_step
        )


        return (

            state,

            reward,

            self.done,

            self._get_info()

        )


    # ======================================================
    # GA Evaluation Helper
    # ======================================================

    def evaluate_against_ga(self):
        """
        Evaluate the best RL solution against the GA teacher.

        IMPORTANT:

        This function is NOT used by PPO reward calculation.

        It exists solely for testing, logging and final
        comparison.

        Returns
        -------
        dict
        """

        if (
            self.best_state is None
            or self.best_scb is None
        ):

            return {

                "rl_best_scb":
                    None,

                "ga_scb":
                    self.teacher_scb,

                "ratio":
                    None,

            }


        rl_scb = (
            self.best_scb
        )


        ratio = (
            self.teacher_scb
            / max(
                rl_scb,
                EPSILON
            )
        )


        return {

            "rl_best_scb":
                rl_scb,

            "ga_scb":
                self.teacher_scb,

            "ratio":
                ratio,

        }


    # ======================================================
    # Render
    # ======================================================

    def render(self):

        print(
            self.current_state
        )