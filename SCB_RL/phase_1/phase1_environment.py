"""
SCB RL Environment
==================

Curriculum-based environment for the Sparsest Cut Bound
reinforcement-learning project.

CURRENTLY IMPLEMENTED
---------------------
Phase 1:
    Learn to construct low-SCB cuts.

    Available operations:
        ADD
        REMOVE
        SWAP

    STOP is intentionally disabled.

    Episodes terminate automatically after max_steps.

FUTURE PHASES
-------------
Phase 2:
    Learn when to STOP.

Phase 3:
    Learn aggressive SCB minimization.

Phase 4:
    Joint construction + stopping + optimization.

IMPORTANT
---------
The GA evaluator is used to evaluate the cut and obtain the
SCB/separation state.

The GA solution itself is NOT used in the RL reward.

The following teacher values are retained only for:
    - logging
    - evaluation
    - comparison after training

They must never influence PPO reward.
"""


from GA_SCB.graph import SCBProblem
from GA_SCB.graph import evaluate


from .reward import (
    compute_reward,
    compute_reward_components,
    PHASE_CONSTRUCTION,
    PHASE_TERMINATION,
    PHASE_OPTIMIZATION,
    PHASE_JOINT,
)


from .actions import ActionType
from .state import SCBState


# ==========================================================
# CONSTANTS
# ==========================================================

EPSILON = 1e-8


# ==========================================================
# SCB ENVIRONMENT
# ==========================================================

class SCBEnvironment:

    def __init__(
        self,
        graph,
        max_steps=30,
        phase=PHASE_CONSTRUCTION,
    ):
        """
        Create an SCB reinforcement-learning environment.

        Parameters
        ----------
        graph :
            Dataset graph dictionary.

        max_steps :
            Maximum number of environment actions before
            automatic episode termination.

        phase :
            Curriculum phase.

            1 = construction
            2 = termination
            3 = optimization
            4 = joint
        """

        # ==================================================
        # GRAPH
        # ==================================================

        self.graph = graph


        # ==================================================
        # SCB PROBLEM
        # ==================================================

        self.problem = SCBProblem(
            graph["nodes"],
            graph["edges"],
            graph["sessions"],
        )


        # ==================================================
        # GA / TEACHER INFORMATION
        # ==================================================
        #
        # Retained ONLY for evaluation and logging.
        #
        # These values are NEVER passed into the reward.
        #

        self.teacher_scb = graph["ga_scb"]

        self.teacher_cut = graph["ga_cut"]

        self.teacher_sep = graph["ga_sep"]

        self.teacher_cut_edges = (
            graph["ga_cut_edges"]
        )

        self.teacher_components = (
            graph["ga_components"]
        )


        # ==================================================
        # CURRICULUM CONFIGURATION
        # ==================================================

        self.phase = phase

        self.max_steps = int(
            max_steps
        )

        if self.max_steps <= 0:

            raise ValueError(
                "max_steps must be greater than zero."
            )


        # ==================================================
        # RUNTIME STATE
        # ==================================================

        self.current_state = None

        self.best_state = None

        self.episode_step = 0

        self.done = False


        # ==================================================
        # SCB BOOKKEEPING
        # ==================================================

        # Most recent finite SCB.

        self.previous_valid_scb = None


        # Best finite SCB discovered during this episode.

        self.best_scb = None


        # Step at which the best SCB was found.

        self.best_step = None


        # ==================================================
        # REWARD DIAGNOSTICS
        # ==================================================

        self._last_reward_components = {

            "separation":
                0.0,

            "scb":
                0.0,

            "best_bonus":
                0.0,

            "stop":
                0.0,

            "final":
                0.0,

            "time":
                0.0,

            "total":
                0.0,
        }


        # ==================================================
        # ACTION DISPATCH
        # ==================================================

        self._ACTION_MAP = {

            ActionType.ADD:
                self._apply_add,

            ActionType.REMOVE:
                self._apply_remove,

            ActionType.SWAP:
                self._apply_swap,

            # STOP remains in the map because later
            # curriculum phases will use it.
            ActionType.STOP:
                self._apply_stop,
        }


    # ==========================================================
    # PRIVATE HELPERS
    # ==========================================================

    def _is_valid_scb(
        self,
        state,
    ):
        """
        Return True when a state contains a meaningful SCB.
        """

        if state is None:

            return False


        if state.separated_count <= 0:

            return False


        if state.cut_size <= 0:

            return False


        try:

            scb = float(
                state.scb
            )

        except (
            TypeError,
            ValueError,
        ):

            return False


        if scb == float("inf"):

            return False


        if scb != scb:

            return False


        if scb <= EPSILON:

            return False


        return True


    # ==========================================================
    # BUILD STATE
    # ==========================================================

    def _build_state(
        self,
        cut,
    ):
        """
        Convert the RL cut representation into the format
        expected by the SCB evaluator.

        Internally the RL environment represents a cut as:

            integer edge indices

        The evaluator receives:

            edge tuples

        The resulting SCBState stores the evaluated
        representation expected by the rest of the RL system.
        """

        # --------------------------------------------------
        # Normalize cut to integer edge indices.
        # --------------------------------------------------

        normalized_cut = set()


        for edge in cut:

            # ----------------------------------------------
            # Already an integer edge index
            # ----------------------------------------------

            if isinstance(
                edge,
                int,
            ):

                if (
                    edge < 0
                    or edge >= len(self.problem.edges)
                ):

                    raise IndexError(
                        f"Edge index out of range: {edge}"
                    )


                normalized_cut.add(
                    edge
                )

                continue


            # ----------------------------------------------
            # Edge tuple
            # ----------------------------------------------

            if isinstance(
                edge,
                tuple,
            ):

                edge_tuple = tuple(
                    edge
                )


                # Exact orientation.

                if edge_tuple in self.problem.edge_to_idx:

                    normalized_cut.add(
                        self.problem.edge_to_idx[
                            edge_tuple
                        ]
                    )

                    continue


                # Try reverse orientation.

                if len(edge_tuple) == 2:

                    reverse_edge = (
                        edge_tuple[1],
                        edge_tuple[0],
                    )


                    if (
                        reverse_edge
                        in self.problem.edge_to_idx
                    ):

                        normalized_cut.add(
                            self.problem.edge_to_idx[
                                reverse_edge
                            ]
                        )

                        continue


            # ----------------------------------------------
            # Unsupported edge representation
            # ----------------------------------------------

            raise TypeError(
                "Unsupported cut edge "
                f"type/value: {type(edge)} / {edge}"
            )


        # --------------------------------------------------
        # Convert indices -> actual graph edges.
        # --------------------------------------------------

        ga_cut = {

            self.problem.edges[i]

            for i in normalized_cut
        }


        # --------------------------------------------------
        # Evaluate current cut.
        #
        # This is the SCB evaluator.
        #
        # It is NOT a GA search.
        # --------------------------------------------------

        evaluation = evaluate(
            self.problem,
            ga_cut,
        )


        # --------------------------------------------------
        # evaluate() may return cut_edges in tuple form.
        #
        # Normalize them so SCBState continues to expose
        # the expected edge representation.
        # --------------------------------------------------

        if "cut_edges" in evaluation:

            converted_cut_edges = []


            for edge in evaluation["cut_edges"]:

                if isinstance(
                    edge,
                    int,
                ):

                    converted_cut_edges.append(
                        self.problem.edges[edge]
                    )

                else:

                    converted_cut_edges.append(
                        tuple(edge)
                    )


            evaluation["cut_edges"] = (
                converted_cut_edges
            )


        # --------------------------------------------------
        # Build SCB state.
        # --------------------------------------------------

        return SCBState(
            problem=self.problem,
            evaluation=evaluation,
            step=self.episode_step,
        )


    # ==========================================================
    # BEST SOLUTION TRACKING
    # ==========================================================

    def _update_best_state(self):
        """
        Update the best finite SCB discovered during this
        episode.

        Returns
        -------
        bool
            True if a new best solution was discovered.
        """

        current = self.current_state


        # --------------------------------------------------
        # Ignore invalid/non-finite states.
        # --------------------------------------------------

        if not self._is_valid_scb(
            current
        ):

            return False


        # --------------------------------------------------
        # First valid SCB.
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
        # Better SCB.
        # --------------------------------------------------

        if (
            current.scb
            < self.best_scb
        ):

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


    # ==========================================================
    # INFORMATION
    # ==========================================================

    def _get_info(self):
        """
        Return lightweight environment diagnostics.

        GA information is included for evaluation/logging only.
        """

        best_scb = None


        if self.best_scb is not None:

            best_scb = (
                self.best_scb
            )


        return {

            # ------------------------------------------------
            # Curriculum
            # ------------------------------------------------

            "phase":
                self.phase,


            # ------------------------------------------------
            # GA evaluation information
            # ------------------------------------------------

            "teacher_scb":
                self.teacher_scb,


            # ------------------------------------------------
            # Current RL state
            # ------------------------------------------------

            "current_scb":
                self.current_state.scb,

            "current_cut":
                self.current_state.cut_size,

            "current_sep":
                self.current_state.separated_count,


            # ------------------------------------------------
            # Best RL state
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
            # Reward diagnostics
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


    # ==========================================================
    # RESET
    # ==========================================================

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

            "separation":
                0.0,

            "scb":
                0.0,

            "best_bonus":
                0.0,

            "stop":
                0.0,

            "final":
                0.0,

            "time":
                0.0,

            "total":
                0.0,
        }


        # --------------------------------------------------
        # Empty initial cut.
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


    # ==========================================================
    # ACTION VALIDATION
    # ==========================================================

    def _is_valid_action(
        self,
        action,
    ):
        """
        Validate an action against the current cut.

        Phase 1 intentionally rejects STOP.

        STOP remains available in the action enum so that
        Phase 2 can activate it later.
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
                action.edge
                not in cut
            )


        # --------------------------------------------------
        # REMOVE
        # --------------------------------------------------

        if action.is_remove():

            return (
                action.edge
                in cut
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

            # ----------------------------------------------
            # Phase 1:
            #
            # STOP is not part of the action set.
            # ----------------------------------------------

            if (
                self.phase
                == PHASE_CONSTRUCTION
            ):

                return False


            # ----------------------------------------------
            # Future phases.
            # ----------------------------------------------

            return True


        return False


    # ==========================================================
    # ACTION APPLICATION
    # ==========================================================

    def _apply_add(
        self,
        action,
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
        action,
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
        action,
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
        action,
    ):
        """
        STOP does not modify the cut.

        Actual STOP semantics are handled by step().
        """

        return (
            self.current_state.cut.copy()
        )


    # ==========================================================
    # STEP
    # ==========================================================

    def step(
        self,
        action,
    ):
        """
        Execute one environment transition.
        """

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
            #
            # Phase 1 STOP therefore receives the normal
            # invalid-action penalty.
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

                    is_stop=
                        False,

                    new_best=
                        False,

                    phase=
                        self.phase,

                    episode_done=
                        self.episode_step
                        >= self.max_steps,
                )
            )


            reward = (
                self._last_reward_components[
                    "total"
                ]
            )


            # ------------------------------------------------
            # Fixed-horizon termination.
            # ------------------------------------------------

            if (
                self.episode_step
                >= self.max_steps
            ):

                self.done = True


                # Recalculate invalid-action transition
                # with episode_done=True so Phase 1 gets its
                # final best-SCB reward.
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

                        is_stop=
                            False,

                        new_best=
                            False,

                        phase=
                            self.phase,

                        episode_done=
                            True,
                    )
                )


                reward = (
                    self._last_reward_components[
                        "total"
                    ]
                )


                # ------------------------------------------------
                # IMPORTANT:
                #
                # Invalid-action reward intentionally does NOT
                # receive a final best-SCB reward through the
                # invalid branch because reward.py treats an
                # invalid action as an immediate fixed penalty.
                #
                # We therefore add the Phase-1 final reward
                # explicitly below.
                # ------------------------------------------------

                final_components = (
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

                        is_stop=
                            False,

                        new_best=
                            False,

                        phase=
                            self.phase,

                        episode_done=
                            True,
                    )
                )


                final_reward = (
                    final_components.get(
                        "final",
                        0.0
                    )
                )


                self._last_reward_components[
                    "final"
                ] = final_reward


                self._last_reward_components[
                    "total"
                ] += final_reward


                reward = (
                    self._last_reward_components[
                        "total"
                    ]
                )


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
                self._get_info(),
            )


        # ==================================================
        # STOP
        # ==================================================
        #
        # Phase 1 should never reach this branch because
        # STOP is rejected above.
        #
        # This branch exists for Phase 2+.
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

                    is_stop=
                        True,

                    new_best=
                        False,

                    phase=
                        self.phase,

                    episode_done=
                        True,
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
                self._get_info(),
            )


        # ==================================================
        # NORMAL ACTION
        # ==================================================

        old_state = (
            self.current_state.copy()
        )


        # --------------------------------------------------
        # Apply ADD / REMOVE / SWAP.
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


        # ==================================================
        # DETERMINE NEW BEST
        # ==================================================

        old_best_scb = (
            self.best_scb
        )


        found_new_best = False


        if self._is_valid_scb(
            self.current_state
        ):

            if (
                old_best_scb is None
                or self.current_state.scb
                < old_best_scb
            ):

                found_new_best = True


        # ==================================================
        # UPDATE PREVIOUS VALID SCB
        # ==================================================

        if self._is_valid_scb(
            self.current_state
        ):

            self.previous_valid_scb = (
                self.current_state.scb
            )


        # ==================================================
        # MAX-STEP CHECK
        # ==================================================

        reaches_horizon = (
            self.episode_step
            >= self.max_steps
        )


        if reaches_horizon:

            self.done = True


        # ==================================================
        # TRANSITION REWARD
        # ==================================================

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

                is_stop=
                    False,

                new_best=
                    found_new_best,

                phase=
                    self.phase,

                episode_done=
                    reaches_horizon,
            )
        )


        reward = (
            self._last_reward_components[
                "total"
            ]
        )


        # ==================================================
        # UPDATE BEST STATE
        # ==================================================

        self._update_best_state()


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
            self._get_info(),
        )


    # ==========================================================
    # TERMINAL REWARD
    # ==========================================================

    def _compute_terminal_reward(self):
        """
        Return the Phase-1 terminal reward based only on the
        best SCB discovered by the RL agent.

        This method is retained as a compatibility helper for
        existing code/tests.

        GA information is NOT used.
        """

        if (
            self.best_scb is None
        ):

            return -1.0


        try:

            best_scb = float(
                self.best_scb
            )

        except (
            TypeError,
            ValueError,
        ):

            return -1.0


        if (
            best_scb <= EPSILON
            or best_scb == float("inf")
        ):

            return -1.0


        # Same bounded SCB-quality concept used by reward.py.

        quality = (
            1.0
            / (1.0 + best_scb)
        )


        return quality


    # ==========================================================
    # RENDER
    # ==========================================================

    def render(self):

        print(
            self.current_state
        )