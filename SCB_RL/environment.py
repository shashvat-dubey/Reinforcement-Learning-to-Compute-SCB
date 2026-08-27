"""
SCB-RL Phase 2 Environment
==========================

Environment for teaching the agent:

    1. construct a useful cut
    2. obtain separation
    3. recognize when the CURRENT SCB is good enough
    4. STOP

GA information is retained for evaluation/logging only.
It is never passed into the Phase-2 reward.

IMPORTANT:
    There is NO artificial max-step termination in Phase 2.

The episode ends when the agent selects STOP.
"""

from GA_SCB.graph import SCBProblem, evaluate

from .reward import (
    compute_reward_components,
    PHASE_TERMINATION,
)

from .actions import ActionType
from .state import SCBState


EPSILON = 1e-8


class SCBEnvironment:

    def __init__(self, graph, phase=PHASE_TERMINATION):

        self.graph = graph

        self.problem = SCBProblem(
            graph["nodes"],
            graph["edges"],
            graph["sessions"],
        )

        # --------------------------------------------------
        # GA / teacher information
        #
        # Evaluation only. NEVER reward.
        # --------------------------------------------------

        self.teacher_scb = graph.get("ga_scb")
        self.teacher_cut = graph.get("ga_cut")
        self.teacher_sep = graph.get("ga_sep")
        self.teacher_cut_edges = graph.get("ga_cut_edges")
        self.teacher_components = graph.get("ga_components")

        # --------------------------------------------------
        # Curriculum
        # --------------------------------------------------

        self.phase = phase

        if self.phase != PHASE_TERMINATION:
            raise ValueError(
                "This environment file is specifically for Phase 2."
            )

        # --------------------------------------------------
        # Runtime
        # --------------------------------------------------

        self.current_state = None
        self.best_state = None

        self.episode_step = 0
        self.done = False

        self.previous_valid_scb = None
        self.best_scb = None
        self.best_step = None

        self.stop_step = None
        self.stop_was_valid = False

        self._last_reward_components = {
            "separation": 0.0,
            "scb": 0.0,
            "stop": 0.0,
            "time": 0.0,
            "total": 0.0,
        }

        self._ACTION_MAP = {
            ActionType.ADD: self._apply_add,
            ActionType.REMOVE: self._apply_remove,
            ActionType.SWAP: self._apply_swap,
            ActionType.STOP: self._apply_stop,
        }

    # ======================================================
    # State construction
    # ======================================================

    def _build_state(self, cut):

        normalized_cut = set()

        for edge in cut:

            if isinstance(edge, int):
                normalized_cut.add(edge)

            elif isinstance(edge, tuple):

                if edge in self.problem.edge_to_idx:
                    normalized_cut.add(
                        self.problem.edge_to_idx[edge]
                    )

                else:

                    reverse = (
                        edge[1],
                        edge[0],
                    )

                    if reverse in self.problem.edge_to_idx:
                        normalized_cut.add(
                            self.problem.edge_to_idx[reverse]
                        )
                    else:
                        raise ValueError(
                            f"Unknown cut edge: {edge}"
                        )

            else:
                raise TypeError(
                    f"Unsupported cut edge type: {type(edge)}"
                )

        ga_cut = {
            self.problem.edges[i]
            for i in normalized_cut
        }

        evaluation = evaluate(
            self.problem,
            ga_cut,
        )

        return SCBState(
            problem=self.problem,
            evaluation=evaluation,
            step=self.episode_step,
        )

    # ======================================================
    # Best-state tracking
    # ======================================================

    def _update_best_state(self):

        current = self.current_state

        # Empty / invalid state cannot become best.
        if current.separated_count <= 0:
            return False

        if current.cut_size <= 0:
            return False

        if not (
            current.scb > EPSILON
            and current.scb != float("inf")
        ):
            return False

        if self.best_scb is None:

            self.best_state = current.copy()
            self.best_scb = current.scb
            self.best_step = self.episode_step

            return True

        if current.scb < self.best_scb:

            self.best_state = current.copy()
            self.best_scb = current.scb
            self.best_step = self.episode_step

            return True

        return False

    # ======================================================
    # Information
    # ======================================================

    def _get_info(self):

        return {
            # Evaluation only.
            "teacher_scb": self.teacher_scb,

            "current_scb": self.current_state.scb,
            "current_cut": self.current_state.cut_size,
            "current_sep": self.current_state.separated_count,

            "best_scb": self.best_scb,
            "best_step": self.best_step,

            "stop_step": self.stop_step,
            "stop_valid": self.stop_was_valid,

            "reward_components":
                dict(self._last_reward_components),

            "step": self.episode_step,
            "phase": self.phase,
        }

    # ======================================================
    # Reset
    # ======================================================

    def reset(self):

        self.done = False
        self.episode_step = 0

        self.previous_valid_scb = None
        self.best_scb = None
        self.best_step = None

        self.stop_step = None
        self.stop_was_valid = False

        self._last_reward_components = {
            "separation": 0.0,
            "scb": 0.0,
            "stop": 0.0,
            "time": 0.0,
            "total": 0.0,
        }

        empty_cut = set()

        self.current_state = self._build_state(
            empty_cut
        )

        # Empty cut is deliberately NOT a valid solution.
        self.best_state = self.current_state.copy()

        return self.current_state.copy()

    # ======================================================
    # Action validation
    # ======================================================

    def _is_valid_action(self, action):

        action.validate()

        cut = self.current_state.cut

        if action.is_add():
            return action.edge not in cut

        if action.is_remove():
            return action.edge in cut

        if action.is_swap():

            if action.remove_edge not in cut:
                return False

            if action.add_edge in cut:
                return False

            return True

        if action.is_stop():
            return True

        return False

    # ======================================================
    # Action application
    # ======================================================

    def _apply_add(self, action):

        cut = self.current_state.cut.copy()
        cut.add(action.edge)
        return cut

    def _apply_remove(self, action):

        cut = self.current_state.cut.copy()
        cut.remove(action.edge)
        return cut

    def _apply_swap(self, action):

        cut = self.current_state.cut.copy()

        cut.remove(action.remove_edge)
        cut.add(action.add_edge)

        return cut

    def _apply_stop(self, action):

        return self.current_state.cut.copy()

    # ======================================================
    # Step
    # ======================================================

    def step(self, action):

        if self.done:
            raise RuntimeError(
                "Episode already finished. Call reset()."
            )

        # --------------------------------------------------
        # INVALID ACTION
        # --------------------------------------------------

        if not self._is_valid_action(action):

            self.episode_step += 1

            old_state = self.current_state.copy()
            new_state = self.current_state.copy()

            self._last_reward_components = (
                compute_reward_components(
                    old_state=old_state,
                    new_state=new_state,
                    action=action,
                    invalid=True,
                    phase=self.phase,
                )
            )

            reward = self._last_reward_components["total"]

            state = self.current_state.copy()
            state.step = self.episode_step

            return (
                state,
                reward,
                False,
                self._get_info(),
            )

        # --------------------------------------------------
        # STOP
        # --------------------------------------------------

        if action.is_stop():

            self.episode_step += 1
            self.stop_step = self.episode_step
            self.done = True

            state = self.current_state.copy()
            state.step = self.episode_step

            self.stop_was_valid = (
                self.current_state.separated_count > 0
                and self.current_state.cut_size > 0
                and self.current_state.scb != float("inf")
            )

            self._last_reward_components = (
                compute_reward_components(
                    old_state=self.current_state,
                    new_state=self.current_state,
                    action=action,
                    invalid=False,
                    best_scb=self.best_scb,
                    is_stop=True,
                    phase=self.phase,
                    episode_done=True,
                )
            )

            reward = self._last_reward_components["total"]

            return (
                state,
                reward,
                True,
                self._get_info(),
            )

        # --------------------------------------------------
        # NORMAL ACTION
        # --------------------------------------------------

        old_state = self.current_state.copy()

        new_cut = self._ACTION_MAP[
            action.action_type
        ](action)

        self.episode_step += 1

        self.current_state = self._build_state(
            new_cut
        )

        # Reward the current transition.
        self._last_reward_components = (
            compute_reward_components(
                old_state=old_state,
                new_state=self.current_state,
                action=action,
                invalid=False,
                best_scb=self.best_scb,
                is_stop=False,
                phase=self.phase,
                episode_done=False,
            )
        )

        reward = self._last_reward_components["total"]

        # Update best AFTER reward calculation.
        found_new_best = self._update_best_state()

        self._last_reward_components["new_best"] = (
            1.0 if found_new_best else 0.0
        )

        # Keep last valid SCB for diagnostics only.
        if (
            self.current_state.separated_count > 0
            and self.current_state.cut_size > 0
            and self.current_state.scb != float("inf")
            and self.current_state.scb > EPSILON
        ):
            self.previous_valid_scb = self.current_state.scb

        state = self.current_state.copy()
        state.step = self.episode_step

        # --------------------------------------------------
        # NO MAX-STEP TERMINATION
        # --------------------------------------------------
        #
        # Phase 2 ends only when STOP is selected.
        #
        # The step penalty is what creates the time pressure.
        # --------------------------------------------------

        return (
            state,
            reward,
            False,
            self._get_info(),
        )

    # ======================================================
    # Render
    # ======================================================

    def render(self):
        print(self.current_state)
